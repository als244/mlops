"""mHC preprocessing with TileKernels' fused GLM-compatible Sinkhorn VJP.

The full preprocessing boundary allows later providers to fuse projection,
gates and collapse too. Only the small Sinkhorn state is opaque here.
"""

from importlib.util import find_spec

import torch
from torch.nn import functional as F

from ...dispatch import logical_costs as costs
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ..native_torch.hyper_connection import supports_combine, supports_pre


@torch.compiler.assume_constant_result
def _available():
    if find_spec("tile_kernels") is None:
        return False
    from importlib.metadata import version

    parts = version("tilelang").split(".")
    return tuple(int(x) for x in parts[:3]) >= (0, 1, 15)


@torch.library.custom_op("mlops::mhc_sinkhorn_tilekernels", mutates_args=())
def _sinkhorn(logits: torch.Tensor, iterations: int, eps: float) -> torch.Tensor:
    from tile_kernels.mhc import mhc_sinkhorn_fwd

    out = torch.empty_like(logits)
    with torch.cuda.device(logits.device):
        mhc_sinkhorn_fwd(logits, out, iterations, eps)
    return out


@_sinkhorn.register_fake
def _sinkhorn_fake(logits, iterations, eps):
    return torch.empty_like(logits)


@torch.library.custom_op("mlops::mhc_sinkhorn_tilekernels_backward", mutates_args=())
def _sinkhorn_backward(
    grad: torch.Tensor, logits: torch.Tensor, iterations: int, eps: float
) -> torch.Tensor:
    from tile_kernels.mhc import mhc_sinkhorn_bwd

    dx = torch.empty_like(logits)
    with torch.cuda.device(logits.device):
        mhc_sinkhorn_bwd(grad.contiguous(), logits, dx, iterations, eps)
    return dx


@_sinkhorn_backward.register_fake
def _sinkhorn_backward_fake(grad, logits, iterations, eps):
    return torch.empty_like(logits)


def _setup(ctx, inputs, output):
    logits, ctx.iterations, ctx.eps = inputs
    ctx.save_for_backward(logits)


def _backward(ctx, grad):
    (logits,) = ctx.saved_tensors
    return _sinkhorn_backward(grad, logits, ctx.iterations, ctx.eps), None, None


_sinkhorn.register_autograd(_backward, setup_context=_setup)


@flop_formula(_sinkhorn)
def _fwd_flops(logits, iterations, eps, **_kwargs):
    return costs.mhc_sinkhorn_flops(logits, iterations)


@flop_formula(_sinkhorn_backward)
def _bwd_flops(grad, logits, iterations, eps, **_kwargs):
    return costs.mhc_sinkhorn_flops(logits, iterations, backward=True, recompute=True)


def _supports(streams, projection, bias, scale, *, surface, **kwargs):
    support = supports_pre(streams, projection, bias, scale, surface=surface, **kwargs)
    if not support:
        return support
    if not streams.is_cuda or torch.cuda.get_device_capability(streams.device) not in (
        (9, 0),
        (10, 0),
    ):
        return SupportResult.no("TileKernels mHC requires SM90 or SM100")
    if (
        streams.numel() == 0
        or streams.shape[-2] != 4
        or (streams.numel() // (4 * streams.shape[-1])) % 8
    ):
        return SupportResult.no(
            "TileKernels mHC requires four streams and token count divisible by eight"
        )
    if not _available():
        return SupportResult.no("requires TileKernels and TileLang >= 0.1.15")
    return SupportResult.yes()


def apply(
    streams, projection, bias, scale, *, eps=1e-5, sinkhorn_eps=1e-6, iterations=20
):
    from ...hyper_connection import mhc_normalize_streams

    with torch.autocast(device_type=streams.device.type, enabled=False):
        projected = F.linear(
            mhc_normalize_streams(streams, eps=eps), projection.float()
        )
        pre, post, logits = projected.split((4, 4, 16), dim=-1)
        pre_bias, post_bias, mixing_bias = bias.float().split((4, 4, 16))
        pre_scale, post_scale, mixing_scale = scale.float().unbind()
        pre = (pre * pre_scale + pre_bias).sigmoid() + sinkhorn_eps
        post = 2 * (post * post_scale + post_bias).sigmoid()
        logits = (logits * mixing_scale + mixing_bias).reshape(-1, 4, 4).contiguous()
        mixing = _sinkhorn(logits, iterations, sinkhorn_eps).reshape(
            *streams.shape[:-2], 4, 4
        )
        collapsed = (pre.unsqueeze(-1) * streams).sum(-2).to(streams.dtype)
        return post, mixing, collapsed


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="mhc_pre",
        implementation_id="tilekernels.mhc_pre.sinkhorn",
        provider="tilekernels",
        priority=100,
        deterministic=True,
        supports=_supports,
        apply=apply,
    )
)


@torch.library.custom_op("mlops::mhc_post_tilekernels", mutates_args=())
def _post(
    branch: torch.Tensor,
    streams: torch.Tensor,
    post: torch.Tensor,
    mixing: torch.Tensor,
) -> torch.Tensor:
    from tile_kernels.mhc import mhc_post_fwd

    with torch.cuda.device(streams.device):
        return mhc_post_fwd(branch, streams, post, mixing)


@_post.register_fake
def _post_fake(branch, streams, post, mixing):
    return torch.empty_like(streams)


@torch.library.custom_op("mlops::mhc_post_tilekernels_backward", mutates_args=())
def _post_vjp(
    grad: torch.Tensor,
    branch: torch.Tensor,
    streams: torch.Tensor,
    post: torch.Tensor,
    mixing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from tile_kernels.mhc import mhc_post_bwd

    with torch.cuda.device(streams.device):
        return mhc_post_bwd(branch, streams, post, mixing, grad)


@_post_vjp.register_fake
def _post_vjp_fake(grad, branch, streams, post, mixing):
    return tuple(torch.empty_like(t) for t in (branch, streams, post, mixing))


def _post_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _post_backward(ctx, grad):
    return _post_vjp(grad, *ctx.saved_tensors)


_post.register_autograd(_post_backward, setup_context=_post_setup)


@flop_formula(_post)
def _post_flops(branch, streams, post, mixing, **_kwargs):
    return costs.mhc_post_flops(streams)


@flop_formula(_post_vjp)
def _post_bwd_flops(grad, branch, streams, post, mixing, **_kwargs):
    return costs.mhc_post_flops(streams, backward=True)


def _supports_post(
    branch, streams, post, mixing, *, surface, coefficient_precision="fp32", **kwargs
):
    result = supports_combine(
        branch,
        streams,
        post,
        mixing,
        surface=surface,
        coefficient_precision=coefficient_precision,
        **kwargs,
    )
    if not result:
        return result
    if coefficient_precision != "fp32":
        return SupportResult.no(
            "TileKernels post uses FP32 coefficients and accumulation"
        )
    if (
        not streams.is_cuda
        or streams.dtype != torch.bfloat16
        or torch.cuda.get_device_capability(streams.device) not in ((9, 0), (10, 0))
    ):
        return SupportResult.no(
            "TileKernels post requires BF16 activations on SM90 or SM100"
        )
    if (
        streams.numel() == 0
        or streams.shape[-2] != 4
        or streams.shape[-1] % 128
        or not streams.is_contiguous()
        or not branch.is_contiguous()
    ):
        return SupportResult.no(
            "requires four contiguous streams and hidden width divisible by 128"
        )
    if not _available():
        return SupportResult.no("requires TileKernels and TileLang >= 0.1.15")
    return SupportResult.yes()


def combine(branch, streams, post, mixing, *, coefficient_precision="fp32"):
    d = streams.shape[-1]
    return _post(
        branch.reshape(1, -1, d),
        streams.reshape(1, -1, 4, d),
        post.float().reshape(1, -1, 4, 1).contiguous(),
        mixing.float().reshape(1, -1, 4, 4).contiguous(),
    ).reshape_as(streams)


COMBINE = register_implementation(
    Implementation(
        operation="mhc_combine",
        implementation_id="tilekernels.mhc_combine",
        provider="tilekernels",
        priority=200,
        deterministic=True,
        supports=_supports_post,
        apply=combine,
    )
)
