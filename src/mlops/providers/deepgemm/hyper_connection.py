"""DeepGEMM projection plus TileKernels fused four-stream mHC preprocessing.

Forward avoids a full FP32 normalized activation. Backward uses upstream
component VJPs and a fused projection/input-gradient kernel, without a full
FP32 activation workspace. All tensor state is invocation-owned.
"""

from importlib.util import find_spec

import torch

from ...dispatch import logical_costs as costs
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ..native_torch.hyper_connection import supports_pre


@torch.library.custom_op("mlops::mhc_pre_deepgemm", mutates_args=())
@torch.autocast(device_type="cuda", enabled=False)
def _forward(
    streams: torch.Tensor,
    projection: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    sinkhorn_eps: float,
    iterations: int,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import deep_gemm
    from tile_kernels.mhc.pre_big_fuse_kernel import mhc_pre_big_fuse_fwd

    if torch.cuda.get_device_capability(streams.device) not in ((9, 0), (10, 0)):
        raise ValueError("DeepGEMM mHC pre supports only SM90/SM100")
    if (
        streams.dtype != torch.bfloat16
        or streams.shape[-2] != 4
        or not streams.is_contiguous()
    ):
        raise ValueError(
            "DeepGEMM mHC pre needs contiguous BF16 four-stream activations"
        )
    x = streams.reshape(-1, 4, streams.shape[-1])
    tokens, _, hidden = x.shape
    if tokens % 32 or hidden % 128 or projection.shape != (24, 4 * hidden):
        raise ValueError(
            "DeepGEMM mHC pre requires tokens divisible by 32, hidden divisible by 128, projection [24,4H]"
        )
    if splits < 1 or (4 * hidden) % (64 * splits):
        raise ValueError("Each projection split must contain whole 64-element blocks")
    w = projection.float().contiguous()
    b = bias.float().contiguous()
    s = scale.float().contiguous()
    partials = torch.empty((splits, tokens, 24), device=x.device, dtype=torch.float32)
    square_partials = torch.empty(
        (splits, tokens), device=x.device, dtype=torch.float32
    )
    post = torch.empty((tokens, 4), device=x.device, dtype=torch.float32)
    mixing = torch.empty((tokens, 16), device=x.device, dtype=torch.float32)
    branch = torch.empty((tokens, hidden), device=x.device, dtype=torch.bfloat16)
    with torch.cuda.device(x.device):
        deep_gemm.tf32_hc_prenorm_gemm(
            x.view(tokens, 4 * hidden), w, partials, square_partials, num_splits=splits
        )
        mhc_pre_big_fuse_fwd(
            partials,
            square_partials,
            s,
            b,
            x,
            post,
            mixing,
            branch,
            eps,
            sinkhorn_eps,
            sinkhorn_eps,
            2.0,
            iterations,
            n_splits=splits,
        )
    # Only small sufficient statistics survive forward, not the FP32 normalized activation.
    dot = partials.sum(0)
    square = square_partials.sum(0)
    leading = streams.shape[:-2]
    return (
        post.view(*leading, 4),
        mixing.view(*leading, 4, 4),
        branch.view(*leading, hidden),
        dot,
        square,
    )


@torch.library.custom_op(
    "mlops::mhc_pre_deepgemm_backward",
    mutates_args=(),
    schema="(Tensor streams, Tensor projection, Tensor bias, Tensor scale, Tensor post, Tensor dot, Tensor square, Tensor dpost, Tensor dmixing, Tensor dbranch, float eps, float seps, int iterations, bool projection_grad, bool bias_grad, bool scale_grad) -> (Tensor, Tensor?, Tensor?, Tensor?)",
)
@torch.autocast(device_type="cuda", enabled=False)
def _backward_op(
    streams: torch.Tensor,
    projection: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    post: torch.Tensor,
    dot: torch.Tensor,
    square: torch.Tensor,
    dpost: torch.Tensor,
    dmixing: torch.Tensor,
    dbranch: torch.Tensor,
    eps: float,
    seps: float,
    iterations: int,
    projection_grad: bool,
    bias_grad: bool,
    scale_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    from tile_kernels.mhc import mhc_sinkhorn_bwd
    from tile_kernels.mhc.norm_fn_kernel import mhc_reduce_partials_and_rmsnorm_bwd
    from tile_kernels.mhc.pre_apply_mix_kernel import mhc_pre_apply_mix_bwd
    from tile_kernels.mhc.pre_split_mixes_kernel import mhc_pre_split_mixes_bwd

    x = streams.reshape(-1, 4, streams.shape[-1])
    tokens, _, hidden = x.shape
    k = 4 * hidden
    b = bias.float().contiguous()
    s = scale.float().contiguous()
    z = dot * torch.rsqrt(square[:, None] / k + eps)
    pre = torch.sigmoid(z[:, :4] * s[0] + b[:4]) + seps
    logits = (z[:, 8:] * s[2] + b[8:]).reshape(tokens, 4, 4).contiguous()
    direct = torch.zeros_like(x)  # Upstream collapse VJP accumulates into this tensor.
    dz = torch.empty_like(z)
    ds = torch.empty_like(s)
    db = torch.empty_like(b)
    dlogits = torch.empty_like(logits)
    du = torch.empty((tokens, 1, 24), device=x.device, dtype=torch.float32)
    dss = torch.empty((tokens, 1), device=x.device, dtype=torch.float32)
    with torch.cuda.device(x.device):
        dp = mhc_pre_apply_mix_bwd(
            dbranch.reshape(tokens, hidden).contiguous(), x, pre.contiguous(), direct
        )
        mhc_sinkhorn_bwd(
            dmixing.reshape(tokens, 4, 4).contiguous(),
            logits,
            dlogits,
            iterations,
            seps,
        )
        mhc_pre_split_mixes_bwd(
            dp,
            dpost.reshape(tokens, 4).contiguous(),
            dlogits.reshape(tokens, 16),
            z.contiguous(),
            post.reshape(tokens, 4).contiguous(),
            s,
            b,
            dz,
            ds,
            db,
            2.0,
        )
        mhc_reduce_partials_and_rmsnorm_bwd(
            dz, dot.view(tokens, 1, 24), square.view(tokens, 1), du, dss, k, eps
        )
    du = du.view(tokens, 24)
    from ...kernels.mhc_pre import project_input_gradient

    dx = project_input_gradient(du, projection, x, dss, direct).view_as(streams)
    dw = None
    if projection_grad:
        # Avoid a full FP32 activation conversion for optional parameter gradients.
        dw = torch.zeros_like(projection, dtype=torch.float32)
        for start in range(0, tokens, 256):
            dw.add_(
                du[start : start + 256].T
                @ x[start : start + 256].reshape(-1, k).float()
            )
        dw = dw.to(projection.dtype)
    return (
        dx,
        dw,
        db.to(bias.dtype) if bias_grad else None,
        ds.to(scale.dtype) if scale_grad else None,
    )


@_forward.register_fake
def _forward_fake(
    streams, projection, bias, scale, eps, sinkhorn_eps, iterations, splits
):
    prefix, hidden = streams.shape[:-2], streams.shape[-1]
    tokens = streams.numel() // (4 * hidden)
    opts = {"device": streams.device, "dtype": torch.float32}
    return (
        torch.empty((*prefix, 4), **opts),
        torch.empty((*prefix, 4, 4), **opts),
        torch.empty((*prefix, hidden), device=streams.device, dtype=streams.dtype),
        torch.empty((tokens, 24), **opts),
        torch.empty((tokens,), **opts),
    )


@_backward_op.register_fake
def _backward_fake(
    streams,
    projection,
    bias,
    scale,
    post,
    dot,
    square,
    dpost,
    dmixing,
    dbranch,
    eps,
    seps,
    iterations,
    projection_grad,
    bias_grad,
    scale_grad,
):
    return (
        torch.empty_like(streams),
        torch.empty_like(projection) if projection_grad else None,
        torch.empty_like(bias) if bias_grad else None,
        torch.empty_like(scale) if scale_grad else None,
    )


def _setup(ctx, inputs, output):
    ctx.options = inputs[4:7]
    ctx.save_for_backward(*inputs[:4], output[0], output[3], output[4])
    ctx.mark_non_differentiable(output[3], output[4])


def _backward(ctx, dpost, dmixing, dbranch, _ddot, _dsquare):
    result = _backward_op(
        *ctx.saved_tensors,
        dpost,
        dmixing,
        dbranch,
        *ctx.options,
        *ctx.needs_input_grad[1:4],
    )
    return (*result, None, None, None, None)


_forward.register_autograd(_backward, setup_context=_setup)


@flop_formula(_forward)
def _forward_flops(
    streams, projection, bias, scale, eps, sinkhorn_eps, iterations, splits, **_
):
    return costs.mhc_pre(
        streams, projection, bias, scale, iterations=iterations
    ).logical_flops


@flop_formula(_backward_op)
def _backward_flops(
    streams,
    projection,
    bias,
    scale,
    post,
    dot,
    square,
    dpost,
    dmixing,
    dbranch,
    eps,
    seps,
    iterations,
    projection_grad,
    bias_grad,
    scale_grad,
    **_,
):
    total = costs.mhc_pre(
        streams, projection, bias, scale, iterations=iterations, entrypoint="backward"
    ).logical_flops
    if not projection_grad:
        total -= 2 * streams.numel() * projection.shape[0]
    # The component VJP reconstructs Sinkhorn iteration states from logits.
    return total + costs.mhc_sinkhorn_flops(dmixing, iterations)


@torch.compiler.assume_constant_result
def _available():
    if find_spec("deep_gemm") is None or find_spec("tile_kernels") is None:
        return False
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import Version

    try:
        return Version(version("tilelang")) >= Version("0.1.15") and Version(
            version("deep_gemm")
        ) >= Version("2.8.2")
    except PackageNotFoundError:
        return False


def _supports(streams, projection, bias, scale, *, surface, **kwargs):
    result = supports_pre(streams, projection, bias, scale, surface=surface, **kwargs)
    if not result:
        return result
    if not streams.is_cuda or torch.cuda.get_device_capability(streams.device) not in (
        (9, 0),
        (10, 0),
    ):
        return SupportResult.no("DeepGEMM mHC pre requires SM90 or SM100")
    if (
        streams.dtype != torch.bfloat16
        or not streams.is_contiguous()
        or streams.numel() == 0
        or streams.shape[-2] != 4
        or streams.shape[-1] % 128
        or (streams.numel() // (4 * streams.shape[-1])) % 32
        or projection.dtype not in (torch.bfloat16, torch.float32)
    ):
        return SupportResult.no(
            "requires contiguous BF16 four-stream inputs, H multiple of 128, T multiple of 32 and BF16/FP32 projection"
        )
    if not _available():
        return SupportResult.no(
            "requires the pinned DeepGEMM provider, TileKernels and TileLang >= 0.1.15"
        )
    return SupportResult.yes()


def apply(
    streams, projection, bias, scale, *, eps=1e-5, sinkhorn_eps=1e-6, iterations=20
):
    # One split already supplies >=256 CTAs at T=16K. Keep workspace bounded.
    return _forward(streams, projection, bias, scale, eps, sinkhorn_eps, iterations, 1)[
        :3
    ]


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="mhc_pre",
        implementation_id="deepgemm.mhc_pre",
        provider="deepgemm",
        # Preferred on supported four-stream geometry; SM90 numerically qualified.
        priority=150,
        deterministic=True,
        supports=_supports,
        apply=apply,
    )
)
