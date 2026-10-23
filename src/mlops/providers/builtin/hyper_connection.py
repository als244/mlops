"""Shape-specialized four-stream mHC matrix product with explicit autograd."""

import torch

from ...dispatch import logical_costs as costs
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.hyper_connection import (
    mix_four,
    post_four,
    post_four_coefficient_grads,
    post_four_input_grads,
)
from ..native_torch.hyper_connection import supports_combine as _reference_supports


def _validate(matrix, streams):
    if (
        streams.ndim != 3
        or streams.shape[1] != 4
        or matrix.shape != (streams.shape[0], 4, 4)
        or not streams.is_cuda
        or matrix.device != streams.device
        or matrix.dtype != streams.dtype
        or streams.dtype not in (torch.bfloat16, torch.float16)
    ):
        raise ValueError("mHC mix requires CUDA BF16/FP16 [N,4,4] and [N,4,D]")


@torch.library.custom_op("mlops::mhc_mix_four", mutates_args=())
def mix(matrix: torch.Tensor, streams: torch.Tensor) -> torch.Tensor:
    _validate(matrix, streams)
    return mix_four(matrix, streams.contiguous())


@mix.register_fake
def _fake(matrix, streams):
    _validate(matrix, streams)
    return torch.empty_like(streams, memory_format=torch.contiguous_format)


@flop_formula(mix)
def _flops(matrix, streams, *, out_val=None, **kwargs):
    return 8 * streams.numel()


def _setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad):
    matrix, streams = ctx.saved_tensors
    grad_matrix = grad @ streams.transpose(-1, -2) if ctx.needs_input_grad[0] else None
    grad_streams = (
        mix(matrix.transpose(-1, -2), grad) if ctx.needs_input_grad[1] else None
    )
    return grad_matrix, grad_streams


mix.register_autograd(_backward, setup_context=_setup)


# The specialized matrix multiply is an implementation detail. Dispatch at
# the complete post boundary allows a provider to fuse expansion and addition.


def _supports(
    branch, streams, post, mixing, *, surface, coefficient_precision="fp32", **kwargs
):
    result = _reference_supports(
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
    if not streams.is_cuda or streams.dtype not in (torch.bfloat16, torch.float16):
        return SupportResult.no("specialized mix requires CUDA BF16 or FP16")
    if (
        streams.shape[-2] != 4
        or not streams.is_contiguous()
        or not branch.is_contiguous()
    ):
        return SupportResult.no("specialized mix requires four contiguous streams")
    return SupportResult.yes()


def apply(branch, streams, post, mixing, *, coefficient_precision="fp32"):
    if coefficient_precision == "fp32":
        return _post(
            branch.reshape(-1, streams.shape[-1]),
            streams.reshape(-1, 4, streams.shape[-1]),
            post.float().reshape(-1, 4).contiguous(),
            mixing.float().reshape(-1, 4, 4).contiguous(),
        ).reshape_as(streams)
    matrix = mixing.to(streams.dtype).transpose(-1, -2)
    residual = mix(
        matrix.reshape(-1, 4, 4), streams.reshape(-1, 4, streams.shape[-1])
    ).reshape_as(streams)
    return post.to(streams.dtype).unsqueeze(-1) * branch.unsqueeze(-2) + residual


@torch.library.custom_op("mlops::mhc_post_four", mutates_args=())
def _post(
    branch: torch.Tensor,
    streams: torch.Tensor,
    post: torch.Tensor,
    mixing: torch.Tensor,
) -> torch.Tensor:
    return post_four(branch, streams, post, mixing)


@_post.register_fake
def _post_fake(branch, streams, post, mixing):
    return torch.empty_like(streams)


@torch.library.custom_op("mlops::mhc_post_four_input_grads", mutates_args=())
def _input_grads(
    grad: torch.Tensor, post: torch.Tensor, mixing: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return post_four_input_grads(grad.contiguous(), post, mixing)


@_input_grads.register_fake
def _input_grads_fake(grad, post, mixing):
    return (
        torch.empty(
            (grad.shape[0], grad.shape[2]), device=grad.device, dtype=grad.dtype
        ),
        torch.empty_like(grad, memory_format=torch.contiguous_format),
    )


@torch.library.custom_op("mlops::mhc_post_four_coefficient_grads", mutates_args=())
def _coefficient_grads(
    grad: torch.Tensor, branch: torch.Tensor, streams: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return post_four_coefficient_grads(grad.contiguous(), branch, streams)


@_coefficient_grads.register_fake
def _coefficient_grads_fake(grad, branch, streams):
    opts = {"device": streams.device, "dtype": torch.float32}
    return torch.empty((streams.shape[0], 4), **opts), torch.empty(
        (streams.shape[0], 4, 4), **opts
    )


@flop_formula(_post)
def _post_flops(branch, streams, post, mixing, **_kwargs):
    return costs.mhc_post_flops(streams)


@flop_formula(_input_grads)
def _input_grads_flops(grad, post, mixing, **_kwargs):
    return costs.mhc_post_flops(grad, backward=True) // 2


@flop_formula(_coefficient_grads)
def _coefficient_grads_flops(grad, branch, streams, **_kwargs):
    return costs.mhc_post_flops(streams, backward=True) // 2


def _post_backward(ctx, grad):
    branch, streams, post, mixing = ctx.saved_tensors
    db = dx = dp = dm = None
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        db, dx = _input_grads(grad, post, mixing)
    if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
        dp, dm = _coefficient_grads(grad, branch, streams)
    return db, dx, dp, dm


_post.register_autograd(_post_backward, setup_context=_setup)


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="mhc_combine",
        implementation_id="builtin.mhc_combine.four_stream",
        provider="builtin",
        priority=100,
        deterministic=True,
        supports=_supports,
        apply=apply,
    )
)
