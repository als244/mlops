"""Builtin Triton partial-channel RoPE implementation adapter."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.partial_rope import partial_rope_backward, partial_rope_forward


def _supports(
    x, positions, _base, rotary_dim, cosine, sine, *, surface, **_kwargs
):
    del surface
    if not isinstance(x, torch.Tensor) or not isinstance(positions, torch.Tensor):
        return SupportResult.no("x and positions must be tensors")
    if int(rotary_dim) <= 0 or int(rotary_dim) > x.shape[-1] or int(rotary_dim) % 2:
        return SupportResult.no("rotary_dim must be positive, even, and within x")
    if not x.is_cuda or not x.is_contiguous() or positions.device != x.device:
        return SupportResult.no("requires contiguous CUDA x and colocated positions")
    if cosine.shape != sine.shape or cosine.shape[-1] != int(rotary_dim):
        return SupportResult.no("cosine/sine tables must match rotary_dim")
    if cosine.dtype != torch.float32 or sine.dtype != torch.float32:
        return SupportResult.no("cosine/sine tables must be FP32")
    if cosine.device != x.device or sine.device != x.device:
        return SupportResult.no("cosine/sine tables must be colocated with x")
    return SupportResult.yes()


def forward(x, positions, base, rotary_dim, cosine, sine):
    return partial_rope_forward(
        x, positions, float(base), int(rotary_dim), cosine, sine
    )


def backward(grad_output, positions, base, rotary_dim, cosine, sine):
    return partial_rope_backward(
        grad_output,
        positions,
        float(base),
        int(rotary_dim),
        cosine,
        sine,
    )


@torch.library.custom_op(
    "mlops::partial_rope_builtin_triton_fwd", mutates_args=()
)
def _forward_op(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    rotary_dim: int,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    return forward(
        x, positions, float(base), int(rotary_dim), cosine, sine
    )


@_forward_op.register_fake
def _forward_fake(x, positions, base, rotary_dim, cosine, sine):
    del positions, base, rotary_dim, cosine, sine
    return torch.empty_like(x)


@torch.library.custom_op(
    "mlops::partial_rope_builtin_triton_bwd", mutates_args=()
)
def _backward_op(
    grad_output: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    rotary_dim: int,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    return backward(
        grad_output,
        positions,
        float(base),
        int(rotary_dim),
        cosine,
        sine,
    )


@_backward_op.register_fake
def _backward_fake(grad_output, positions, base, rotary_dim, cosine, sine):
    del positions, base, rotary_dim, cosine, sine
    return torch.empty_like(grad_output)


def _setup_context(ctx, inputs, output):
    del output
    _x, positions, base, rotary_dim, cosine, sine = inputs
    ctx.save_for_backward(positions, cosine, sine)
    ctx.base = base
    ctx.rotary_dim = rotary_dim


def _autograd_backward(ctx, grad_output):
    positions, cosine, sine = ctx.saved_tensors
    return (
        _backward_op(
            grad_output,
            positions,
            ctx.base,
            ctx.rotary_dim,
            cosine,
            sine,
        ),
        None,
        None,
        None,
        None,
        None,
    )


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def apply(x, positions, base, rotary_dim, cosine, sine):
    return _forward_op(
        x, positions, float(base), int(rotary_dim), cosine, sine
    )


IMPLEMENTATION = register_implementation(
    Implementation(
        "partial_rope", "builtin.partial_rope.triton", "builtin", 100, True,
        _supports, apply, forward, backward,
    )
)

__all__ = ["IMPLEMENTATION", "apply", "backward", "forward"]
