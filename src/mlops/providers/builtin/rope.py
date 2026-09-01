"""Builtin Triton table and analytic RoPE implementation adapters."""

from __future__ import annotations

import torch

from ...dispatch.costs import CostHints
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.rope import rope_backward, rope_forward


def _supports(
    x, positions, _base, cosine, sine, *, surface, **_kwargs
):
    del surface
    if not isinstance(x, torch.Tensor) or not isinstance(positions, torch.Tensor):
        return SupportResult.no("x and positions must be tensors")
    if x.shape[-1] % 2:
        return SupportResult.no("head dimension must be even")
    if not x.is_cuda or not x.is_contiguous() or positions.device != x.device:
        return SupportResult.no("requires contiguous CUDA x and colocated positions")
    if cosine.shape != sine.shape or cosine.shape[-1] != x.shape[-1]:
        return SupportResult.no("cosine/sine tables must match the head width")
    if cosine.dtype != torch.float32 or sine.dtype != torch.float32:
        return SupportResult.no("cosine/sine tables must be FP32")
    if cosine.device != x.device or sine.device != x.device:
        return SupportResult.no("cosine/sine tables must be colocated with x")
    return SupportResult.yes()


def _forward(variant, x, positions, base, cosine, sine):
    return rope_forward(
        x, positions, float(base), cosine, sine, variant=variant
    )


def _backward(variant, grad_output, positions, base, cosine, sine):
    return rope_backward(
        grad_output,
        positions,
        float(base),
        cosine,
        sine,
        variant=variant,
    )


def _table_estimate(
    x, positions, _base, cosine, sine, *, entrypoint, **_kwargs
):
    del x, positions, cosine, sine, entrypoint
    return CostHints(
        workspace_bytes=0,
        notes=(
            "cosine/sine tables are caller-owned inputs, not workspace",
            "physical FLOPs and bytes are undefined pending kernel accounting",
        ),
    )


def _analytic_estimate(
    x, positions, _base, cosine, sine, *, entrypoint, **_kwargs
):
    del x, positions, cosine, sine, entrypoint
    return CostHints(
        workspace_bytes=0,
        notes=(
            "analytic RoPE computes angles in the Triton kernel without a table",
            "physical FLOPs and bytes are undefined pending kernel accounting",
        ),
    )


def table_forward(x, positions, base, cosine, sine):
    return _forward("triton_table", x, positions, base, cosine, sine)


def table_backward(grad_output, positions, base, cosine, sine):
    return _backward(
        "triton_table", grad_output, positions, base, cosine, sine
    )


def table_apply(x, positions, base, cosine, sine):
    return _table_forward_op(
        x.contiguous(), positions, float(base), cosine, sine
    )


def analytic_forward(x, positions, base, cosine, sine):
    return _forward("triton_analytic", x, positions, base, cosine, sine)


def analytic_backward(grad_output, positions, base, cosine, sine):
    return _backward(
        "triton_analytic", grad_output, positions, base, cosine, sine
    )


def analytic_apply(x, positions, base, cosine, sine):
    return _analytic_forward_op(
        x.contiguous(), positions, float(base), cosine, sine
    )


@torch.library.custom_op(
    "mlops::rope_builtin_triton_table_fwd", mutates_args=()
)
def _table_forward_op(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    return table_forward(x, positions, float(base), cosine, sine)


@_table_forward_op.register_fake
def _forward_fake(x, positions, base, cosine, sine):
    del positions, base, cosine, sine
    return torch.empty_like(x)


@torch.library.custom_op(
    "mlops::rope_builtin_triton_table_bwd", mutates_args=()
)
def _table_backward_op(
    grad_output: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    return table_backward(
        grad_output, positions, float(base), cosine, sine
    )


@_table_backward_op.register_fake
def _backward_fake(grad_output, positions, base, cosine, sine):
    del positions, base, cosine, sine
    return torch.empty_like(grad_output)


def _setup_context(ctx, inputs, output):
    del output
    _x, positions, base, cosine, sine = inputs
    ctx.save_for_backward(positions, cosine, sine)
    ctx.base = base


def _table_autograd_backward(ctx, grad_output):
    positions, cosine, sine = ctx.saved_tensors
    return (
        _table_backward_op(grad_output, positions, ctx.base, cosine, sine),
        None,
        None,
        None,
        None,
    )


_table_forward_op.register_autograd(
    _table_autograd_backward, setup_context=_setup_context
)


@torch.library.custom_op(
    "mlops::rope_builtin_triton_analytic_fwd", mutates_args=()
)
def _analytic_forward_op(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    return analytic_forward(x, positions, float(base), cosine, sine)


@_analytic_forward_op.register_fake
def _analytic_forward_fake(x, positions, base, cosine, sine):
    del positions, base, cosine, sine
    return torch.empty_like(x)


@torch.library.custom_op(
    "mlops::rope_builtin_triton_analytic_bwd", mutates_args=()
)
def _analytic_backward_op(
    grad_output: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    return analytic_backward(
        grad_output, positions, float(base), cosine, sine
    )


@_analytic_backward_op.register_fake
def _analytic_backward_fake(grad_output, positions, base, cosine, sine):
    del positions, base, cosine, sine
    return torch.empty_like(grad_output)


def _analytic_autograd_backward(ctx, grad_output):
    positions, cosine, sine = ctx.saved_tensors
    return (
        _analytic_backward_op(grad_output, positions, ctx.base, cosine, sine),
        None,
        None,
        None,
        None,
    )


_analytic_forward_op.register_autograd(
    _analytic_autograd_backward, setup_context=_setup_context
)


IMPLEMENTATIONS = tuple(
    register_implementation(implementation)
    for implementation in (
        Implementation(
            "rope", "builtin.rope.triton_table", "builtin", 100, True,
            _supports, table_apply, table_forward, table_backward,
            _table_estimate,
        ),
        Implementation(
            "rope", "builtin.rope.triton_analytic", "builtin", 90, True,
            _supports, analytic_apply, analytic_forward, analytic_backward,
            _analytic_estimate,
        ),
    )
)

__all__ = ["IMPLEMENTATIONS"]
