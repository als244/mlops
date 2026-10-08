"""Trace-visible grouped linear projections with demand-driven gradients.

Offsets stay on device; each GPU call is one grouped kernel, not a Python loop
launching one GEMM per expert. Frozen weights have no weight-gradient output.
"""

import torch

from ..dispatch.context import weight_gradient_dtype
from ..dispatch.costs import flop_formula
from ..kernels.moe_grouped_gemm import (
    grouped_mm_dgrad,
    grouped_mm_forward,
    grouped_mm_wgrad,
)


@torch.library.custom_op("mlops::lora_grouped_linear_fwd", mutates_args=())
def _forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    grad_dtype: torch.dtype | None,
    need_x: bool,
    need_weight: bool,
) -> torch.Tensor:
    return grouped_mm_forward(x, weight, offsets)


@_forward.register_fake
def _fake(x, weight, offsets, grad_dtype, need_x, need_weight):
    return x.new_empty((x.shape[0], weight.shape[2]))


@torch.library.custom_op(
    "mlops::lora_grouped_linear_bwd",
    mutates_args=(),
    schema="(Tensor grad, Tensor? x, Tensor? weight, Tensor offsets, SymInt[] weight_shape, ScalarType? grad_dtype) -> (Tensor?, Tensor?)",
)
def _backward(
    grad: torch.Tensor,
    x: torch.Tensor | None,
    weight: torch.Tensor | None,
    offsets: torch.Tensor,
    weight_shape: list[int],
    grad_dtype: torch.dtype | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    dx = None if weight is None else grouped_mm_dgrad(grad, weight, offsets)
    dw = (
        None
        if x is None
        else grouped_mm_wgrad(x, grad, offsets, tuple(weight_shape), grad_dtype)
    )
    return dx, dw


@_backward.register_fake
def _backward_fake(grad, x, weight, offsets, weight_shape, grad_dtype):
    return (
        None if weight is None else grad.new_empty((grad.shape[0], weight.shape[1])),
        None
        if x is None
        else torch.empty(
            tuple(weight_shape), device=x.device, dtype=grad_dtype or x.dtype
        ),
    )


def _context(ctx, inputs, output):
    x, weight, offsets, grad_dtype, need_x, need_weight = inputs
    ctx.save_for_backward(
        x if need_weight else None, weight if need_x else None, offsets
    )
    ctx.weight_shape, ctx.grad_dtype = tuple(weight.shape), grad_dtype


def _autograd(ctx, grad):
    dx, dw = _backward(grad, *ctx.saved_tensors, list(ctx.weight_shape), ctx.grad_dtype)
    return dx, dw, None, None, None, None


_forward.register_autograd(_autograd, setup_context=_context)


@flop_formula(_forward)
def _forward_flops(x, weight, *_rest, out_val=None):
    return 2 * x.shape[0] * weight.shape[1] * weight.shape[2]


@flop_formula(_backward)
def _backward_flops(grad, x, weight, offsets, weight_shape, grad_dtype, out_val=None):
    return (
        2
        * grad.shape[0]
        * weight_shape[1]
        * weight_shape[2]
        * ((x is not None) + (weight is not None))
    )


def grouped_linear(x, weight, offsets):
    return _forward(
        x,
        weight,
        offsets,
        weight_gradient_dtype(),
        torch.is_grad_enabled() and x.requires_grad,
        torch.is_grad_enabled() and weight.requires_grad,
    )
