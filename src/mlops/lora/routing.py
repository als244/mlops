"""Autograd routing/dispatch/combine using the existing MLOps kernels."""

import torch

from ..dispatch.costs import flop_formula
from ..kernels.moe_dispatch import (
    combine,
    dispatch,
    dispatch_backward,
    row_dot,
    scale_rows_,
    sort_assignments,
)
from ..kernels.moe_router import current_route_weight_precision, route, route_backward


@torch.library.custom_op("mlops::lora_route", mutates_args=())
def _route(
    logits: torch.Tensor, top_k: int, mode: str, precision: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return route(logits, top_k, mode, weight_precision=precision)


@_route.register_fake
def _route_fake(logits, top_k, mode, precision):
    shape = (logits.shape[0], top_k)
    return (
        logits.new_empty(
            shape, dtype=torch.float32 if precision == "float32" else logits.dtype
        ),
        logits.new_empty(shape, dtype=torch.int32),
    )


@torch.library.custom_op("mlops::lora_route_bwd", mutates_args=())
def _route_backward(
    grad: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    logits: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    return route_backward(grad, weights, ids, logits, mode).to(logits.dtype)


@_route_backward.register_fake
def _route_backward_fake(grad, weights, ids, logits, mode):
    return torch.empty_like(logits)


def _route_context(ctx, inputs, output):
    logits, _, mode, _ = inputs
    weights, ids = output
    ctx.save_for_backward(weights, ids, logits)
    ctx.mode = mode
    ctx.mark_non_differentiable(ids)


def _route_autograd(ctx, grad, _ids):
    return _route_backward(grad, *ctx.saved_tensors, ctx.mode), None, None, None


_route.register_autograd(_route_autograd, setup_context=_route_context)


@torch.library.custom_op("mlops::lora_assignment_layout", mutates_args=())
def layout(
    ids: torch.Tensor, experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return sort_assignments(ids, experts)


@layout.register_fake
def _layout_fake(ids, experts):
    return (
        ids.new_empty((ids.numel(),), dtype=torch.int32),
        ids.new_empty((experts + 1,), dtype=torch.int32),
        torch.empty_like(ids, dtype=torch.int32),
    )


@torch.library.custom_op("mlops::lora_dispatch", mutates_args=())
def _dispatch(
    x: torch.Tensor, order: torch.Tensor, slots: torch.Tensor
) -> torch.Tensor:
    return dispatch(x, order, slots.shape[1])


@_dispatch.register_fake
def _dispatch_fake(x, order, slots):
    return x.new_empty((order.numel(), x.shape[1]))


@torch.library.custom_op("mlops::lora_dispatch_bwd", mutates_args=())
def _dispatch_backward(grad: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    return dispatch_backward(grad, slots)


@_dispatch_backward.register_fake
def _dispatch_backward_fake(grad, slots):
    return grad.new_empty((slots.shape[0], grad.shape[1]))


def _dispatch_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[2])


def _dispatch_autograd(ctx, grad):
    return _dispatch_backward(grad, *ctx.saved_tensors), None, None


_dispatch.register_autograd(_dispatch_autograd, setup_context=_dispatch_context)


@torch.library.custom_op("mlops::lora_combine", mutates_args=())
def _combine(
    y: torch.Tensor,
    order: torch.Tensor,
    slots: torch.Tensor,
    weights: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return combine(y, slots, weights, residual)


@_combine.register_fake
def _combine_fake(y, order, slots, weights, residual):
    return torch.empty_like(residual)


@torch.library.custom_op("mlops::lora_combine_bwd", mutates_args=())
def _combine_backward(
    grad: torch.Tensor,
    y: torch.Tensor,
    order: torch.Tensor,
    slots: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dispatched = dispatch(grad.contiguous(), order, slots.shape[1])
    dp = (
        row_dot(dispatched, y)[slots.reshape(-1).long()]
        .view_as(weights)
        .to(weights.dtype)
    )
    scale_rows_(dispatched, weights.reshape(-1)[order.long()].float())
    return dispatched, dp


@_combine_backward.register_fake
def _combine_backward_fake(grad, y, order, slots, weights):
    return torch.empty_like(y), torch.empty_like(weights)


def _combine_context(ctx, inputs, output):
    y, order, slots, weights, _ = inputs
    ctx.save_for_backward(y, order, slots, weights)


def _combine_autograd(ctx, grad):
    dy, dp = _combine_backward(grad, *ctx.saved_tensors)
    return dy, None, None, dp, grad


_combine.register_autograd(_combine_autograd, setup_context=_combine_context)


def routing(logits, top_k, mode):
    return _route(logits, top_k, mode, current_route_weight_precision())


def dispatch_rows(x, order, slots):
    return _dispatch(x, order, slots)


def combine_rows(y, order, slots, weights, residual):
    return _combine(y, order, slots, weights, residual)


@flop_formula(layout, _dispatch)
def _gather_flops(*args, out_val=None, **kwargs):
    return 0


@flop_formula(_dispatch_backward)
def _dispatch_backward_flops(grad, slots, out_val=None):
    return slots.shape[0] * (slots.shape[1] - 1) * grad.shape[1]


@flop_formula(_combine)
def _combine_flops(y, order, slots, weights, residual, out_val=None):
    return 2 * y.numel()


@flop_formula(_combine_backward)
def _combine_backward_flops(grad, y, order, slots, weights, out_val=None):
    return 3 * y.numel()


@flop_formula(_route)
def _route_flops(logits, top_k, mode, precision, out_val=None):
    return (
        6
        * logits.shape[0]
        * (top_k if mode == "topk_then_softmax" else logits.shape[1])
    )


@flop_formula(_route_backward)
def _route_backward_flops(grad, weights, ids, logits, mode, out_val=None):
    return 4 * logits.numel()
