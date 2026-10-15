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
from ..kernels.router import (
    router_input_gradient,
    router_logits,
    router_weight_gradient,
)


# The router's logits as an operator with its own backward: `routed_logits`'s
# autograd.Function is lost to torch.export, and the multiply it wraps (bf16
# operands, fp32 sums and output) has no derivative of its own.
@torch.library.custom_op("mlops::lora_router_logits", mutates_args=())
def _router_logits(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return router_logits(hidden, weight)


@_router_logits.register_fake
def _router_logits_fake(hidden, weight):
    return hidden.new_empty((hidden.shape[0], weight.shape[0]), dtype=torch.float32)


@torch.library.custom_op("mlops::lora_router_logits_bwd", mutates_args=())
def _router_logits_backward(
    grad: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    need_hidden: bool,
    need_weight: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad = grad.contiguous()
    return (
        router_input_gradient(grad, weight, hidden)
        if need_hidden
        else hidden.new_empty(0),
        router_weight_gradient(grad, hidden, weight.dtype)
        if need_weight
        else weight.new_empty(0),
    )


@_router_logits_backward.register_fake
def _router_logits_backward_fake(grad, hidden, weight, need_hidden, need_weight):
    del grad
    return (
        torch.empty_like(hidden) if need_hidden else hidden.new_empty(0),
        torch.empty_like(weight) if need_weight else weight.new_empty(0),
    )


def _router_logits_context(ctx, inputs, output):
    del output
    ctx.save_for_backward(*inputs)


def _router_logits_autograd(ctx, grad):
    hidden, weight = ctx.saved_tensors
    need_hidden, need_weight = ctx.needs_input_grad
    grad_hidden, grad_weight = _router_logits_backward(
        grad, hidden, weight, bool(need_hidden), bool(need_weight)
    )
    return (
        grad_hidden if need_hidden else None,
        grad_weight if need_weight else None,
    )


_router_logits.register_autograd(
    _router_logits_autograd, setup_context=_router_logits_context
)


def routing_logits(hidden, weight):
    """`mlops.kernels.router`'s logits ``[rows, E]`` (operands at the
    activation's dtype, fp32 sums and output) as a traced graph keeps them:
    one operator, its backward registered."""
    return _router_logits(hidden, weight)


@torch.library.custom_op("mlops::lora_route", mutates_args=())
def _route(
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    mode: str,
    n_group: int,
    topk_group: int,
    routed_scaling: float,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return route(
        logits,
        top_k,
        mode,
        bias=None if bias.numel() == 0 else bias,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling=routed_scaling,
        weight_precision=precision,
    )


@_route.register_fake
def _route_fake(
    logits, bias, top_k, mode, n_group, topk_group, routed_scaling, precision
):
    del bias, mode, n_group, topk_group, routed_scaling
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
    logits, _bias, _top_k, mode, *_ = inputs
    weights, ids = output
    ctx.save_for_backward(weights, ids, logits)
    ctx.mode = mode
    ctx.mark_non_differentiable(ids)


def _route_autograd(ctx, grad, _ids):
    # The correction bias only selects; no gradient reaches it.
    return (
        _route_backward(grad, *ctx.saved_tensors, ctx.mode),
        *([None] * 7),
    )


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


def routing(
    logits, top_k, mode, *, bias=None, n_group=1, topk_group=1, routed_scaling=1.0
):
    """``(weights, ids)`` of `mlops.moe`'s routing: a softmax mode, or
    ``"sigmoid_noaux_tc"`` with its correction ``bias`` (selection only),
    expert groups and ``routed_scaling``."""
    return _route(
        logits,
        logits.new_empty(0) if bias is None else bias,
        int(top_k),
        mode,
        int(n_group),
        int(topk_group),
        float(routed_scaling),
        current_route_weight_precision(),
    )


def statistics(logits, ids, offsets, *, routing_mode, lengths=None):
    """``(aux, counts, probability_sum)`` as `mlops.moe` reports them, in
    differentiable operations on ``logits``: the softmax modes balance the
    whole batch, ``"sigmoid_noaux_tc"`` every sequence of ``lengths`` (all
    rows one sequence when ``None``) on its own."""
    experts = logits.shape[1]
    counts = (offsets[1:] - offsets[:-1]).long()
    if routing_mode != "sigmoid_noaux_tc":
        probabilities = logits.float().softmax(-1)
        auxiliary = (
            experts * ((counts.float() / ids.numel()) * probabilities.mean(0)).sum()
        )
        return auxiliary, counts, probabilities.sum(0)
    from ..providers.moe_common import routing_statistics

    rows = logits.shape[0]
    auxiliary, _counts, probability_sum = routing_statistics(
        logits,
        ids,
        routing_mode=routing_mode,
        lengths=(rows,) if lengths is None else tuple(int(n) for n in lengths),
        experts=experts,
    )
    return auxiliary, counts, probability_sum


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


@flop_formula(_router_logits)
def _router_logits_flops(hidden, weight, out_val=None):
    del out_val
    return 2 * hidden.shape[0] * hidden.shape[1] * weight.shape[0]


@flop_formula(_router_logits_backward)
def _router_logits_backward_flops(
    grad, hidden, weight, need_hidden, need_weight, out_val=None
):
    del grad, out_val
    # The weight's gradient takes two products (the cotangent's bf16 pieces).
    product = 2 * hidden.shape[0] * hidden.shape[1] * weight.shape[0]
    return product * (int(need_hidden) + 2 * int(need_weight))


@flop_formula(_route)
def _route_flops(logits, bias, top_k, mode, *_rest, out_val=None):
    del bias, out_val
    return (
        6
        * logits.shape[0]
        * (top_k if mode == "topk_then_softmax" else logits.shape[1])
    )


@flop_formula(_route_backward)
def _route_backward_flops(grad, weights, ids, logits, mode, out_val=None):
    return 4 * logits.numel()
