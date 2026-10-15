"""Trace-visible routed MoE through SonicMoE's kernels, as two opaque regions.

Like the composed builtin, the routing/up-projection region yields the saved
pre-activation ``h`` and the activation/down-projection/combine region
consumes it, so AOTAutograd can rematerialize the first without replaying the
second. The activation is recomputed from ``h`` where the second region needs
it, as the builtin recomputes it from ``h13``.
"""

from __future__ import annotations

import torch

from ...dispatch import logical_costs as logical
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.moe_router import current_route_weight_precision, route
from ...kernels.router import router_logits
from ...moe import _production_shared_base
from ..moe_common import (
    add_router_input_gradient_,
    requested,
    router_weight_gradient,
    routing_gradient,
    routing_statistics,
    skipped_gradient,
)
from . import fp8
from .moe import (
    activation,
    down_backward,
    down_projection,
    metadata,
    normalized,
    residual_fakes,
    statistic_fakes,
    up_backward,
    up_projection,
)
from .moe import supports as single_supports


def _supports(*args, surface, **kwargs):
    if surface == "explicit":
        return SupportResult.no(
            "trace-visible MoE composition is a semantic-only implementation"
        )
    return single_supports(*args, surface=surface, **kwargs)


def _prepare_forward(
    h2,
    router_weight,
    w13_experts,
    router_bias,
    *,
    top_k,
    routing_mode,
    n_group,
    topk_group,
    routed_scaling,
    lengths,
    weight_precision,
    compute_precision="native",
):
    """Route, build the metadata and compute ``h``, the value the down
    projection needs."""
    rows = h2.numel() // h2.shape[-1]
    normalized_lengths = normalized(lengths, rows)
    experts = router_weight.shape[0]
    with torch.no_grad():
        h2_flat = h2.reshape(-1, h2.shape[-1]).contiguous()
        logits = router_logits(h2_flat, router_weight)
        route_weights, route_ids = route(
            logits,
            int(top_k),
            str(routing_mode),
            bias=None if router_bias.numel() == 0 else router_bias,
            n_group=int(n_group),
            topk_group=int(topk_group),
            routed_scaling=float(routed_scaling),
            weight_precision=str(weight_precision),
        )
        offsets, gather, scatter, reverse = metadata(route_ids, experts)
        h, _activation = up_projection(
            h2_flat, w13_experts, offsets, gather, compute_precision
        )
        aux, counts, probability_sum = routing_statistics(
            logits,
            route_ids,
            routing_mode=str(routing_mode),
            lengths=normalized_lengths,
            experts=experts,
        )
        return (
            aux,
            counts,
            probability_sum,
            logits,
            route_weights,
            route_ids,
            offsets,
            gather,
            scatter,
            reverse,
            h,
        )


def _prepare_backward(
    grad_aux,
    grad_probability_sum,
    grad_route_weights,
    grad_h,
    h2,
    router_weight,
    w13_experts,
    logits,
    route_weights,
    route_ids,
    offsets,
    gather,
    reverse,
    *,
    top_k,
    routing_mode,
    lengths,
    weight_grad_dtype=None,
    compute_precision="native",
    need_router=True,
    need_w13=True,
):
    """Return VJPs for the preparation region's differentiable inputs, the
    router's and the experts' at ``weight_grad_dtype`` (their own when
    ``None``); a weight nobody asked a gradient for gets a placeholder."""
    with torch.no_grad():
        h2_flat = h2.reshape(-1, h2.shape[-1]).contiguous()
        grad_h2, grad_w13 = up_backward(
            h2_flat,
            w13_experts,
            grad_h,
            offsets,
            gather,
            reverse,
            int(top_k),
            weight_grad_dtype,
            compute_precision,
            need_weight=need_w13,
        )
        if grad_w13 is None:
            grad_w13 = skipped_gradient(
                h2,
                weight_grad_dtype
                if weight_grad_dtype is not None
                else w13_experts.dtype,
            )
        grad_logits = routing_gradient(
            grad_aux,
            grad_probability_sum,
            grad_route_weights,
            route_weights,
            route_ids,
            logits,
            routing_mode=str(routing_mode),
            lengths=lengths,
        )
        grad_logits_storage = grad_logits.to(logits.dtype)
        grad_router = (
            router_weight_gradient(
                grad_logits_storage,
                h2_flat,
                router_weight,
                weight_grad_dtype,
            )
            if need_router
            else skipped_gradient(router_weight, weight_grad_dtype)
        )
        add_router_input_gradient_(grad_h2, grad_logits_storage, router_weight)
        return grad_h2.reshape(h2.shape), grad_router, grad_w13


def _finish_forward(
    h,
    w2_experts,
    route_weights,
    offsets,
    reverse,
    residual,
    compute_precision="native",
    swiglu_limit=None,
):
    """Activate (clamped at ``swiglu_limit`` when given), project down, weight
    and combine with the residual."""
    with torch.no_grad():
        residual_flat = residual.reshape(-1, residual.shape[-1]).contiguous()
        in_fp8 = compute_precision in fp8.PRECISIONS
        output = down_projection(
            h if in_fp8 else activation(h, swiglu_limit),
            w2_experts,
            route_weights,
            offsets,
            reverse,
            residual_flat,
            compute_precision,
            from_preactivation=in_fp8,
            limit=swiglu_limit,
        )
        return output.reshape_as(residual)


def _finish_backward(
    grad_output,
    h,
    w2_experts,
    route_weights,
    offsets,
    gather,
    scatter,
    *,
    weight_grad_dtype=None,
    compute_precision="native",
    need_w2=True,
    swiglu_limit=None,
):
    """Return VJPs for ``h``, the expert-down weights, the route weights and
    the residual, the weights' at ``weight_grad_dtype`` (their own when
    ``None``); ``w2`` gets a placeholder when nobody asked for its gradient."""
    with torch.no_grad():
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        grad_h, grad_route_weights, grad_w2 = down_backward(
            grad_output_flat,
            h,
            w2_experts,
            route_weights,
            offsets,
            gather,
            scatter,
            weight_grad_dtype,
            compute_precision,
            need_weight=need_w2,
            limit=swiglu_limit,
        )
        if grad_w2 is None:
            grad_w2 = skipped_gradient(
                h,
                weight_grad_dtype
                if weight_grad_dtype is not None
                else w2_experts.dtype,
            )
        return grad_h, grad_w2, grad_route_weights, grad_output.clone()


@torch.library.custom_op("mlops::moe_sonicmoe_prepare_fwd", mutates_args=())
def _prepare_op(
    h2: torch.Tensor,
    router_weight: torch.Tensor,
    w13_experts: torch.Tensor,
    router_bias: torch.Tensor,
    top_k: int,
    routing_mode: str,
    n_group: int,
    topk_group: int,
    routed_scaling: float,
    lengths: list[int],
    weight_precision: str,
    weight_grad_dtype: torch.dtype | None,
    compute_precision: str,
    weight_components: list[torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del weight_grad_dtype
    (w13_experts,) = fp8.weights(compute_precision, weight_components, w13_experts)
    return _prepare_forward(
        h2,
        router_weight,
        w13_experts,
        router_bias,
        top_k=top_k,
        routing_mode=routing_mode,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling=routed_scaling,
        lengths=lengths,
        weight_precision=weight_precision,
        compute_precision=compute_precision,
    )


@_prepare_op.register_fake
def _prepare_fake(
    h2,
    router_weight,
    w13_experts,
    router_bias,
    top_k,
    routing_mode,
    n_group,
    topk_group,
    routed_scaling,
    lengths,
    weight_precision,
    weight_grad_dtype,
    compute_precision,
    weight_components,
):
    del router_bias, routing_mode, n_group, topk_group, routed_scaling, lengths
    del weight_grad_dtype, compute_precision, weight_components
    return (
        *statistic_fakes(h2, router_weight),
        *residual_fakes(h2, router_weight, w13_experts, top_k, weight_precision),
    )


@torch.library.custom_op("mlops::moe_sonicmoe_prepare_bwd", mutates_args=())
def _prepare_backward_op(
    grad_aux: torch.Tensor,
    grad_probability_sum: torch.Tensor,
    grad_route_weights: torch.Tensor,
    grad_h: torch.Tensor,
    h2: torch.Tensor,
    router_weight: torch.Tensor,
    w13_experts: torch.Tensor,
    logits: torch.Tensor,
    route_weights: torch.Tensor,
    route_ids: torch.Tensor,
    offsets: torch.Tensor,
    gather: torch.Tensor,
    reverse: torch.Tensor,
    top_k: int,
    routing_mode: str,
    lengths: list[int],
    weight_grad_dtype: torch.dtype | None,
    compute_precision: str,
    weight_components: list[torch.Tensor],
    need_router: bool,
    need_w13: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (w13_experts,) = fp8.weights(compute_precision, weight_components, w13_experts)
    return _prepare_backward(
        grad_aux,
        grad_probability_sum,
        grad_route_weights,
        grad_h,
        h2,
        router_weight,
        w13_experts,
        logits,
        route_weights,
        route_ids,
        offsets,
        gather,
        reverse,
        top_k=top_k,
        routing_mode=routing_mode,
        lengths=lengths,
        weight_grad_dtype=weight_grad_dtype,
        compute_precision=compute_precision,
        need_router=need_router,
        need_w13=need_w13,
    )


@_prepare_backward_op.register_fake
def _prepare_backward_fake(
    grad_aux,
    grad_probability_sum,
    grad_route_weights,
    grad_h,
    h2,
    router_weight,
    w13_experts,
    logits,
    route_weights,
    route_ids,
    offsets,
    gather,
    reverse,
    top_k,
    routing_mode,
    lengths,
    weight_grad_dtype,
    compute_precision,
    weight_components,
    need_router,
    need_w13,
):
    del grad_aux, grad_probability_sum, grad_route_weights, grad_h, logits
    del route_weights, route_ids, offsets, gather, reverse, top_k, routing_mode
    del lengths, compute_precision, weight_components
    return (
        torch.empty_like(h2),
        torch.empty_like(router_weight, dtype=weight_grad_dtype)
        if need_router
        else skipped_gradient(router_weight, weight_grad_dtype),
        torch.empty_like(w13_experts, dtype=weight_grad_dtype)
        if need_w13
        else skipped_gradient(w13_experts, weight_grad_dtype),
    )


@flop_formula(_prepare_op)
def _prepare_flops(
    h2, router_weight, w13_experts, router_bias, top_k, *_rest, out_val=None, **_kwargs
):
    del router_bias, out_val
    return logical.moe_prepare(
        h2, router_weight, w13_experts, top_k=top_k, entrypoint="forward"
    ).logical_flops


@flop_formula(_prepare_backward_op)
def _prepare_backward_flops(
    grad_aux,
    grad_probability_sum,
    grad_route_weights,
    grad_h,
    h2,
    router_weight,
    w13_experts,
    logits,
    route_weights,
    route_ids,
    offsets,
    gather,
    reverse,
    top_k,
    *_rest,
    out_val=None,
    **_kwargs,
):
    del grad_aux, grad_probability_sum, grad_route_weights, grad_h, logits
    del route_weights, route_ids, offsets, gather, reverse, out_val
    return logical.moe_prepare(
        h2, router_weight, w13_experts, top_k=top_k, entrypoint="backward"
    ).logical_flops


def _prepare_context(ctx, inputs, output):
    (
        h2,
        router_weight,
        w13_experts,
        _router_bias,
        top_k,
        routing_mode,
        _n_group,
        _topk_group,
        _routed_scaling,
        lengths,
        _weight_precision,
        weight_grad_dtype,
        compute_precision,
        weight_components,
    ) = inputs
    (
        _aux,
        counts,
        _probability_sum,
        logits,
        route_weights,
        route_ids,
        offsets,
        gather,
        scatter,
        reverse,
        _h,
    ) = output
    ctx.save_for_backward(
        h2,
        router_weight,
        w13_experts,
        logits,
        route_weights,
        route_ids,
        offsets,
        gather,
        reverse,
        *weight_components,
    )
    ctx.top_k = int(top_k)
    ctx.routing_mode = routing_mode
    ctx.lengths = list(lengths)
    ctx.weight_grad_dtype = weight_grad_dtype
    ctx.compute_precision = compute_precision
    ctx.mark_non_differentiable(
        counts, logits, route_ids, offsets, gather, scatter, reverse
    )


def _prepare_autograd_backward(
    ctx,
    grad_aux,
    _grad_counts,
    grad_probability_sum,
    _grad_logits,
    grad_route_weights,
    _grad_route_ids,
    _grad_offsets,
    _grad_gather,
    _grad_scatter,
    _grad_reverse,
    grad_h,
):
    residuals, components = ctx.saved_tensors[:9], ctx.saved_tensors[9:]
    h2, router_weight, w13_experts, _logits, route_weights, *_ = residuals
    if grad_aux is None:
        grad_aux = torch.zeros((), dtype=torch.float32, device=h2.device)
    if grad_probability_sum is None:
        grad_probability_sum = torch.zeros(
            router_weight.shape[0], dtype=torch.float32, device=h2.device
        )
    if grad_route_weights is None:
        grad_route_weights = torch.zeros_like(route_weights)
    if grad_h is None:
        grad_h = torch.zeros(
            (route_weights.numel(), w13_experts.shape[1]),
            dtype=h2.dtype,
            device=h2.device,
        )
    # Inputs: h2, router_weight, w13_experts, router_bias, ...
    needs = ctx.needs_input_grad
    grad_h2, grad_router, grad_w13 = _prepare_backward_op(
        grad_aux,
        grad_probability_sum,
        grad_route_weights,
        grad_h,
        *residuals,
        ctx.top_k,
        ctx.routing_mode,
        ctx.lengths,
        ctx.weight_grad_dtype,
        ctx.compute_precision,
        list(components),
        bool(needs[1]),
        bool(needs[2]),
    )
    return (
        grad_h2,
        requested(grad_router, needs[1]),
        requested(grad_w13, needs[2]),
        *([None] * 10),
        [None] * len(components),
    )


_prepare_op.register_autograd(
    _prepare_autograd_backward, setup_context=_prepare_context
)


@torch.library.custom_op("mlops::moe_sonicmoe_finish_fwd", mutates_args=())
def _finish_op(
    h: torch.Tensor,
    w2_experts: torch.Tensor,
    route_weights: torch.Tensor,
    offsets: torch.Tensor,
    gather: torch.Tensor,
    scatter: torch.Tensor,
    reverse: torch.Tensor,
    residual: torch.Tensor,
    top_k: int,
    weight_grad_dtype: torch.dtype | None,
    compute_precision: str,
    swiglu_limit: float | None,
    weight_components: list[torch.Tensor],
) -> torch.Tensor:
    del gather, scatter, top_k, weight_grad_dtype
    (w2_experts,) = fp8.weights(compute_precision, weight_components, w2_experts)
    return _finish_forward(
        h,
        w2_experts,
        route_weights,
        offsets,
        reverse,
        residual,
        compute_precision,
        swiglu_limit,
    )


@_finish_op.register_fake
def _finish_fake(
    h,
    w2_experts,
    route_weights,
    offsets,
    gather,
    scatter,
    reverse,
    residual,
    top_k,
    weight_grad_dtype,
    compute_precision,
    swiglu_limit,
    weight_components,
):
    del h, w2_experts, route_weights, offsets, gather, scatter, reverse, top_k
    del weight_grad_dtype, compute_precision, swiglu_limit, weight_components
    return torch.empty_like(residual)


@torch.library.custom_op("mlops::moe_sonicmoe_finish_bwd", mutates_args=())
def _finish_backward_op(
    grad_output: torch.Tensor,
    h: torch.Tensor,
    w2_experts: torch.Tensor,
    route_weights: torch.Tensor,
    offsets: torch.Tensor,
    gather: torch.Tensor,
    scatter: torch.Tensor,
    top_k: int,
    weight_grad_dtype: torch.dtype | None,
    compute_precision: str,
    swiglu_limit: float | None,
    weight_components: list[torch.Tensor],
    need_w2: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del top_k
    (w2_experts,) = fp8.weights(compute_precision, weight_components, w2_experts)
    return _finish_backward(
        grad_output,
        h,
        w2_experts,
        route_weights,
        offsets,
        gather,
        scatter,
        weight_grad_dtype=weight_grad_dtype,
        compute_precision=compute_precision,
        need_w2=need_w2,
        swiglu_limit=swiglu_limit,
    )


@_finish_backward_op.register_fake
def _finish_backward_fake(
    grad_output,
    h,
    w2_experts,
    route_weights,
    offsets,
    gather,
    scatter,
    top_k,
    weight_grad_dtype,
    compute_precision,
    swiglu_limit,
    weight_components,
    need_w2,
):
    del offsets, gather, scatter, top_k, compute_precision, swiglu_limit
    del weight_components
    return (
        torch.empty_like(h),
        torch.empty_like(w2_experts, dtype=weight_grad_dtype)
        if need_w2
        else skipped_gradient(w2_experts, weight_grad_dtype),
        torch.empty_like(route_weights),
        torch.empty_like(grad_output),
    )


@flop_formula(_finish_op)
def _finish_flops(h, w2_experts, *_rest, out_val=None, **_kwargs):
    del out_val
    return logical.moe_finish(h, w2_experts, entrypoint="forward").logical_flops


@flop_formula(_finish_backward_op)
def _finish_backward_flops(grad_output, h, w2_experts, *_rest, out_val=None, **_kwargs):
    del grad_output, out_val
    return logical.moe_finish(h, w2_experts, entrypoint="backward").logical_flops


def _finish_context(ctx, inputs, output):
    del output
    (
        h,
        w2_experts,
        route_weights,
        offsets,
        gather,
        scatter,
        _reverse,
        _residual,
        top_k,
        weight_grad_dtype,
        compute_precision,
        swiglu_limit,
        weight_components,
    ) = inputs
    ctx.save_for_backward(
        h, w2_experts, route_weights, offsets, gather, scatter, *weight_components
    )
    ctx.top_k = int(top_k)
    ctx.weight_grad_dtype = weight_grad_dtype
    ctx.compute_precision = compute_precision
    ctx.swiglu_limit = swiglu_limit


def _finish_autograd_backward(ctx, grad_output):
    residuals, components = ctx.saved_tensors[:6], ctx.saved_tensors[6:]
    # Inputs: h, w2_experts, route_weights, offsets, gather, scatter, reverse, residual, ...
    needs = ctx.needs_input_grad
    grad_h, grad_w2, grad_route_weights, grad_residual = _finish_backward_op(
        grad_output,
        *residuals,
        ctx.top_k,
        ctx.weight_grad_dtype,
        ctx.compute_precision,
        ctx.swiglu_limit,
        list(components),
        bool(needs[1]),
    )
    return (
        grad_h,
        requested(grad_w2, needs[1]),
        grad_route_weights,
        None,
        None,
        None,
        None,
        grad_residual,
        None,
        None,
        None,
        None,
        [None] * len(components),
    )


_finish_op.register_autograd(_finish_autograd_backward, setup_context=_finish_context)
fp8.register_container_dispatch(
    _prepare_op, _prepare_backward_op, _finish_op, _finish_backward_op
)


def apply(
    h2,
    residual,
    router_weight,
    w13_experts,
    w2_experts,
    *,
    top_k,
    routing_mode,
    router_bias=None,
    n_group=1,
    topk_group=1,
    routed_scaling=1.0,
    lengths=None,
    shared_w13=None,
    shared_w2=None,
    shared_gate_weight=None,
    shared_mode="add",
    compute_precision=None,
    swiglu_limit=None,
):
    rows = h2.numel() // h2.shape[-1]
    normalized_lengths = (rows,) if lengths is None else tuple(int(x) for x in lengths)
    precision, w13_experts, w2_experts, grad_dtype = fp8.prepare(
        compute_precision, w13_experts, w2_experts, h2.dtype
    )
    bias = (
        torch.empty(0, dtype=h2.dtype, device=h2.device)
        if router_bias is None
        else router_bias
    )
    base = _production_shared_base(
        h2,
        residual,
        shared_w13,
        shared_w2,
        shared_gate_weight,
        shared_mode,
        swiglu_limit,
    )
    (
        aux,
        counts,
        probability_sum,
        _logits,
        route_weights,
        _route_ids,
        offsets,
        gather,
        scatter,
        reverse,
        h,
    ) = _prepare_op(
        h2,
        router_weight,
        w13_experts,
        bias,
        int(top_k),
        routing_mode,
        int(n_group),
        int(topk_group),
        float(routed_scaling),
        list(normalized_lengths),
        current_route_weight_precision(),
        grad_dtype,
        precision,
        [],
    )
    output = _finish_op(
        h,
        w2_experts,
        route_weights,
        offsets,
        gather,
        scatter,
        reverse,
        base,
        int(top_k),
        grad_dtype,
        precision,
        None if swiglu_limit is None else float(swiglu_limit),
        [],
    )
    return output, aux, counts, probability_sum


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="moe",
        implementation_id="sonicmoe.moe.composed",
        provider="sonicmoe",
        priority=110,
        deterministic=True,
        supports=_supports,
        apply=apply,
    )
)


__all__ = ["IMPLEMENTATION", "apply"]
