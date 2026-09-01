"""Trace-visible routed-MoE composition built from two opaque kernel regions.

The public :func:`mlops.moe` operation remains unchanged.  This provider keeps
the residual-producing routing/up-projection region separate from the
activation/down-projection/combine region so AOTAutograd can rematerialize the
former without replaying the latter.
"""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.moe_dispatch import (
    combine,
    dispatch,
    dispatch_backward,
    row_dot,
    scale_rows_,
    sort_assignments,
)
from ...kernels.moe_grouped_gemm import (
    grouped_mm_dgrad,
    grouped_mm_forward,
    grouped_mm_wgrad,
)
from ...kernels.moe_router import (
    add_aux_gradient_,
    add_sequence_aux_gradient_,
    current_route_weight_precision,
    route,
    route_backward,
)
from ...kernels.swiglu import (
    swiglu_packed_backward,
    swiglu_packed_backward_out,
    swiglu_packed_forward,
)
from ...moe import _production_shared_base
from .moe import supports as common_support


def _supports(*args, surface, **kwargs):
    if surface == "explicit":
        return SupportResult.no(
            "trace-visible MoE composition is a semantic-only implementation"
        )
    return common_support(*args, surface=surface, **kwargs)


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
):
    """Route and compute ``h13``, the final value needed before expert down."""
    rows = h2.numel() // h2.shape[-1]
    normalized_lengths = tuple(int(length) for length in lengths)
    if (
        not normalized_lengths
        or any(length <= 0 for length in normalized_lengths)
        or sum(normalized_lengths) != rows
    ):
        raise ValueError("MoE lengths must be positive and cover every hidden row")

    with torch.no_grad():
        h2_flat = h2.reshape(-1, h2.shape[-1]).contiguous()
        logits = h2_flat.to(router_weight.dtype) @ router_weight
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
        order, offsets, slots = sort_assignments(route_ids, router_weight.shape[1])
        dispatched = dispatch(h2_flat, order, int(top_k))
        h13 = grouped_mm_forward(dispatched, w13_experts, offsets)

        if routing_mode == "sigmoid_noaux_tc":
            scores = torch.sigmoid(logits.float())
            probabilities = scores / scores.sum(-1, keepdim=True)
        else:
            probabilities = torch.softmax(logits.float(), dim=-1)
        probability_sum = probabilities.sum(0)
        counts = torch.zeros(
            router_weight.shape[1], dtype=torch.int64, device=h2.device
        )
        counts.scatter_add_(
            0,
            route_ids.reshape(-1).long(),
            torch.ones(route_ids.numel(), dtype=torch.int64, device=h2.device),
        )
        if routing_mode == "sigmoid_noaux_tc":
            aux = torch.zeros((), dtype=torch.float32, device=h2.device)
            start = 0
            for length in normalized_lengths:
                stop = start + length
                segment_counts = torch.zeros_like(counts)
                segment_ids = route_ids[start:stop].reshape(-1).long()
                segment_counts.scatter_add_(
                    0,
                    segment_ids,
                    torch.ones_like(segment_ids, dtype=torch.int64),
                )
                frequency = (
                    segment_counts.float()
                    * router_weight.shape[1]
                    / (route_ids.shape[1] * length)
                )
                aux += (frequency * probabilities[start:stop].mean(0)).sum()
                start = stop
        else:
            frequency = counts.float() / route_ids.numel()
            aux = (
                router_weight.shape[1]
                * (frequency * (probability_sum / h2_flat.shape[0])).sum()
            )
        return (
            aux,
            counts,
            probability_sum,
            logits,
            route_weights,
            route_ids,
            order,
            offsets,
            slots,
            h13,
        )


def _prepare_backward(
    grad_aux,
    grad_probability_sum,
    grad_route_weights,
    grad_h13,
    h2,
    router_weight,
    w13_experts,
    logits,
    route_weights,
    route_ids,
    order,
    offsets,
    slots,
    *,
    top_k,
    routing_mode,
    lengths,
):
    """Return VJPs for the preparation region's differentiable inputs."""
    with torch.no_grad():
        original_shape = h2.shape
        h2_flat = h2.reshape(-1, h2.shape[-1])
        dispatched = dispatch(h2_flat, order, int(top_k))
        grad_w13 = grouped_mm_wgrad(
            dispatched, grad_h13, offsets, tuple(w13_experts.shape)
        )
        grad_dispatched = grouped_mm_dgrad(grad_h13, w13_experts, offsets)
        grad_h2 = dispatch_backward(grad_dispatched, slots)

        grad_logits = route_backward(
            grad_route_weights,
            route_weights,
            route_ids,
            logits,
            str(routing_mode),
        )
        normalized_lengths = [int(length) for length in lengths]
        if routing_mode == "sigmoid_noaux_tc":
            add_sequence_aux_gradient_(
                grad_logits, logits, route_ids, grad_aux, normalized_lengths
            )
        else:
            add_aux_gradient_(grad_logits, logits, route_ids, grad_aux)
        probabilities = torch.softmax(logits.float(), dim=-1)
        grad_sum = grad_probability_sum.float()
        grad_logits.add_(
            probabilities
            * (grad_sum.unsqueeze(0) - (probabilities @ grad_sum).unsqueeze(1))
        )
        grad_logits_storage = grad_logits.to(logits.dtype)
        grad_router = h2_flat.to(router_weight.dtype).T @ grad_logits_storage
        if grad_router.stride() != router_weight.stride():
            aligned_grad_router = torch.empty_strided(
                grad_router.shape,
                router_weight.stride(),
                dtype=grad_router.dtype,
                device=grad_router.device,
            )
            aligned_grad_router.copy_(grad_router)
            grad_router = aligned_grad_router
        if router_weight.dtype == grad_h2.dtype:
            grad_h2.addmm_(grad_logits_storage, router_weight.T)
        else:
            grad_h2.add_((grad_logits_storage @ router_weight.T).to(grad_h2.dtype))
        return grad_h2.reshape(original_shape), grad_router, grad_w13


def _finish_forward(
    h13,
    w2_experts,
    route_weights,
    offsets,
    slots,
    residual,
):
    """Apply expert activation/down projection and combine with the residual."""
    with torch.no_grad():
        activated = swiglu_packed_forward(h13)
        expert_output = grouped_mm_forward(activated, w2_experts, offsets)
        return combine(
            expert_output,
            slots,
            route_weights,
            residual.reshape(-1, residual.shape[-1]).contiguous(),
        ).reshape_as(residual)


def _finish_backward(
    grad_output,
    h13,
    w2_experts,
    route_weights,
    order,
    offsets,
    slots,
    *,
    top_k,
    grad_h13_destination=None,
):
    """Return VJPs for ``h13``, expert-down weights, routes, and residual."""
    with torch.no_grad():
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        dispatched_grad = dispatch(grad_output_flat, order, int(top_k))
        activated = swiglu_packed_forward(h13)
        grad_activated = grouped_mm_dgrad(dispatched_grad, w2_experts, offsets)
        grad_weight_slot = row_dot(grad_activated, activated)
        grad_route_weights = (
            grad_weight_slot[slots.reshape(-1).long()]
            .view_as(route_weights)
            .contiguous()
        )

        route_scale = route_weights.reshape(-1)[order.long()].float().contiguous()
        scale_rows_(dispatched_grad, route_scale)
        grad_w2 = grouped_mm_wgrad(
            activated, dispatched_grad, offsets, tuple(w2_experts.shape)
        )
        scale_rows_(grad_activated, route_scale)
        if grad_h13_destination is None:
            grad_h13 = swiglu_packed_backward(
                grad_activated, h13, variant="triton"
            )
        else:
            grad_h13 = swiglu_packed_backward_out(
                grad_activated,
                h13,
                grad_h13_destination,
                variant="triton",
            )
        return grad_h13, grad_w2, grad_route_weights, grad_output.clone()


@torch.library.custom_op(
    "mlops::moe_builtin_grouped_gemm_prepare_fwd", mutates_args=()
)
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
]:
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
):
    del router_bias, routing_mode, n_group, topk_group, routed_scaling, lengths
    rows = h2.numel() // h2.shape[-1]
    experts = router_weight.shape[1]
    assignments = rows * int(top_k)
    route_dtype = torch.float32 if weight_precision == "float32" else h2.dtype
    return (
        torch.empty((), dtype=torch.float32, device=h2.device),
        torch.empty(experts, dtype=torch.int64, device=h2.device),
        torch.empty(experts, dtype=torch.float32, device=h2.device),
        torch.empty((rows, experts), dtype=router_weight.dtype, device=h2.device),
        torch.empty((rows, int(top_k)), dtype=route_dtype, device=h2.device),
        torch.empty((rows, int(top_k)), dtype=torch.int32, device=h2.device),
        torch.empty(assignments, dtype=torch.int32, device=h2.device),
        torch.empty(experts + 1, dtype=torch.int32, device=h2.device),
        torch.empty((rows, int(top_k)), dtype=torch.int32, device=h2.device),
        torch.empty(
            (assignments, w13_experts.shape[2]), dtype=h2.dtype, device=h2.device
        ),
    )


@torch.library.custom_op(
    "mlops::moe_builtin_grouped_gemm_prepare_bwd", mutates_args=()
)
def _prepare_backward_op(
    grad_aux: torch.Tensor,
    grad_probability_sum: torch.Tensor,
    grad_route_weights: torch.Tensor,
    grad_h13: torch.Tensor,
    h2: torch.Tensor,
    router_weight: torch.Tensor,
    w13_experts: torch.Tensor,
    logits: torch.Tensor,
    route_weights: torch.Tensor,
    route_ids: torch.Tensor,
    order: torch.Tensor,
    offsets: torch.Tensor,
    slots: torch.Tensor,
    top_k: int,
    routing_mode: str,
    lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _prepare_backward(
        grad_aux,
        grad_probability_sum,
        grad_route_weights,
        grad_h13,
        h2,
        router_weight,
        w13_experts,
        logits,
        route_weights,
        route_ids,
        order,
        offsets,
        slots,
        top_k=top_k,
        routing_mode=routing_mode,
        lengths=lengths,
    )


@_prepare_backward_op.register_fake
def _prepare_backward_fake(
    grad_aux,
    grad_probability_sum,
    grad_route_weights,
    grad_h13,
    h2,
    router_weight,
    w13_experts,
    logits,
    route_weights,
    route_ids,
    order,
    offsets,
    slots,
    top_k,
    routing_mode,
    lengths,
):
    del (
        grad_aux,
        grad_probability_sum,
        grad_route_weights,
        grad_h13,
        logits,
        route_weights,
        route_ids,
        order,
        offsets,
        slots,
        top_k,
        routing_mode,
        lengths,
    )
    return torch.empty_like(h2), torch.empty_like(router_weight), torch.empty_like(
        w13_experts
    )


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
    ) = inputs
    (
        _aux,
        counts,
        _probability_sum,
        logits,
        route_weights,
        route_ids,
        order,
        offsets,
        slots,
        _h13,
    ) = output
    ctx.save_for_backward(
        h2,
        router_weight,
        w13_experts,
        logits,
        route_weights,
        route_ids,
        order,
        offsets,
        slots,
    )
    ctx.top_k = int(top_k)
    ctx.routing_mode = routing_mode
    ctx.lengths = list(lengths)
    ctx.mark_non_differentiable(counts, logits, route_ids, order, offsets, slots)


def _prepare_autograd_backward(
    ctx,
    grad_aux,
    _grad_counts,
    grad_probability_sum,
    _grad_logits,
    grad_route_weights,
    _grad_route_ids,
    _grad_order,
    _grad_offsets,
    _grad_slots,
    grad_h13,
):
    h2, router_weight, w13_experts, _logits, route_weights, *_ = ctx.saved_tensors
    if grad_aux is None:
        grad_aux = torch.zeros((), dtype=torch.float32, device=h2.device)
    if grad_probability_sum is None:
        grad_probability_sum = torch.zeros(
            router_weight.shape[1], dtype=torch.float32, device=h2.device
        )
    if grad_route_weights is None:
        grad_route_weights = torch.zeros_like(route_weights)
    if grad_h13 is None:
        grad_h13 = torch.zeros(
            (route_weights.numel(), w13_experts.shape[2]),
            dtype=h2.dtype,
            device=h2.device,
        )
    gradients = _prepare_backward_op(
        grad_aux,
        grad_probability_sum,
        grad_route_weights,
        grad_h13,
        *ctx.saved_tensors,
        ctx.top_k,
        ctx.routing_mode,
        ctx.lengths,
    )
    return (*gradients, None, None, None, None, None, None, None, None)


_prepare_op.register_autograd(
    _prepare_autograd_backward, setup_context=_prepare_context
)


@torch.library.custom_op(
    "mlops::moe_builtin_grouped_gemm_finish_fwd", mutates_args=()
)
def _finish_op(
    h13: torch.Tensor,
    w2_experts: torch.Tensor,
    route_weights: torch.Tensor,
    order: torch.Tensor,
    offsets: torch.Tensor,
    slots: torch.Tensor,
    residual: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    del order, top_k
    return _finish_forward(h13, w2_experts, route_weights, offsets, slots, residual)


@_finish_op.register_fake
def _finish_fake(
    h13, w2_experts, route_weights, order, offsets, slots, residual, top_k
):
    del h13, w2_experts, route_weights, order, offsets, slots, top_k
    return torch.empty_like(residual)


@torch.library.custom_op(
    "mlops::moe_builtin_grouped_gemm_finish_bwd", mutates_args=()
)
def _finish_backward_op(
    grad_output: torch.Tensor,
    h13: torch.Tensor,
    w2_experts: torch.Tensor,
    route_weights: torch.Tensor,
    order: torch.Tensor,
    offsets: torch.Tensor,
    slots: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _finish_backward(
        grad_output,
        h13,
        w2_experts,
        route_weights,
        order,
        offsets,
        slots,
        top_k=top_k,
    )


@_finish_backward_op.register_fake
def _finish_backward_fake(
    grad_output, h13, w2_experts, route_weights, order, offsets, slots, top_k
):
    del order, offsets, slots, top_k
    return (
        torch.empty_like(h13),
        torch.empty_like(w2_experts),
        torch.empty_like(route_weights),
        torch.empty_like(grad_output),
    )


@torch.library.custom_op(
    "mlops::moe_builtin_grouped_gemm_finish_bwd_donate_h13",
    mutates_args=("h13",),
)
def _finish_backward_donate_h13_op(
    grad_output: torch.Tensor,
    h13: torch.Tensor,
    w2_experts: torch.Tensor,
    route_weights: torch.Tensor,
    order: torch.Tensor,
    offsets: torch.Tensor,
    slots: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deployment-only VJP that donates a proven-dead ``h13`` allocation.

    The ordinary autograd operator above remains functional and supports
    retained/repeated backward.  A compiler may select this lower-level ABI
    only after proving that ``h13`` has no later value consumer.  The mutated
    tensor then contains ``grad_h13``; the three returned values correspond to
    the remaining functional results.
    """
    _grad_h13, grad_w2, grad_route_weights, grad_residual = _finish_backward(
        grad_output,
        h13,
        w2_experts,
        route_weights,
        order,
        offsets,
        slots,
        top_k=top_k,
        grad_h13_destination=h13,
    )
    return grad_w2, grad_route_weights, grad_residual


@_finish_backward_donate_h13_op.register_fake
def _finish_backward_donate_h13_fake(
    grad_output, h13, w2_experts, route_weights, order, offsets, slots, top_k
):
    del h13, order, offsets, slots, top_k
    return (
        torch.empty_like(w2_experts),
        torch.empty_like(route_weights),
        torch.empty_like(grad_output),
    )


def _finish_context(ctx, inputs, output):
    del output
    h13, w2_experts, route_weights, order, offsets, slots, _residual, top_k = inputs
    ctx.save_for_backward(h13, w2_experts, route_weights, order, offsets, slots)
    ctx.top_k = int(top_k)


def _finish_autograd_backward(ctx, grad_output):
    gradients = _finish_backward_op(
        grad_output, *ctx.saved_tensors, ctx.top_k
    )
    grad_h13, grad_w2, grad_route_weights, grad_residual = gradients
    return (
        grad_h13,
        grad_w2,
        grad_route_weights,
        None,
        None,
        None,
        grad_residual,
        None,
    )


_finish_op.register_autograd(_finish_autograd_backward, setup_context=_finish_context)


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
):
    rows = h2.numel() // h2.shape[-1]
    normalized_lengths = (rows,) if lengths is None else tuple(int(x) for x in lengths)
    bias = (
        torch.empty(0, dtype=h2.dtype, device=h2.device)
        if router_bias is None
        else router_bias
    )
    base = _production_shared_base(
        h2, residual, shared_w13, shared_w2, shared_gate_weight, shared_mode
    )
    (
        aux,
        counts,
        probability_sum,
        _logits,
        route_weights,
        _route_ids,
        order,
        offsets,
        slots,
        h13,
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
    )
    output = _finish_op(
        h13,
        w2_experts,
        route_weights,
        order,
        offsets,
        slots,
        base,
        int(top_k),
    )
    return output, aux, counts, probability_sum


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="moe",
        implementation_id="builtin.moe.grouped_gemm_composed",
        provider="builtin",
        priority=100,
        deterministic=True,
        supports=_supports,
        apply=apply,
    )
)


__all__ = ["IMPLEMENTATION", "apply"]
