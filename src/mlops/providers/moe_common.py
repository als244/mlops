"""Shared stateless routed-MoE mechanics for the builtin and SonicMoE adapters."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..kernels.matmul import product_at
from ..kernels.moe_dispatch import (
    combine,
    dispatch,
    dispatch_backward,
    row_dot,
    scale_rows_,
    sort_assignments,
)
from ..kernels.moe_grouped_gemm import (
    grouped_mm_dgrad,
    grouped_mm_forward,
    grouped_mm_wgrad,
)
from ..kernels.moe_router import (
    add_aux_gradient_,
    add_sequence_aux_gradient_,
    route,
    route_backward,
)
from ..kernels.swiglu import swiglu_packed_backward, swiglu_packed_forward


def routing_statistics(logits, route_ids, *, routing_mode, lengths, experts):
    """Return ``(aux, counts, probability_sum)`` of one routing decision.

    ``sigmoid_noaux_tc`` balances every sequence of ``lengths`` on its own;
    the softmax modes balance the whole batch.
    """
    rows = logits.shape[0]
    if routing_mode == "sigmoid_noaux_tc":
        scores = torch.sigmoid(logits.float())
        probabilities = scores / scores.sum(-1, keepdim=True)
    else:
        probabilities = torch.softmax(logits.float(), dim=-1)
    probability_sum = probabilities.sum(0)
    counts = torch.zeros(experts, dtype=torch.int64, device=logits.device)
    counts.scatter_add_(
        0,
        route_ids.reshape(-1).long(),
        torch.ones(route_ids.numel(), dtype=torch.int64, device=logits.device),
    )
    if routing_mode == "sigmoid_noaux_tc":
        aux = torch.zeros((), dtype=torch.float32, device=logits.device)
        start = 0
        for length in lengths:
            stop = start + int(length)
            segment_counts = torch.zeros_like(counts)
            segment_ids = route_ids[start:stop].reshape(-1).long()
            segment_counts.scatter_add_(
                0,
                segment_ids,
                torch.ones_like(segment_ids, dtype=torch.int64),
            )
            frequency = (
                segment_counts.float() * experts / (route_ids.shape[1] * int(length))
            )
            aux += (frequency * probabilities[start:stop].mean(0)).sum()
            start = stop
    else:
        frequency = counts.float() / route_ids.numel()
        aux = experts * (frequency * (probability_sum / rows)).sum()
    return aux, counts, probability_sum


def routing_gradient(
    grad_aux,
    grad_probability_sum,
    grad_route_weights,
    route_weights,
    route_ids,
    logits,
    *,
    routing_mode,
    lengths,
):
    """Return the fp32 logits gradient from the cotangents of the route
    weights, the auxiliary loss and the probability sum.

    The probability sum is the softmax's column sum in the softmax modes and
    the normalized sigmoid scores' in ``sigmoid_noaux_tc``, so its cotangent
    goes back through the matching Jacobian: ``p (g - <g, p>)`` for the
    softmax, ``(1 - s) p (g - <g, p>)`` for the normalized sigmoid ``p = s /
    sum(s)``.
    """
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
            grad_logits,
            logits,
            route_ids,
            grad_aux,
            normalized_lengths,
        )
    else:
        add_aux_gradient_(grad_logits, logits, route_ids, grad_aux)
    if routing_mode == "sigmoid_noaux_tc":
        scores = torch.sigmoid(logits.float())
        probabilities = scores / scores.sum(-1, keepdim=True)
        jacobian_scale = (1.0 - scores) * probabilities
    else:
        probabilities = torch.softmax(logits.float(), dim=-1)
        jacobian_scale = probabilities
    grad_sum = grad_probability_sum.float()
    grad_logits.add_(
        jacobian_scale
        * (grad_sum.unsqueeze(0) - (probabilities @ grad_sum).unsqueeze(1))
    )
    return grad_logits


def skipped_gradient(like, dtype=None):
    """The placeholder a backward operator returns for a gradient nobody asked
    for (its parameter's ``needs_input_grad`` is false): an empty tensor at
    the gradient's dtype, which the autograd wrapper turns into ``None``.
    Operators return tensors, so the GEMM is skipped and this stands in."""
    return torch.empty(
        0, dtype=like.dtype if dtype is None else dtype, device=like.device
    )


def requested(gradient, needed):
    """``gradient`` when its parameter asked for one, else ``None``."""
    return gradient if needed else None


def router_weight_gradient(grad_logits, router_input, router_weight, weight_grad_dtype):
    """``grad_logits.T @ router_input`` at ``weight_grad_dtype`` (the router's own
    when ``None``), in ``router_weight``'s strides.

    A custom VJP result must honor the advertised input layout, not merely the
    layout the GEMM produces: a caller supplying a view of its parameter gets
    the gradient in that view's strides, so AOT's inverse view restores the
    ordinary contiguous parameter-gradient ABI.
    """
    grad_router = (
        grad_logits.T @ router_input
        if weight_grad_dtype is None
        else product_at(grad_logits.T, router_input, weight_grad_dtype)
    )
    if grad_router.stride() != router_weight.stride():
        aligned = torch.empty_strided(
            grad_router.shape,
            router_weight.stride(),
            dtype=grad_router.dtype,
            device=grad_router.device,
        )
        aligned.copy_(grad_router)
        grad_router = aligned
    return grad_router


def add_router_input_gradient_(grad_h2, grad_logits_storage, router_weight):
    """Add the router's share of the hidden-input gradient in place."""
    if router_weight.dtype == grad_h2.dtype:
        grad_h2.addmm_(grad_logits_storage, router_weight)
    else:
        grad_h2.add_((grad_logits_storage @ router_weight).to(grad_h2.dtype))


def forward_routed(
    h2: torch.Tensor,
    residual: torch.Tensor,
    router_weight: torch.Tensor,
    w13_experts: torch.Tensor,
    w2_experts: torch.Tensor,
    *,
    top_k: int,
    routing_mode: str,
    router_bias: torch.Tensor | None = None,
    n_group: int = 1,
    topk_group: int = 1,
    routed_scaling: float = 1.0,
    lengths: Sequence[int] | None = None,
    weight_precision: str = "float32",
    swiglu_variant: str = "triton",
    use_routing_override: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Return output, auxiliary values, and the complete routed VJP residuals.

    The ordered result is ``(output, aux, counts, probability_sum, logits,
    route_weights, route_ids, order, offsets, slots, h13)``. Every weight is in
    nn.Linear orientation: ``router_weight [E, D]``, ``w13_experts [E, 2F, D]``
    with gate rows then up rows, ``w2_experts [E, D, F]``.

    ``use_routing_override=False`` is the artifact-compilation path: it fixes
    the ordinary no-override behavior without asking Dynamo to trace the
    development-only ``ContextVar``. Model-facing provider calls retain the
    context-local routing ablation by default.
    """
    del swiglu_variant
    rows = h2.numel() // h2.shape[-1]
    normalized_lengths = (rows,) if lengths is None else tuple(map(int, lengths))
    if (
        not normalized_lengths
        or any(length <= 0 for length in normalized_lengths)
        or sum(normalized_lengths) != rows
    ):
        raise ValueError("MoE lengths must be positive and cover every hidden row")
    bias = (
        torch.empty(0, dtype=h2.dtype, device=h2.device)
        if router_bias is None
        else router_bias
    )
    experts = router_weight.shape[0]
    with torch.no_grad():
        original_shape = h2.shape
        h2_flat = h2.reshape(-1, h2.shape[-1]).contiguous()
        residual_flat = residual.reshape_as(h2_flat).contiguous()
        router_input = h2_flat.to(router_weight.dtype)
        logits = router_input @ router_weight.T
        route_weights, route_ids = route(
            logits,
            int(top_k),
            str(routing_mode),
            bias=None if bias.numel() == 0 else bias,
            n_group=int(n_group),
            topk_group=int(topk_group),
            routed_scaling=float(routed_scaling),
            weight_precision=str(weight_precision),
            use_context_override=bool(use_routing_override),
        )
        order, offsets, slots = sort_assignments(route_ids, experts)
        dispatched = dispatch(h2_flat, order, int(top_k))
        h13 = grouped_mm_forward(dispatched, w13_experts, offsets)
        activated = swiglu_packed_forward(h13)
        expert_output = grouped_mm_forward(activated, w2_experts, offsets)
        output = combine(
            expert_output,
            slots,
            route_weights,
            residual_flat,
        ).reshape(original_shape)
        aux, counts, probability_sum = routing_statistics(
            logits,
            route_ids,
            routing_mode=str(routing_mode),
            lengths=normalized_lengths,
            experts=experts,
        )
        return (
            output,
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


def backward_routed(
    grad_output: torch.Tensor,
    grad_aux: torch.Tensor,
    grad_probability_sum: torch.Tensor,
    h2: torch.Tensor,
    router_weight: torch.Tensor,
    w13_experts: torch.Tensor,
    w2_experts: torch.Tensor,
    logits: torch.Tensor,
    route_weights: torch.Tensor,
    route_ids: torch.Tensor,
    order: torch.Tensor,
    offsets: torch.Tensor,
    slots: torch.Tensor,
    h13: torch.Tensor,
    *,
    top_k: int,
    routing_mode: str,
    lengths: Sequence[int],
    swiglu_variant: str = "triton",
    weight_grad_dtype: torch.dtype | None = None,
    need_router: bool = True,
    need_w13: bool = True,
    need_w2: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return VJPs for ``h2, residual, router_weight, w13, w2`` in order.

    The router's and the expert weights' VJPs are summed at fp32 and returned
    at ``weight_grad_dtype`` (their own dtypes when ``None``).
    """
    with torch.no_grad():
        original_shape = h2.shape
        h2_flat = h2.reshape(-1, h2.shape[-1])
        grad_output_flat = grad_output.reshape_as(h2_flat).contiguous()
        dispatched = dispatch(h2_flat, order, int(top_k))
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
        if need_w2:
            scale_rows_(dispatched_grad, route_scale)
            grad_w2 = grouped_mm_wgrad(
                activated,
                dispatched_grad,
                offsets,
                tuple(w2_experts.shape),
                weight_grad_dtype,
            )
        else:
            grad_w2 = skipped_gradient(w2_experts, weight_grad_dtype)
        scale_rows_(grad_activated, route_scale)
        grad_h13 = swiglu_packed_backward(
            grad_activated,
            h13,
            variant=str(swiglu_variant),
        )
        if need_w13:
            grad_w13 = grouped_mm_wgrad(
                dispatched,
                grad_h13,
                offsets,
                tuple(w13_experts.shape),
                weight_grad_dtype,
            )
        else:
            grad_w13 = skipped_gradient(w13_experts, weight_grad_dtype)
        grad_dispatched = grouped_mm_dgrad(grad_h13, w13_experts, offsets)
        grad_h2 = dispatch_backward(grad_dispatched, slots)

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
                h2_flat.to(router_weight.dtype),
                router_weight,
                weight_grad_dtype,
            )
            if need_router
            else skipped_gradient(router_weight, weight_grad_dtype)
        )
        add_router_input_gradient_(grad_h2, grad_logits_storage, router_weight)
        return (
            grad_h2.reshape(original_shape),
            grad_output.clone(),
            grad_router,
            grad_w13,
            grad_w2,
        )


__all__ = [
    "add_router_input_gradient_",
    "backward_routed",
    "forward_routed",
    "router_weight_gradient",
    "routing_gradient",
    "routing_statistics",
]
