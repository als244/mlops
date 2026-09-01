"""Shared stateless routed-MoE mechanics for builtin and ScatterMoE adapters."""

from __future__ import annotations

from collections.abc import Sequence

import torch

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


def _forward_with_engine(
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
    expert_engine: str,
    use_routing_override: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Return output, auxiliary values, and the complete routed VJP residuals.

    The ordered result is ``(output, aux, counts, probability_sum, logits,
    route_weights, route_ids, order, offsets, slots, h13)``.

    ``use_routing_override=False`` is the artifact-compilation path: it fixes
    the ordinary no-override behavior without asking Dynamo to trace the
    development-only ``ContextVar``. Model-facing provider calls retain the
    context-local routing ablation by default.
    """
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
    with torch.no_grad():
        original_shape = h2.shape
        h2_flat = h2.reshape(-1, h2.shape[-1]).contiguous()
        residual_flat = residual.reshape_as(h2_flat).contiguous()
        router_input = h2_flat.to(router_weight.dtype)
        logits = router_input @ router_weight
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
        order, offsets, slots = sort_assignments(route_ids, router_weight.shape[1])
        if expert_engine == "scattermoe":
            from .scattermoe.engine import forward as expert_forward

            h13, expert_output = expert_forward(
                h2_flat,
                w13_experts,
                w2_experts,
                route_ids,
                order,
                offsets,
                int(top_k),
            )
        elif expert_engine == "builtin":
            dispatched = dispatch(h2_flat, order, int(top_k))
            h13 = grouped_mm_forward(dispatched, w13_experts, offsets)
            activated = swiglu_packed_forward(h13)
            expert_output = grouped_mm_forward(activated, w2_experts, offsets)
        else:
            raise ValueError(f"unknown private expert engine {expert_engine!r}")
        output = combine(
            expert_output,
            slots,
            route_weights,
            residual_flat,
        ).reshape(original_shape)

        if routing_mode == "sigmoid_noaux_tc":
            scores = torch.sigmoid(logits.float())
            probabilities = scores / scores.sum(-1, keepdim=True)
        else:
            probabilities = torch.softmax(logits.float(), dim=-1)
        probability_sum = probabilities.sum(0)
        counts = torch.zeros(
            router_weight.shape[1],
            dtype=torch.int64,
            device=h2.device,
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
                stop = start + int(length)
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
                    / (route_ids.shape[1] * int(length))
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


def _backward_with_engine(
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
    expert_engine: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return VJPs for ``h2, residual, router_weight, w13, w2`` in order."""
    with torch.no_grad():
        original_shape = h2.shape
        h2_flat = h2.reshape(-1, h2.shape[-1])
        grad_output_flat = grad_output.reshape_as(h2_flat).contiguous()
        if expert_engine == "scattermoe":
            from .scattermoe.engine import backward as expert_backward

            grad_h2, grad_route_weights, grad_w13, grad_w2 = expert_backward(
                grad_output_flat,
                h2_flat,
                w13_experts,
                w2_experts,
                route_weights,
                route_ids,
                order,
                offsets,
                slots,
                h13,
                int(top_k),
                swiglu_variant=str(swiglu_variant),
            )
        elif expert_engine == "builtin":
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
            scale_rows_(dispatched_grad, route_scale)
            grad_w2 = grouped_mm_wgrad(
                activated,
                dispatched_grad,
                offsets,
                tuple(w2_experts.shape),
            )
            scale_rows_(grad_activated, route_scale)
            grad_h13 = swiglu_packed_backward(
                grad_activated,
                h13,
                variant=str(swiglu_variant),
            )
            grad_w13 = grouped_mm_wgrad(
                dispatched,
                grad_h13,
                offsets,
                tuple(w13_experts.shape),
            )
            grad_dispatched = grouped_mm_dgrad(grad_h13, w13_experts, offsets)
            grad_h2 = dispatch_backward(grad_dispatched, slots)
        else:
            raise ValueError(f"unknown private expert engine {expert_engine!r}")

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
        probabilities = torch.softmax(logits.float(), dim=-1)
        grad_sum = grad_probability_sum.float()
        grad_logits.add_(
            probabilities
            * (grad_sum.unsqueeze(0) - (probabilities @ grad_sum).unsqueeze(1))
        )
        grad_logits_storage = grad_logits.to(logits.dtype)
        grad_router = h2_flat.to(router_weight.dtype).T @ grad_logits_storage
        # A custom VJP result must honor the advertised input-view layout, not
        # merely the layout selected by the GEMM implementation. Model code
        # commonly supplies ``nn.Linear.weight.T`` here. Returning a contiguous
        # [D, E] result would make AOT's inverse transpose expose a
        # non-contiguous parameter gradient, contrary to the optimizer ABI.
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
        return (
            grad_h2.reshape(original_shape),
            grad_output.clone(),
            grad_router,
            grad_w13,
            grad_w2,
        )


__all__ = ["_backward_with_engine", "_forward_with_engine"]
