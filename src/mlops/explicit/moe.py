"""Explicit dropless routed-MoE forward and VJP entry points."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..dispatch import resolve_implementation


def forward(
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
    swiglu_implementation: str = "builtin.packed_swiglu.triton",
) -> tuple[torch.Tensor, ...]:
    """Return output, auxiliary statistics, and explicit VJP residuals."""
    implementation = resolve_implementation(
        "moe",
        h2,
        residual,
        router_weight,
        w13_experts,
        w2_experts,
        surface="explicit",
        top_k=int(top_k),
        routing_mode=str(routing_mode),
        router_bias=router_bias,
        n_group=int(n_group),
        topk_group=int(topk_group),
        routed_scaling=float(routed_scaling),
        lengths=lengths,
    )
    with torch.no_grad():
        return implementation.forward(
            h2,
            residual,
            router_weight,
            w13_experts,
            w2_experts,
            top_k=int(top_k),
            routing_mode=str(routing_mode),
            router_bias=router_bias,
            n_group=int(n_group),
            topk_group=int(topk_group),
            routed_scaling=float(routed_scaling),
            lengths=lengths,
            weight_precision=str(weight_precision),
            swiglu_implementation=str(swiglu_implementation),
        )


def backward(
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
    swiglu_implementation: str = "builtin.packed_swiglu.triton",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return VJPs for hidden, residual, router, first expert, and second expert."""
    implementation = resolve_implementation(
        "moe",
        h2,
        h2,
        router_weight,
        w13_experts,
        w2_experts,
        surface="explicit",
        top_k=int(top_k),
        routing_mode=str(routing_mode),
        lengths=lengths,
    )
    with torch.no_grad():
        return implementation.backward(
            grad_output,
            grad_aux,
            grad_probability_sum,
            h2,
            router_weight,
            w13_experts,
            w2_experts,
            logits,
            route_weights,
            route_ids,
            order,
            offsets,
            slots,
            h13,
            top_k=int(top_k),
            routing_mode=str(routing_mode),
            lengths=lengths,
            swiglu_implementation=str(swiglu_implementation),
        )


__all__ = ["backward", "forward"]
