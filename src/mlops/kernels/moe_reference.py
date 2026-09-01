"""Native-Torch oracle for the complete routed SwiGLU MoE operation."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from .moe_router import route_torch


def project(x, engine_weight):
    """Apply either an nn.Linear-owned transposed view or engine weight."""
    return (
        functional.linear(x, engine_weight.T)
        if not engine_weight.is_contiguous()
        else x @ engine_weight
    )


def native_torch_moe(
    h2,
    residual,
    router_weight,
    w13_experts,
    w2_experts,
    router_bias,
    shared_w13,
    shared_w2,
    shared_gate_weight,
    *,
    top_k,
    routing_mode,
    n_group,
    topk_group,
    routed_scaling,
    lengths,
    shared_mode,
):
    """Return the native-Torch output and load-balance statistics."""
    original_shape = h2.shape
    h2_flat = h2.reshape(-1, h2.shape[-1])
    residual_flat = residual.reshape_as(h2_flat)
    router_input = h2_flat.to(router_weight.dtype)
    logits = project(router_input, router_weight)
    route_weights, route_ids = route_torch(
        logits,
        int(top_k),
        routing_mode,
        bias=None if router_bias.numel() == 0 else router_bias,
        n_group=int(n_group),
        topk_group=int(topk_group),
        routed_scaling=float(routed_scaling),
    )
    expert_count = router_weight.shape[1]
    counts = torch.bincount(route_ids.reshape(-1).long(), minlength=expert_count)
    if routing_mode == "sigmoid_noaux_tc":
        scores = torch.sigmoid(logits.float())
        probabilities = scores / scores.sum(-1, keepdim=True)
        aux = torch.zeros((), dtype=torch.float32, device=h2.device)
        start = 0
        for length in lengths:
            stop = start + int(length)
            segment_counts = torch.bincount(
                route_ids[start:stop].reshape(-1).long(),
                minlength=expert_count,
            ).float()
            frequency = (
                segment_counts * expert_count / (route_ids.shape[1] * int(length))
            )
            aux = aux + (frequency * probabilities[start:stop].mean(0)).sum()
            start = stop
    else:
        probabilities = torch.softmax(logits.float(), dim=-1)
        frequency = counts.float() / route_ids.numel()
        aux = expert_count * (frequency * probabilities.mean(0)).sum()

    width = w2_experts.shape[1]
    routed = torch.zeros_like(h2_flat, dtype=torch.float32)
    for expert in range(expert_count):
        coefficient = (route_weights * (route_ids == expert)).sum(-1)
        h13 = h2_flat @ w13_experts[expert]
        gate, up = h13[:, :width], h13[:, width:]
        activated = functional.silu(gate.float()).to(gate.dtype) * up
        expert_output = activated @ w2_experts[expert]
        routed = routed + coefficient[:, None] * expert_output.float()
    base = residual_flat
    if shared_w13 is not None:
        shared_h13 = project(h2_flat, shared_w13)
        shared_width = shared_w2.shape[0]
        shared_gate, shared_up = (
            shared_h13[:, :shared_width],
            shared_h13[:, shared_width:],
        )
        shared_activated = (
            functional.silu(shared_gate.float()).to(shared_gate.dtype) * shared_up
        )
        shared = project(shared_activated, shared_w2)
        if shared_mode == "sigmoid":
            gate = torch.sigmoid(project(h2_flat, shared_gate_weight).float())
            shared = (gate * shared.float()).to(residual_flat.dtype)
        base = residual_flat + shared
    output = (base.float() + routed).to(h2.dtype)
    return output.reshape(original_shape), aux, counts, probabilities.sum(0)


__all__ = ["native_torch_moe", "project"]
