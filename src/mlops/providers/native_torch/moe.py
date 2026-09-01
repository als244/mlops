"""Ordinary native-PyTorch per-expert MoE graph."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.moe_reference import native_torch_moe
from ..builtin.moe import supports as common_support


def supports(*args, surface, **kwargs):
    if surface == "explicit":
        return SupportResult.no("native per-expert MoE is apply-only")
    return common_support(*args, surface=surface, **kwargs)


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
    return native_torch_moe(
        h2,
        residual,
        router_weight,
        w13_experts,
        w2_experts,
        bias,
        shared_w13,
        shared_w2,
        shared_gate_weight,
        top_k=int(top_k),
        routing_mode=routing_mode,
        n_group=int(n_group),
        topk_group=int(topk_group),
        routed_scaling=float(routed_scaling),
        lengths=normalized_lengths,
        shared_mode=shared_mode,
    )


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="moe",
        implementation_id="native_torch.moe",
        provider="native_torch",
        priority=0,
        deterministic=True,
        supports=supports,
        apply=apply,
    )
)

__all__ = ["IMPLEMENTATION", "apply", "supports"]
