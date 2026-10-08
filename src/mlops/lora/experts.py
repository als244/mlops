"""Joint gate/up and down LoRA with independent factors for every expert."""

import torch
from torch import nn
from torch.nn import functional as F

from ..swiglu import swiglu
from .grouped import grouped_linear
from .routing import combine_rows, dispatch_rows, layout, routing


def initialize_factors(module, gate_up, down, config):
    """Register only small factors, preserving the existing base parameters."""
    options = {"device": gate_up.device, "dtype": config.factor_dtype}
    shapes = {
        "gate_up_a": (gate_up.shape[0], gate_up.shape[1], config.rank),
        "gate_up_b": (gate_up.shape[0], config.rank, gate_up.shape[2]),
        "down_a": (down.shape[0], down.shape[1], config.rank),
        "down_b": (down.shape[0], config.rank, down.shape[2]),
    }
    for name, shape in shapes.items():
        module.register_parameter(
            "lora_" + name, nn.Parameter(torch.empty(shape, **options))
        )
    reset_factors(module)


def reset_factors(module):
    for prefix in ("gate_up", "down"):
        a, b = getattr(module, f"lora_{prefix}_a"), getattr(module, f"lora_{prefix}_b")
        nn.init.normal_(a, std=a.shape[1] ** -0.5)
        nn.init.zeros_(b)


def _project(x, weight, a, b, offsets, scale):
    low = grouped_linear(x, a.to(x.dtype), offsets)
    return grouped_linear(x, weight, offsets) + scale * grouped_linear(
        low, b.to(x.dtype), offsets
    )


def expert_lora(
    hidden,
    residual,
    router_weight,
    gate_up,
    down,
    a13,
    b13,
    a2,
    b2,
    *,
    top_k,
    routing_mode,
    scale,
):
    """Fused kernels with compiler-visible low-rank projections and lifetimes.

    Weight shapes are [E,D,2H], [E,H,D]; router_weight is [D,E]. No dense
    B@A update or frozen base weight gradient is constructed.
    """
    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = F.linear(flat.to(router_weight.dtype), router_weight.T)
    weights, ids = routing(logits, top_k, routing_mode)
    order, offsets, slots = layout(ids, gate_up.shape[0])
    counts = (offsets[1:] - offsets[:-1]).long()
    probabilities = logits.float().softmax(-1)
    probability_sum = probabilities.sum(0)
    auxiliary = (
        gate_up.shape[0]
        * ((counts.float() / ids.numel()) * probabilities.mean(0)).sum()
    )
    x = dispatch_rows(flat, order, slots)
    h13 = _project(x, gate_up, a13, b13, offsets, scale)
    gate, up = h13.chunk(2, -1)
    activated = swiglu(gate, up)
    y = _project(activated, down, a2, b2, offsets, scale)
    output = combine_rows(
        y, order, slots, weights, residual.reshape_as(flat).contiguous()
    )
    return output.reshape_as(hidden), auxiliary, counts, probability_sum
