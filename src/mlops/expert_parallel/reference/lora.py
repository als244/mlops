"""Independent per-expert LoRA reference; imports only PyTorch.

Routing is supplied by the caller. Gate/up can use adjacent pairs (Quack) or
contiguous halves (TE), without changing the per-expert parameterization.
"""

import torch
from torch.nn import functional as F


def expert_computation_lora(
    x,
    expert_ids,
    routing_weights,
    gate_up_weight,
    down_weight,
    gate_up_a,
    gate_up_b,
    down_a,
    down_b,
    *,
    scale=1.0,
    interleaved=False,
    shared_weights=(),
    compute_dtype=torch.bfloat16,
):
    accumulation_dtype = (
        torch.float32
        if compute_dtype in (torch.bfloat16, torch.float16)
        else compute_dtype
    )
    output = torch.zeros_like(x, dtype=accumulation_dtype)
    for expert in range(gate_up_weight.shape[0]):
        token, slot = (expert_ids == expert).nonzero(as_tuple=True)
        value = x[token].to(compute_dtype)
        z = F.linear(value, gate_up_a[expert].to(compute_dtype))
        pre = (
            F.linear(value, gate_up_weight[expert].to(compute_dtype)).to(
                accumulation_dtype
            )
            + scale
            * F.linear(z, gate_up_b[expert].to(compute_dtype)).to(accumulation_dtype)
        ).to(compute_dtype)
        gate, up = (pre[:, 0::2], pre[:, 1::2]) if interleaved else pre.chunk(2, -1)
        h = (F.silu(gate.to(accumulation_dtype)) * up.to(accumulation_dtype)).to(
            compute_dtype
        )
        z = F.linear(h, down_a[expert].to(compute_dtype))
        raw = (
            F.linear(h, down_weight[expert].to(compute_dtype)).to(accumulation_dtype)
            + scale
            * F.linear(z, down_b[expert].to(compute_dtype)).to(accumulation_dtype)
        ).to(compute_dtype)
        contribution = (
            raw.to(accumulation_dtype) * routing_weights[token, slot, None]
        ).to(compute_dtype)
        output = output.index_add(0, token, contribution.to(accumulation_dtype))
    output = output.to(compute_dtype)
    if shared_weights:
        gate = F.linear(x.to(compute_dtype), shared_weights[0].to(compute_dtype))
        up = F.linear(x.to(compute_dtype), shared_weights[1].to(compute_dtype))
        h = (F.silu(gate.to(accumulation_dtype)) * up.to(accumulation_dtype)).to(
            compute_dtype
        )
        output = output + F.linear(h, shared_weights[2].to(compute_dtype))
    return output
