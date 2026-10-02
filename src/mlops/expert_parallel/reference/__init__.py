"""Independent MoE reference. This module depends only on PyTorch.

Routing and expert computation are separate: expert_computation never chooses
experts. Pass exactly the same expert_ids and routing_weights to every backend.
All arguments are ordinary tensors, including dequantized expert weights.
"""

import torch
from torch.nn import functional as F


def router_logits(x, weight, *, dtype=torch.float32):
    return F.linear(x.to(dtype), weight.to(dtype)).float()


def routing_probabilities(logits, expert_ids, *, renormalize_topk=True):
    """Differentiable probabilities for fixed IDs; no top-k decision here."""
    ids = expert_ids.long()
    if renormalize_topk:
        return logits.gather(-1, ids).softmax(-1)
    return logits.softmax(-1).gather(-1, ids)


def route(x, weight, *, top_k, dtype=torch.float32, renormalize_topk=True):
    """Reference routing, tested separately from expert computation."""
    logits = router_logits(x, weight, dtype=dtype)
    ids = logits.topk(top_k, dim=-1).indices.to(torch.int32)
    return ids, routing_probabilities(logits, ids, renormalize_topk=renormalize_topk)


def expert_computation(
    x,
    expert_ids,
    routing_weights,
    gate_weight,
    up_weight,
    down_weight,
    *,
    shared_weights=(),
    latent_down=None,
    latent_up=None,
):
    """BF16 expert arithmetic and FP32 accumulation for caller-supplied routes.

    Weights have shapes [E,H,D], [E,H,D], [E,D,H]. Shared weights are
    [shared_H,D], [shared_H,D], [D,shared_H]. FP8 tests pass dequantized
    compute weights, measuring approximation against this BF16 baseline.
    """
    z = x if latent_down is None else F.linear(x, latent_down.bfloat16())
    output = torch.zeros_like(z, dtype=torch.float32)
    for expert in range(gate_weight.shape[0]):
        token, slot = (expert_ids == expert).nonzero(as_tuple=True)
        value = z[token]
        gate = F.linear(value, gate_weight[expert].bfloat16())
        up = F.linear(value, up_weight[expert].bfloat16())
        hidden = (F.silu(gate.float()) * up.float()).bfloat16()
        raw = F.linear(hidden, down_weight[expert].bfloat16())
        contribution = (raw.float() * routing_weights[token, slot, None]).bfloat16()
        output = output.index_add(0, token, contribution.float())
    output = output.bfloat16()
    if latent_up is not None:
        output = F.linear(output, latent_up.bfloat16())
    if shared_weights:
        gate = F.linear(x, shared_weights[0].bfloat16())
        up = F.linear(x, shared_weights[1].bfloat16())
        hidden = (F.silu(gate.float()) * up.float()).bfloat16()
        shared = F.linear(hidden, shared_weights[2].bfloat16())
        output = (output.float() + shared.float()).bfloat16()
    return output
