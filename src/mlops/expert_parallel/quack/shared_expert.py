"""Ordinary BF16 shared-expert math, compiled independently of EP scheduling.

The enclosing custom operator exposes all weights and saved tensors. These
functions own no persistent activations, process groups, or communication state.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F


def shared_forward(x, gate_weight, up_weight, down_weight, swiglu_limit=None):
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    g, u = gate.float(), up.float()
    if swiglu_limit is not None:
        g, u = g.clamp(max=swiglu_limit), u.clamp(-swiglu_limit, swiglu_limit)
    hidden = (F.silu(g) * u).to(x.dtype)
    output = F.linear(hidden, down_weight)
    return output, gate, up, hidden


def shared_backward(dy, x, gate, up, hidden, gate_weight, up_weight, down_weight):
    dh = (dy @ down_weight).float()
    g = gate.float()
    sigmoid = torch.sigmoid(g)
    dgate = (dh * up.float() * sigmoid * (1 + g * (1 - sigmoid))).to(x.dtype)
    dup = (dh * F.silu(g)).to(x.dtype)
    dx = dgate @ gate_weight + dup @ up_weight
    return dx, dgate.T @ x, dup.T @ x, dy.T @ hidden


def frozen_shared_backward(
    dy, x, gate, up, hidden, gate_weight, up_weight, down_weight, swiglu_limit=None
):
    dh = (dy @ down_weight).float()
    g, u = gate.float(), up.float()
    if swiglu_limit is not None:
        g, u = g.clamp(max=swiglu_limit), u.clamp(-swiglu_limit, swiglu_limit)
    sigmoid = g.sigmoid()
    dg = dh * u * sigmoid * (1 + g * (1 - sigmoid))
    du = dh * g * sigmoid
    if swiglu_limit is not None:
        dg = torch.where(gate <= swiglu_limit, dg, 0.0)
        du = torch.where(up.abs() <= swiglu_limit, du, 0.0)
    return (dg.to(x.dtype) @ gate_weight + du.to(x.dtype) @ up_weight,)


@lru_cache(maxsize=1)
def compiled_shared_math():
    # This compilation contains only ordinary local tensor math. EP operations
    # remain in the enclosing operator and the explicit stream schedule.
    options = {"triton.cudagraphs": False, "emulate_precision_casts": True}
    return (
        torch.compile(shared_forward, fullgraph=True, options=options),
        torch.compile(shared_backward, fullgraph=True, options=options),
    )
