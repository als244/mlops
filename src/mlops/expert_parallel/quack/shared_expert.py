"""Ordinary BF16 shared-expert math, compiled independently of EP scheduling.

The enclosing custom operator exposes all weights and saved tensors. These
functions own no persistent activations, process groups, or communication state.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F


def shared_forward(x, gate_weight, up_weight, down_weight):
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    hidden = (F.silu(gate.float()) * up.float()).to(x.dtype)
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


@lru_cache(maxsize=1)
def compiled_shared_math():
    # This compilation contains only ordinary local tensor math. EP operations
    # remain in the enclosing operator and the explicit stream schedule.
    options = {"triton.cudagraphs": False, "emulate_precision_casts": True}
    return (
        torch.compile(shared_forward, fullgraph=True, options=options),
        torch.compile(shared_backward, fullgraph=True, options=options),
    )
