"""Ordinary BF16 shared-expert math, run eagerly beside EP scheduling.

The enclosing custom operator exposes all weights and saved tensors. These
functions own no persistent activations, process groups, or communication state.
The projections are ordinary GEMMs; the SwiGLU between them and its gradient
are single-pass row kernels in FP32, so the operator compiles nothing when it
runs and keeps no FP32 copy of its activations.
"""

from functools import partial

import torch.nn.functional as F

from mlops.kernels.quack_moe.swiglu_rows import (
    swiglu_pair_rows,
    swiglu_pair_rows_backward,
)


def _rows(value):
    return value.reshape(-1, value.shape[-1])


def shared_forward(x, gate_weight, up_weight, down_weight, swiglu_limit=None):
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    hidden = swiglu_pair_rows(_rows(gate), _rows(up), limit=swiglu_limit)
    hidden = hidden.view(gate.shape)
    output = F.linear(hidden, down_weight)
    return output, gate, up, hidden


def _projection_gradients(dy, gate, up, down_weight, swiglu_limit):
    dh = dy @ down_weight
    dgate, dup = swiglu_pair_rows_backward(
        _rows(dh), _rows(gate), _rows(up), limit=swiglu_limit
    )
    return dgate.view(gate.shape), dup.view(up.shape)


def shared_backward(dy, x, gate, up, hidden, gate_weight, up_weight, down_weight):
    dgate, dup = _projection_gradients(dy, gate, up, down_weight, None)
    dx = dgate @ gate_weight + dup @ up_weight
    return dx, dgate.T @ x, dup.T @ x, dy.T @ hidden


def frozen_shared_backward(
    dy, x, gate, up, hidden, gate_weight, up_weight, down_weight, swiglu_limit=None
):
    dgate, dup = _projection_gradients(dy, gate, up, down_weight, swiglu_limit)
    return (dgate @ gate_weight + dup @ up_weight,)


def shared_math(swiglu_limit=None):
    """``(forward, backward)`` of a trainable shared expert."""
    return partial(shared_forward, swiglu_limit=swiglu_limit), shared_backward


def frozen_shared_math(swiglu_limit=None):
    """``(forward, backward)`` of a frozen shared expert: only its input's
    gradient, through the clipped SwiGLU."""
    return (
        partial(shared_forward, swiglu_limit=swiglu_limit),
        partial(frozen_shared_backward, swiglu_limit=swiglu_limit),
    )
