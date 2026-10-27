"""Exportable router autograd boundary, including packed BF16 parameters.

Directly reading a wrapper's payload in an autograd.Function forward makes
Dynamo lift it as a constant. Keep the logical parameter until dispatch/AOT
unwraps it, so export retains its identity and floating-point gradient.
"""

from __future__ import annotations

import torch

from ..expert_parallel.parameters import BF16ComputeWeight
from .router import router_input_gradient, router_logits, router_weight_gradient


@torch.library.custom_op("mlops_router::logits", mutates_args=())
def logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    weight_grad_dtype: torch.dtype | None,
) -> torch.Tensor:
    return router_logits(hidden, weight, bias)


@logits.register_fake
def _fake(hidden, weight, bias, weight_grad_dtype):
    return hidden.new_empty((hidden.shape[0], weight.shape[0]), dtype=torch.float32)


def _dispatch(cls, func, types, args, kwargs):
    hidden, weight, bias, dtype = args
    return logits(hidden, weight._data, bias, dtype)


def _setup(ctx, inputs, output):
    hidden, weight, bias, dtype = inputs
    ctx.save_for_backward(hidden, weight)
    ctx.has_bias = bias is not None
    ctx.weight_grad_dtype = weight.dtype if dtype is None else dtype


def _backward(ctx, grad_logits):
    hidden, weight = ctx.saved_tensors
    grad_logits = grad_logits.contiguous()
    grad_hidden = grad_weight = grad_bias = None
    if ctx.needs_input_grad[0]:
        grad_hidden = router_input_gradient(grad_logits, weight, hidden)
    if ctx.needs_input_grad[1]:
        grad_weight = router_weight_gradient(grad_logits, hidden, ctx.weight_grad_dtype)
    if ctx.has_bias and ctx.needs_input_grad[2]:
        grad_bias = grad_logits.sum(0)
    return grad_hidden, grad_weight, grad_bias, None


logits.register_autograd(_backward, setup_context=_setup)
logits.register_torch_dispatch(BF16ComputeWeight, _dispatch)
