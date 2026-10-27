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
    return func(
        *(
            value._data if isinstance(value, BF16ComputeWeight) else value
            for value in args
        ),
        **kwargs,
    )


@torch.library.custom_op(
    "mlops_router::gradients",
    mutates_args=(),
    schema="(Tensor grad_logits, Tensor hidden, Tensor weight, ScalarType weight_grad_dtype, "
    "bool needs_hidden, bool needs_weight, bool needs_bias) -> (Tensor?, Tensor?, Tensor?)",
)
def gradients(
    grad_logits: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    weight_grad_dtype: torch.dtype,
    needs_hidden: bool,
    needs_weight: bool,
    needs_bias: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    # Keep dtype-output GEMMs behind this boundary as well: current Inductor
    # lowers aten.addmm.dtype through a handler with the ordinary addmm schema.
    grad_logits = grad_logits.contiguous()
    return (
        router_input_gradient(grad_logits, weight, hidden) if needs_hidden else None,
        router_weight_gradient(grad_logits, hidden, weight_grad_dtype)
        if needs_weight
        else None,
        grad_logits.sum(0) if needs_bias else None,
    )


@gradients.register_fake
def _fake_gradients(
    grad_logits, hidden, weight, dtype, needs_hidden, needs_weight, needs_bias
):
    return (
        torch.empty_like(hidden) if needs_hidden else None,
        weight.new_empty(weight.shape, dtype=dtype) if needs_weight else None,
        grad_logits.new_empty((grad_logits.shape[1],)) if needs_bias else None,
    )


def _setup(ctx, inputs, output):
    hidden, weight, bias, dtype = inputs
    ctx.save_for_backward(hidden, weight)
    ctx.has_bias = bias is not None
    ctx.weight_grad_dtype = weight.dtype if dtype is None else dtype


def _backward(ctx, grad_logits):
    hidden, weight = ctx.saved_tensors
    return (
        *gradients(
            grad_logits,
            hidden,
            weight,
            ctx.weight_grad_dtype,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.has_bias and ctx.needs_input_grad[2],
        ),
        None,
    )


logits.register_autograd(_backward, setup_context=_setup)
logits.register_torch_dispatch(BF16ComputeWeight, _dispatch)
gradients.register_torch_dispatch(BF16ComputeWeight, _dispatch)
