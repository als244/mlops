"""Exportable expert autograd with logical parameters and explicit physical ops.

PyTorch's registered autograd API requires a functional operator. The physical
forward returns a fresh accumulator; its internal chunk kernels may mutate
that owned result. No accumulator value is saved for backward.
"""

import torch

from ..precision.parameters import ComputeWeight, payload
from ..precision.quantized_parameter import QuantizedWeight
from ..profiling import annotate
from .operators import _backward_fake, expert_backward, expert_forward


def _components(weights, train_base):
    if not train_base:
        return weights
    return [
        part
        for value in weights
        for part in (
            value.components()
            if isinstance(value, QuantizedWeight)
            else [payload(value)]
        )
    ]


@torch.library.custom_op("mlops_seqmoe::expert_value", mutates_args=())
def expert_value(
    x: torch.Tensor,
    acc: torch.Tensor,
    probs: torch.Tensor,
    order: torch.Tensor,
    bounds: torch.Tensor,
    weights: list[torch.Tensor],
    factors: list[torch.Tensor],
    hidden: int,
    capacity: int,
    multiple: int,
    scale: float,
    clip: float,
    precision: str,
    specs: str,
) -> torch.Tensor:
    with annotate("fwd/expert"):
        with annotate("fwd/expert/accumulator_clone"):
            output = acc.clone()
        expert_forward(
            x,
            output,
            probs,
            order,
            bounds,
            weights,
            factors,
            hidden,
            capacity,
            multiple,
            scale,
            clip,
            precision,
            specs,
        )
        return output


@expert_value.register_fake
def _value_fake(x, acc, probs, order, bounds, weights, factors, *options):
    return torch.empty_like(acc)


def _forward(x, acc, probs, order, bounds, weights, factors, *options):
    *kernel_options, _grad_dtype, train_base = options
    return expert_value(
        x,
        acc,
        probs,
        order,
        bounds,
        _components(weights, train_base),
        [payload(value) for value in factors],
        *kernel_options,
    )


@torch.library.custom_op("mlops_seqmoe::expert", mutates_args=())
def expert(
    x: torch.Tensor,
    acc: torch.Tensor,
    probs: torch.Tensor,
    order: torch.Tensor,
    bounds: torch.Tensor,
    weights: list[torch.Tensor],
    factors: list[torch.Tensor],
    hidden: int,
    capacity: int,
    multiple: int,
    scale: float,
    clip: float,
    precision: str,
    specs: str,
    grad_dtype: str,
    train_base: bool,
) -> torch.Tensor:
    return _forward(
        x,
        acc,
        probs,
        order,
        bounds,
        weights,
        factors,
        hidden,
        capacity,
        multiple,
        scale,
        clip,
        precision,
        specs,
        grad_dtype,
        train_base,
    )


expert.register_fake(_value_fake)


def _backward(dy, x, probs, order, bounds, weights, factors, *options):
    train_base = options[-1]
    return expert_backward(
        dy,
        x,
        probs,
        order,
        bounds,
        _components(weights, train_base),
        [payload(value) for value in factors],
        *options,
    )


@torch.library.custom_op("mlops_seqmoe::expert_vjp", mutates_args=())
def expert_vjp(
    dy: torch.Tensor,
    x: torch.Tensor,
    probs: torch.Tensor,
    order: torch.Tensor,
    bounds: torch.Tensor,
    weights: list[torch.Tensor],
    factors: list[torch.Tensor],
    hidden: int,
    capacity: int,
    multiple: int,
    scale: float,
    clip: float,
    precision: str,
    specs: str,
    grad_dtype: str,
    train_base: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    return _backward(
        dy,
        x,
        probs,
        order,
        bounds,
        weights,
        factors,
        hidden,
        capacity,
        multiple,
        scale,
        clip,
        precision,
        specs,
        grad_dtype,
        train_base,
    )


expert_vjp.register_fake(_backward_fake)


def _setup(ctx, inputs, output):
    x, _, probs, order, bounds, weights, factors, *options = inputs
    ctx.save_for_backward(x, probs, order, bounds, *weights, *factors)
    ctx.nweights, ctx.nfactors = len(weights), len(factors)
    ctx.options = options
    ctx.set_materialize_grads(False)


def _autograd(ctx, dy):
    if dy is None:
        return (
            None,
            None,
            None,
            None,
            None,
            [None] * ctx.nweights,
            [None] * ctx.nfactors,
            *([None] * len(ctx.options)),
        )
    x, probs, order, bounds, *values = ctx.saved_tensors
    weights, factors = values[: ctx.nweights], values[ctx.nweights :]
    dx, dp, gradients = expert_vjp(
        dy.contiguous(),
        x,
        probs,
        order,
        bounds,
        weights,
        factors,
        *ctx.options,
    )
    if ctx.options[-1]:
        base_grads, factor_grads = gradients[:2], []
    else:
        base_grads, factor_grads = [None] * ctx.nweights, gradients
    return (
        dx,
        dy,
        dp,
        None,
        None,
        base_grads,
        factor_grads,
        *([None] * len(ctx.options)),
    )


def _dispatch(cls, func, types, args, kwargs):
    if kwargs:
        raise TypeError("Sequential expert operators require positional arguments")
    return (_forward if func is expert._opoverload else _backward)(*args)


expert.register_autograd(_autograd, setup_context=_setup)
for operation in (expert, expert_vjp):
    for wrapper in (ComputeWeight, QuantizedWeight):
        operation.register_torch_dispatch(wrapper, _dispatch)
