"""Explicit functional expert operator schemas and fake shape contracts."""

import torch

from ..profiling import annotate
from .execution import backward_expert_into, forward_expert


@torch.library.custom_op("mlops_seqmoe::expert_forward", mutates_args=("acc",))
def expert_forward(
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
) -> None:
    forward_expert(
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
    )


@expert_forward.register_fake
def _forward_fake(*args, **kwargs):
    return None


@torch.library.custom_op("mlops_seqmoe::expert_backward", mutates_args=())
def expert_backward(
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

    with annotate("bwd/expert/input_probability_gradient_zero"):
        dx, dp = torch.zeros_like(x), torch.zeros_like(probs)
    gradients = backward_expert_into(
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
        dx,
        dp,
        False,
    )
    return dx, dp, gradients


@expert_backward.register_fake
def _backward_fake(
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
):
    dtype = getattr(torch, grad_dtype)
    base_grads = (
        [
            x.new_empty((2 * hidden, x.shape[1]), dtype=dtype),
            x.new_empty((x.shape[1], hidden), dtype=dtype),
        ]
        if train_base
        else []
    )
    return (
        torch.empty_like(x),
        torch.empty_like(probs),
        base_grads + [torch.empty_like(v, dtype=dtype) for v in factors],
    )
