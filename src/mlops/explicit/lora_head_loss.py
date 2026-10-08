"""Autograd-independent LoRA head loss and unit-cotangent VJPs."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation


def forward(
    hidden: torch.Tensor,
    head_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    targets: torch.Tensor,
    *,
    scale: float = 1.0,
    chunk_size: int | None = None,
    valid_rows: int | None = None,
    reduction: str = "mean",
    weight_grad_dtype: torch.dtype | None = None,
    need_hidden_grad: bool = True,
    need_head_grad: bool = False,
    need_lora_a_grad: bool = True,
    need_lora_b_grad: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Return loss, hidden/base/A/B gradient seeds; unrequested seeds are None.

    Unlike the semantic call, explicit gradient requests are independent of
    `requires_grad`. Factor products compute at the hidden dtype, and their
    seed gradients use `weight_grad_dtype` or that compute dtype.
    """
    options = {
        "scale": scale,
        "chunk_size": chunk_size,
        "valid_rows": valid_rows,
        "reduction": reduction,
        "weight_grad_dtype": weight_grad_dtype,
        "need_hidden_grad": need_hidden_grad,
        "need_head_grad": need_head_grad,
        "need_lora_a_grad": need_lora_a_grad,
        "need_lora_b_grad": need_lora_b_grad,
    }
    implementation = resolve_implementation(
        "lora_head_loss",
        hidden,
        head_weight,
        lora_a,
        lora_b,
        targets,
        surface="explicit",
        **options,
    )
    with torch.no_grad():
        return implementation.forward(
            hidden,
            head_weight,
            lora_a,
            lora_b,
            targets,
            **options,
        )


def backward(
    grad_loss: torch.Tensor,
    grad_hidden_seed: torch.Tensor | None,
    grad_head_seed: torch.Tensor | None,
    grad_lora_a_seed: torch.Tensor | None,
    grad_lora_b_seed: torch.Tensor | None,
) -> tuple[
    torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
]:
    """Scale immutable seeds by the scalar cotangent, preserving None."""
    seeds = grad_hidden_seed, grad_head_seed, grad_lora_a_seed, grad_lora_b_seed
    implementation = resolve_implementation(
        "lora_head_loss",
        *seeds,
        surface="explicit",
        _entrypoint="backward",
    )
    with torch.no_grad():
        return implementation.backward(grad_loss, *seeds)


__all__ = ["backward", "forward"]
