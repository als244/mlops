"""Explicit selected sparse-attention forward and VJP entry points."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..dispatch import resolve_implementation


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    lengths: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(output, lse)`` with selected-key indices held fixed."""
    lengths = tuple(int(length) for length in lengths)
    implementation = resolve_implementation(
        "dsa_attention", q, k, v, indices, lengths, surface="explicit"
    )
    with torch.no_grad():
        return implementation.forward(q, k, v, indices, lengths)


def backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    lengths: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(grad_q, grad_k, grad_v)``; indices have no VJP."""
    lengths = tuple(int(length) for length in lengths)
    implementation = resolve_implementation(
        "dsa_attention", q, k, v, indices, lengths, surface="explicit"
    )
    with torch.no_grad():
        return implementation.backward(
            grad_output, q, k, v, indices, lse, lengths
        )


__all__ = ["backward", "forward"]
