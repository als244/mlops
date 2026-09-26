"""Explicit packed causal depthwise-convolution-plus-SiLU entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation


def forward(
    x,
    weight,
    cumulative_lengths,
    chunk_indices=None,
):
    """Return the output using caller-owned cumulative lengths."""
    implementation = resolve_implementation(
        "causal_conv_silu",
        x,
        weight,
        cumulative_lengths,
        chunk_indices,
        surface="explicit",
    )
    with torch.no_grad():
        return implementation.forward(x, weight, cumulative_lengths, chunk_indices)


def backward(
    grad_output, x, weight, cumulative_lengths, chunk_indices=None
):
    """Return ``(grad_x, grad_weight)`` using saved sequence metadata."""
    implementation = resolve_implementation(
        "causal_conv_silu",
        x,
        weight,
        cumulative_lengths,
        chunk_indices,
        surface="explicit",
    )
    with torch.no_grad():
        return implementation.backward(
            grad_output, x, weight, cumulative_lengths, chunk_indices
        )


__all__ = ["backward", "forward"]
