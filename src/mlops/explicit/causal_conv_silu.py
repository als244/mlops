"""Explicit packed causal depthwise-convolution-plus-SiLU entry points."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..dispatch import resolve_implementation


def forward(
    x,
    weight,
    lengths: Sequence[int],
    cumulative_lengths,
    chunk_indices=None,
):
    """Return the output using caller-owned cumulative lengths."""
    normalized = tuple(int(length) for length in lengths)
    implementation = resolve_implementation(
        "causal_conv_silu",
        x,
        weight,
        normalized,
        cumulative_lengths,
        chunk_indices,
        surface="explicit",
    )
    with torch.no_grad():
        return implementation.forward(
            x, weight, normalized, cumulative_lengths, chunk_indices
        )


def backward(
    grad_output, x, weight, cumulative_lengths, chunk_indices=None
):
    """Return ``(grad_x, grad_weight)`` using saved sequence metadata."""
    implementation = resolve_implementation(
        "causal_conv_silu",
        x,
        weight,
        (),
        cumulative_lengths,
        chunk_indices,
        surface="explicit",
    )
    with torch.no_grad():
        return implementation.backward(
            grad_output, x, weight, cumulative_lengths, chunk_indices
        )


__all__ = ["backward", "forward"]
