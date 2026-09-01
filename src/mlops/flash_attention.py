"""Model-facing variable-length attention semantic façade."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .dispatch import resolve_implementation


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: Sequence[int],
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Apply an exact variable-length attention implementation."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    normalized_lengths = tuple(int(length) for length in lengths)
    implementation = resolve_implementation(
        "flash_attention",
        q,
        k,
        v,
        normalized_lengths,
        surface="semantic",
        causal=bool(causal),
        softmax_scale=softmax_scale,
    )
    return implementation.apply(
        q,
        k,
        v,
        normalized_lengths,
        causal=bool(causal),
        softmax_scale=softmax_scale,
    )


__all__ = ["flash_attention"]
