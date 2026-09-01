"""Packed-sequence graph selection for hybrid linear mixers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from .dispatch import resolve_implementation


def linear_mixer(
    forward_segment: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor],
    hidden: torch.Tensor,
    lengths: Sequence[int],
) -> torch.Tensor:
    """Run one packed projection graph or the raw per-sequence graph."""
    lengths = tuple(int(length) for length in lengths)
    implementation = resolve_implementation(
        "linear_mixer", forward_segment, hidden, lengths, surface="semantic"
    )
    return implementation.apply(forward_segment, hidden, lengths)


__all__ = ["linear_mixer"]
