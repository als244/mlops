"""Memory-bounded language-model projection and cross-entropy operation."""

from __future__ import annotations

import torch

from .dispatch import resolve_implementation
from .kernels.head import HEAD_CHUNK_SCRATCH_BYTES, default_head_chunk_size


def head_loss(
    hidden: torch.Tensor,
    head_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int | None = None,
    valid_rows: int | None = None,
) -> torch.Tensor:
    """Return mean cross entropy from normalized hidden states."""
    implementation = resolve_implementation(
        "head_loss",
        hidden,
        head_weight,
        targets,
        surface="semantic",
        chunk_size=chunk_size,
        valid_rows=valid_rows,
    )
    return implementation.apply(
        hidden,
        head_weight,
        targets,
        chunk_size=chunk_size,
        valid_rows=valid_rows,
    )


__all__ = ["HEAD_CHUNK_SCRATCH_BYTES", "default_head_chunk_size", "head_loss"]
