"""Memory-bounded language-model projection and cross-entropy operation."""

from __future__ import annotations

from typing import Literal

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
    reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    """Return the next-token cross entropy from normalized hidden states.

    ``"mean"`` divides the summed cross entropy by the rows, or by
    ``valid_rows`` of them when given; ``"sum"`` returns the sum itself, for
    a caller that divides by a total of its own -- a step's trained tokens
    across its microbatches, say. A row whose target is negative counts in
    neither: it adds nothing to the sum and, under ``"mean"`` without
    ``valid_rows``, is still one of the rows divided by.
    """
    implementation = resolve_implementation(
        "head_loss",
        hidden,
        head_weight,
        targets,
        surface="semantic",
        chunk_size=chunk_size,
        valid_rows=valid_rows,
        reduction=reduction,
    )
    return implementation.apply(
        hidden,
        head_weight,
        targets,
        chunk_size=chunk_size,
        valid_rows=valid_rows,
        reduction=reduction,
    )


__all__ = ["HEAD_CHUNK_SCRATCH_BYTES", "default_head_chunk_size", "head_loss"]
