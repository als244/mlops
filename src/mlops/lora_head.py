"""Memory-bounded cross entropy with a low-rank output-head update."""

from __future__ import annotations

from typing import Literal

import torch

from .dispatch import resolve_implementation


def lora_head_loss(
    hidden: torch.Tensor,
    head_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    targets: torch.Tensor,
    *,
    scale: float = 1.0,
    chunk_size: int | None = None,
    valid_rows: int | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    """Cross entropy of `X W.T + scale * (X A.T) B.T`, with bounded logits.

    Factors have shapes `[rank, width]` and `[vocabulary, rank]`. They
    compute at the hidden dtype; autograd propagates through factor casts.
    The base weight must share the hidden dtype. Set its `requires_grad`
    to False to omit its dense gradient. Reduction and ignored-target
    semantics match `head_loss`. This operation does not apply dropout.
    """
    implementation = resolve_implementation(
        "lora_head_loss",
        hidden,
        head_weight,
        lora_a,
        lora_b,
        targets,
        surface="semantic",
        scale=scale,
        chunk_size=chunk_size,
        valid_rows=valid_rows,
        reduction=reduction,
    )
    return implementation.apply(
        hidden,
        head_weight,
        lora_a,
        lora_b,
        targets,
        scale=scale,
        chunk_size=chunk_size,
        valid_rows=valid_rows,
        reduction=reduction,
    )


__all__ = ["lora_head_loss"]
