"""Model-facing cross-entropy semantic façade."""

from __future__ import annotations

from typing import Literal

import torch

from .dispatch import resolve_implementation


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    ignore_index: int = -100,
    reduction: Literal["none", "sum", "mean"] = "mean",
) -> torch.Tensor:
    """Return class-index cross entropy over two-dimensional logits."""
    implementation = resolve_implementation(
        "cross_entropy",
        logits,
        targets,
        weight,
        int(ignore_index),
        str(reduction),
        surface="semantic",
    )
    return implementation.apply(
        logits,
        targets,
        weight,
        int(ignore_index),
        str(reduction),
    )


__all__ = ["cross_entropy"]
