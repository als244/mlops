"""Explicit cross-entropy forward and VJP entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation


def forward(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row ``(losses, logsumexp)`` for an explicit VJP."""
    implementation = resolve_implementation(
        "cross_entropy",
        logits,
        targets,
        weight,
        int(ignore_index),
        "none",
        surface="explicit",
    )
    with torch.no_grad():
        return implementation.forward(logits, targets, weight, int(ignore_index))


def backward(
    grad_losses: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    logsumexp: torch.Tensor,
    weight: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return the logits VJP for an arbitrary per-row loss cotangent."""
    implementation = resolve_implementation(
        "cross_entropy",
        logits,
        targets,
        weight,
        int(ignore_index),
        "none",
        surface="explicit",
    )
    with torch.no_grad():
        return implementation.backward(
            grad_losses,
            logits,
            targets,
            logsumexp,
            weight,
            int(ignore_index),
        )


__all__ = ["backward", "forward"]
