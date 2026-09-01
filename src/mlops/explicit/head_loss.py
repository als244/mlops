"""Explicit bounded-logits head-loss forward and VJP entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation


def forward(
    hidden: torch.Tensor,
    head_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int | None = None,
    valid_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return loss plus seed-one VJPs for hidden and head weight."""
    implementation = resolve_implementation(
        "head_loss",
        hidden,
        head_weight,
        targets,
        surface="explicit",
        chunk_size=chunk_size,
        valid_rows=valid_rows,
    )
    with torch.no_grad():
        return implementation.forward(
            hidden,
            head_weight,
            targets,
            chunk_size=chunk_size,
            valid_rows=valid_rows,
        )


def backward(
    grad_loss: torch.Tensor,
    grad_hidden_seed: torch.Tensor,
    grad_head_weight_seed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale and return the two seed-one VJPs by ``grad_loss``."""
    implementation = resolve_implementation(
        "head_loss",
        grad_hidden_seed,
        grad_head_weight_seed,
        surface="explicit",
        _entrypoint="backward",
    )
    with torch.no_grad():
        return implementation.backward(
            grad_loss,
            grad_hidden_seed,
            grad_head_weight_seed,
        )


__all__ = ["backward", "forward"]
