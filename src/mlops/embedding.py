"""Model-facing embedding semantic façade."""

from __future__ import annotations

import torch

from .dispatch import resolve_implementation


def embedding(tokens: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Look up ``tokens[*]`` in ``weight[V, D]`` and return ``[* , D]``."""
    implementation = resolve_implementation(
        "embedding", tokens, weight, surface="semantic"
    )
    return implementation.apply(tokens, weight)


__all__ = ["embedding"]
