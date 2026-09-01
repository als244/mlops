"""Model-facing tanh-approximate GELU semantic façade."""

from __future__ import annotations

import torch

from .dispatch import resolve_implementation


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Apply the tanh-approximate GELU elementwise without changing shape."""
    implementation = resolve_implementation("gelu", x, surface="semantic")
    return implementation.apply(x)


__all__ = ["gelu"]
