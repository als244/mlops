"""Model-facing L2-normalization semantic façade."""

from __future__ import annotations

import torch

from .dispatch import resolve_implementation


def l2_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize each final-dimension vector in ``x[..., D]``."""
    implementation = resolve_implementation(
        "l2_norm", x, float(eps), surface="semantic"
    )
    return implementation.apply(x, float(eps))


__all__ = ["l2_norm"]
