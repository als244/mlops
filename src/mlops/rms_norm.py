"""Model-facing RMSNorm semantic façade."""

from __future__ import annotations

import torch

from .dispatch import resolve_implementation


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Apply the selected RMSNorm implementation."""
    implementation = resolve_implementation(
        "rms_norm", x, weight, float(eps), surface="semantic"
    )
    return implementation.apply(x, weight, float(eps))


__all__ = ["rms_norm"]
