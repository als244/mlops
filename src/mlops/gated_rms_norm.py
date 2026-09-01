"""Model-facing gated RMSNorm semantic façade."""

from __future__ import annotations

import torch

from .dispatch import resolve_implementation


def gated_rms_norm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Return gated RMS normalization for matching input and gate rows."""
    implementation = resolve_implementation(
        "gated_rms_norm", x, gate, weight, float(eps), surface="semantic"
    )
    return implementation.apply(x, gate, weight, float(eps))


__all__ = ["gated_rms_norm"]
