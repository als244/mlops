"""Model-facing gated RMSNorm semantic façade."""

from __future__ import annotations

import torch

from .dispatch import resolve_implementation


def _validate(x, gate, weight, activation):
    if activation not in ("silu", "sigmoid"):
        raise ValueError("gated RMSNorm activation must be 'silu' or 'sigmoid'")
    if x.shape != gate.shape or weight.shape != (x.shape[-1],):
        raise ValueError(
            "Gate must match input and weight must match its final dimension"
        )


def gated_rms_norm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    """Return gated RMS normalization for matching input and gate rows."""
    _validate(x, gate, weight, activation)
    implementation = resolve_implementation(
        "gated_rms_norm",
        x,
        gate,
        weight,
        float(eps),
        surface="semantic",
        activation=activation,
    )
    return implementation.apply(x, gate, weight, float(eps), activation=activation)


__all__ = ["gated_rms_norm"]
