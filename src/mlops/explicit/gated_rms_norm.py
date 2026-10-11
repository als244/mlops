"""Explicit gated RMSNorm forward and VJP entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation
from ..gated_rms_norm import _validate


def forward(x, gate, weight, eps: float = 1e-5, *, activation: str = "silu"):
    """Return ``(output, rstd)`` for the selected gate activation times RMSNorm."""
    _validate(x, gate, weight, activation)
    implementation = resolve_implementation(
        "gated_rms_norm",
        x,
        gate,
        weight,
        float(eps),
        surface="explicit",
        activation=activation,
    )
    with torch.no_grad():
        return implementation.forward(
            x, gate, weight, float(eps), activation=activation
        )


def backward(
    grad_output, x, gate, weight, rstd, eps: float = 1e-5, *, activation: str = "silu"
):
    """Return ``(grad_x, grad_gate, grad_weight)``."""
    _validate(x, gate, weight, activation)
    implementation = resolve_implementation(
        "gated_rms_norm",
        x,
        gate,
        weight,
        float(eps),
        surface="explicit",
        activation=activation,
    )
    with torch.no_grad():
        return implementation.backward(
            grad_output, x, gate, weight, rstd, float(eps), activation=activation
        )


__all__ = ["backward", "forward"]
