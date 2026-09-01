"""Explicit gated RMSNorm forward and VJP entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation


def forward(x, gate, weight, eps: float = 1e-5):
    """Return ``(output, rstd)`` for ``silu(gate) * RMSNorm(x) * weight``."""
    implementation = resolve_implementation(
        "gated_rms_norm", x, gate, weight, float(eps), surface="explicit"
    )
    with torch.no_grad():
        return implementation.forward(x, gate, weight, float(eps))


def backward(grad_output, x, gate, weight, rstd, eps: float = 1e-5):
    """Return ``(grad_x, grad_gate, grad_weight)``."""
    implementation = resolve_implementation(
        "gated_rms_norm", x, gate, weight, float(eps), surface="explicit"
    )
    with torch.no_grad():
        return implementation.backward(
            grad_output, x, gate, weight, rstd, float(eps)
        )


__all__ = ["backward", "forward"]
