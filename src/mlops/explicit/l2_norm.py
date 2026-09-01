"""Explicit L2-normalization forward and VJP entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation


def forward(x, eps: float = 1e-6):
    """Return ``(normalized, rstd)`` for final-dimension L2 normalization."""
    implementation = resolve_implementation(
        "l2_norm", x, float(eps), surface="explicit"
    )
    with torch.no_grad():
        return implementation.forward(x, float(eps))


def backward(grad_output, output, rstd, eps: float = 1e-6):
    """Return the input VJP using saved normalized output and ``rstd``."""
    implementation = resolve_implementation(
        "l2_norm", output, float(eps), surface="explicit"
    )
    with torch.no_grad():
        return implementation.backward(grad_output, output, rstd, float(eps))


__all__ = ["backward", "forward"]
