"""Explicit RMSNorm forward and VJP entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation


def forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(output, rstd)``; retain ``x``, ``weight``, and ``rstd`` for VJP."""
    implementation = resolve_implementation(
        "rms_norm", x, weight, float(eps), surface="explicit"
    )
    with torch.no_grad():
        return implementation.forward(x, weight, float(eps))


def backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(grad_x, grad_weight)`` for an explicit output cotangent."""
    implementation = resolve_implementation(
        "rms_norm", x, weight, surface="explicit"
    )
    with torch.no_grad():
        return implementation.backward(grad_output, x, weight, rstd)


__all__ = ["backward", "forward"]
