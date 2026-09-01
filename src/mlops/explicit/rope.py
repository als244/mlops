"""Explicit full-channel RoPE forward and VJP entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation


def forward(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Rotate all head channels; no operation-produced tensor is saved."""
    x = x.contiguous()
    implementation = resolve_implementation(
        "rope", x, positions, float(base), cosine, sine, surface="explicit"
    )
    with torch.no_grad():
        return implementation.forward(x, positions, float(base), cosine, sine)


def backward(
    grad_output: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Apply the inverse rotation to return the input VJP."""
    grad_output = grad_output.contiguous()
    implementation = resolve_implementation(
        "rope",
        grad_output,
        positions,
        float(base),
        cosine,
        sine,
        surface="explicit",
    )
    with torch.no_grad():
        return implementation.backward(
            grad_output, positions, float(base), cosine, sine
        )


__all__ = ["backward", "forward"]
