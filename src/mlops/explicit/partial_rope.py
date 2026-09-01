"""Explicit leading-channel RoPE forward and VJP entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation


def forward(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    rotary_dim: int,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Rotate the first ``rotary_dim`` channels and copy all remaining channels."""
    x = x.contiguous()
    implementation = resolve_implementation(
        "partial_rope",
        x,
        positions,
        float(base),
        int(rotary_dim),
        cosine,
        sine,
        surface="explicit",
    )
    with torch.no_grad():
        return implementation.forward(
            x, positions, float(base), int(rotary_dim), cosine, sine
        )


def backward(
    grad_output: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    rotary_dim: int,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Return the VJP, inverse-rotating only the leading channels."""
    grad_output = grad_output.contiguous()
    implementation = resolve_implementation(
        "partial_rope",
        grad_output,
        positions,
        float(base),
        int(rotary_dim),
        cosine,
        sine,
        surface="explicit",
    )
    with torch.no_grad():
        return implementation.backward(
            grad_output,
            positions,
            float(base),
            int(rotary_dim),
            cosine,
            sine,
        )


__all__ = ["backward", "forward"]
