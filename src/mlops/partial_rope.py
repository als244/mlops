"""Model-facing partial-channel RoPE semantic façade."""

from __future__ import annotations

import torch

from .dispatch import resolve_implementation


def partial_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    rotary_dim: int,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Rotate the first ``rotary_dim`` channels of ``x[..., H, Dh]``."""
    x = x.contiguous()
    implementation = resolve_implementation(
        "partial_rope",
        x,
        positions,
        float(base),
        int(rotary_dim),
        cosine,
        sine,
        surface="semantic",
    )
    return implementation.apply(
        x, positions, float(base), int(rotary_dim), cosine, sine
    )


__all__ = ["partial_rope"]
