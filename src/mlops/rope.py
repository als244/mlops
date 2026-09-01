"""Model-facing Llama RoPE semantic façade."""

from __future__ import annotations

import torch

from .dispatch import resolve_implementation


def rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Rotate ``(..., heads, head_dim)`` using caller-owned tables.

    ``positions``, ``cosine``, and ``sine`` are ordinary operation inputs and
    may be retained by normal autograd for the inverse rotation. The operation
    does not build, cache, or own the tables.
    """
    # MLA rotates a projection slice. Materialize it before support filtering
    # so the selected implementation sees the canonical contiguous geometry.
    x = x.contiguous()
    implementation = resolve_implementation(
        "rope",
        x,
        positions,
        float(base),
        cosine,
        sine,
        surface="semantic",
    )
    return implementation.apply(x, positions, float(base), cosine, sine)


__all__ = ["rope"]
