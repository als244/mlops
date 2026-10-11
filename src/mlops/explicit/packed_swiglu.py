"""Explicit packed SwiGLU forward and VJP entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation
from ..swiglu import _validate_limit


def forward(
    packed: torch.Tensor,
    *,
    limit: float | None = None,
) -> torch.Tensor:
    """Split ``packed[..., 2F]`` and return packed SwiGLU output ``[..., F]``."""
    _validate_limit(limit)
    implementation = resolve_implementation(
        "packed_swiglu", packed, surface="explicit", limit=limit
    )
    with torch.no_grad():
        return implementation.forward(packed, limit=limit)


def backward(
    grad_output: torch.Tensor,
    packed: torch.Tensor,
    *,
    limit: float | None = None,
) -> torch.Tensor:
    """Return the packed input VJP with shape ``[..., 2F]``."""
    _validate_limit(limit)
    implementation = resolve_implementation(
        "packed_swiglu", packed, surface="explicit", limit=limit
    )
    with torch.no_grad():
        return implementation.backward(grad_output, packed, limit=limit)


__all__ = ["backward", "forward"]
