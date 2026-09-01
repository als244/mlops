"""Explicit packed SwiGLU forward and VJP entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation


def forward(
    packed: torch.Tensor,
) -> torch.Tensor:
    """Split ``packed[R, 2F]`` and return packed SwiGLU output ``[R, F]``."""
    implementation = resolve_implementation(
        "packed_swiglu", packed, surface="explicit"
    )
    with torch.no_grad():
        return implementation.forward(packed)


def backward(
    grad_output: torch.Tensor,
    packed: torch.Tensor,
) -> torch.Tensor:
    """Return the packed input VJP with shape ``[R, 2F]``."""
    implementation = resolve_implementation(
        "packed_swiglu", packed, surface="explicit"
    )
    with torch.no_grad():
        return implementation.backward(grad_output, packed)


__all__ = ["backward", "forward"]
