"""Explicit two-input SwiGLU forward and VJP entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation
from ..swiglu import _validate_limit


def forward(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    limit: float | None = None,
) -> torch.Tensor:
    """Return ``silu(gate) * up`` using the selected implementation."""
    _validate_limit(limit)
    implementation = resolve_implementation(
        "swiglu", gate, up, surface="explicit", limit=limit
    )
    with torch.no_grad():
        return implementation.forward(gate, up, limit=limit)


def backward(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    limit: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(grad_gate, grad_up)`` for ``grad_output``."""
    _validate_limit(limit)
    implementation = resolve_implementation(
        "swiglu", gate, up, surface="explicit", limit=limit
    )
    with torch.no_grad():
        return implementation.backward(grad_output, gate, up, limit=limit)


__all__ = ["backward", "forward"]
