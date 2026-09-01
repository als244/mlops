"""Explicit two-input SwiGLU forward and VJP entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation


def forward(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Return ``silu(gate) * up`` using the selected implementation."""
    implementation = resolve_implementation(
        "swiglu", gate, up, surface="explicit"
    )
    with torch.no_grad():
        return implementation.forward(gate, up)


def backward(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(grad_gate, grad_up)`` for ``grad_output``."""
    implementation = resolve_implementation(
        "swiglu", gate, up, surface="explicit"
    )
    with torch.no_grad():
        return implementation.backward(grad_output, gate, up)


__all__ = ["backward", "forward"]
