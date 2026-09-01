"""Explicit tanh-approximate GELU forward and VJP entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation


def forward(x: torch.Tensor) -> torch.Tensor:
    """Return the elementwise tanh-approximate GELU of ``x``."""
    implementation = resolve_implementation("gelu", x, surface="explicit")
    with torch.no_grad():
        return implementation.forward(x)


def backward(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return the input VJP for ``grad_output``."""
    implementation = resolve_implementation("gelu", x, surface="explicit")
    with torch.no_grad():
        return implementation.backward(grad_output, x)


__all__ = ["backward", "forward"]
