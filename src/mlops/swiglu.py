"""Model-facing SwiGLU semantic façades."""

from __future__ import annotations

import torch

from .dispatch import resolve_implementation


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Return ``silu(gate) * up`` for equal-shaped ``[..., F]`` inputs."""
    implementation = resolve_implementation(
        "swiglu", gate, up, surface="semantic"
    )
    return implementation.apply(gate, up)


def packed_swiglu(packed: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU to ``packed[..., 2F]`` and return ``[..., F]``."""
    implementation = resolve_implementation(
        "packed_swiglu", packed, surface="semantic"
    )
    return implementation.apply(packed)


__all__ = ["packed_swiglu", "swiglu"]
