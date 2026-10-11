"""Model-facing SwiGLU semantic façades."""

from __future__ import annotations

import torch

from .dispatch import resolve_implementation


def _validate_limit(limit):
    if limit is not None and not 0 < limit < float("inf"):
        raise ValueError("SwiGLU clipping limit must be finite and positive")


def swiglu(
    gate: torch.Tensor, up: torch.Tensor, *, limit: float | None = None
) -> torch.Tensor:
    """Return ``silu(gate) * up`` for equal-shaped ``[..., F]`` inputs."""
    _validate_limit(limit)
    implementation = resolve_implementation(
        "swiglu", gate, up, surface="semantic", limit=limit
    )
    return implementation.apply(gate, up, limit=limit)


def packed_swiglu(packed: torch.Tensor, *, limit: float | None = None) -> torch.Tensor:
    """Apply SwiGLU to ``packed[..., 2F]`` and return ``[..., F]``."""
    _validate_limit(limit)
    implementation = resolve_implementation(
        "packed_swiglu", packed, surface="semantic", limit=limit
    )
    return implementation.apply(packed, limit=limit)


__all__ = ["packed_swiglu", "swiglu"]
