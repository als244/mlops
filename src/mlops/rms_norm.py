"""Model-facing RMSNorm semantic façade."""

from __future__ import annotations

import torch

from .dispatch import resolve_implementation


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    *,
    zero_centered: bool = False,
) -> torch.Tensor:
    """Apply the selected RMSNorm implementation.

    The normalized value is rounded to ``x``'s dtype and then scaled by
    ``weight``. A zero-centered weight (``zero_centered=True``, as Qwen 3.5
    keeps its norms) is an offset from one instead: the FP32 normalized value
    is scaled by ``1 + weight`` before the one cast, so stored weights stay
    small and the scale keeps full precision.
    """
    implementation = resolve_implementation(
        "rms_norm",
        x,
        weight,
        float(eps),
        surface="semantic",
        zero_centered=bool(zero_centered),
    )
    return implementation.apply(x, weight, float(eps), bool(zero_centered))


__all__ = ["rms_norm"]
