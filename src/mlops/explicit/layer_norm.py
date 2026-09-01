"""Explicit affine LayerNorm forward and VJP entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation


def forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(output, mean, rstd)`` with FP32 row statistics."""
    implementation = resolve_implementation(
        "layer_norm", x, weight, bias, float(eps), surface="explicit"
    )
    with torch.no_grad():
        return implementation.forward(x, weight, bias, float(eps))


def backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(grad_x, grad_weight, grad_bias)``.

    ``grad_bias`` is computed even when the corresponding forward omitted a
    bias; a caller that used ``bias=None`` must discard it.
    """
    implementation = resolve_implementation(
        "layer_norm", x, weight, surface="explicit"
    )
    with torch.no_grad():
        return implementation.backward(grad_output, x, weight, mean, rstd)


__all__ = ["backward", "forward"]
