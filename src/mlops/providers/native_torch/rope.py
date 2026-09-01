"""Ordinary native-PyTorch RoPE graph and inverse-rotation VJP."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.rope import reference_rope, reference_rope_backward


def _supports(x, positions, _base, cosine, sine, *, surface, **_kwargs):
    del surface
    if not isinstance(x, torch.Tensor) or not isinstance(positions, torch.Tensor):
        return SupportResult.no("x and positions must be tensors")
    if x.shape[-1] % 2:
        return SupportResult.no("head dimension must be even")
    if x.device != positions.device:
        return SupportResult.no("x and positions must share a device")
    if cosine.shape != sine.shape or cosine.shape[-1] != x.shape[-1]:
        return SupportResult.no("cosine/sine tables must match the head width")
    return SupportResult.yes()


def apply(x, positions, base, cosine, sine):
    del base
    return reference_rope(x, positions, cosine, sine)


def forward(x, positions, base, cosine, sine):
    return apply(x, positions, base, cosine, sine)


def backward(grad_output, positions, base, cosine, sine):
    del base
    return reference_rope_backward(grad_output, positions, cosine, sine)


IMPLEMENTATION = register_implementation(
    Implementation(
        "rope", "native_torch.rope", "native_torch", 0, True,
        _supports, apply, forward, backward,
    )
)

__all__ = ["IMPLEMENTATION", "apply", "backward", "forward"]
