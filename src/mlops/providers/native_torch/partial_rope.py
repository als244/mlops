"""Ordinary native-PyTorch partial-channel RoPE implementation."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.rope import reference_rope, reference_rope_backward


def _supports(
    x, positions, _base, rotary_dim, cosine, sine, *, surface, **_kwargs
):
    del surface
    if not isinstance(x, torch.Tensor) or not isinstance(positions, torch.Tensor):
        return SupportResult.no("x and positions must be tensors")
    if int(rotary_dim) <= 0 or int(rotary_dim) > x.shape[-1] or int(rotary_dim) % 2:
        return SupportResult.no("rotary_dim must be positive, even, and within x")
    if cosine.shape != sine.shape or cosine.shape[-1] != int(rotary_dim):
        return SupportResult.no("cosine/sine tables must match rotary_dim")
    return SupportResult.yes()


def apply(x, positions, base, rotary_dim, cosine, sine):
    del base
    rotary_dim = int(rotary_dim)
    rotated = reference_rope(
        x[..., :rotary_dim], positions, cosine, sine
    )
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


def forward(x, positions, base, rotary_dim, cosine, sine):
    return apply(x, positions, base, rotary_dim, cosine, sine)


def backward(grad_output, positions, base, rotary_dim, cosine, sine):
    del base
    rotary_dim = int(rotary_dim)
    rotated = reference_rope_backward(
        grad_output[..., :rotary_dim], positions, cosine, sine
    )
    return torch.cat((rotated, grad_output[..., rotary_dim:]), dim=-1)


IMPLEMENTATION = register_implementation(
    Implementation(
        "partial_rope", "native_torch.partial_rope", "native_torch", 0, True,
        _supports, apply, forward, backward,
    )
)

__all__ = ["IMPLEMENTATION", "apply", "backward", "forward"]
