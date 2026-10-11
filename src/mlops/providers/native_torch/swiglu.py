"""Ordinary native-PyTorch two-input and packed SwiGLU graphs."""

from __future__ import annotations

import torch
from torch.nn import functional

from ...dispatch.registry import Implementation, SupportResult, register_implementation


def _supports(gate, up, *, surface, **_kwargs):
    del surface
    if not isinstance(gate, torch.Tensor) or not isinstance(up, torch.Tensor):
        return SupportResult.no("gate and up must be tensors")
    if gate.shape != up.shape:
        return SupportResult.no("gate and up must have equal shape")
    return SupportResult.yes()


def _supports_packed(packed, *, surface, **_kwargs):
    del surface
    if not isinstance(packed, torch.Tensor) or packed.ndim < 1:
        return SupportResult.no("packed must be a tensor with a packed final dimension")
    if packed.shape[-1] % 2:
        return SupportResult.no("packed width must be even")
    return SupportResult.yes()


def apply(gate, up, *, limit=None):
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return functional.silu(gate.float()).to(gate.dtype) * up


def forward(gate, up, *, limit=None):
    return apply(gate, up, limit=limit)


def backward(grad_output, gate, up, *, limit=None):
    raw_gate, raw_up = gate, up
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    gate_float = gate.float()
    sigmoid = torch.sigmoid(gate_float)
    grad_silu = (grad_output * up).to(gate.dtype).float()
    silu = (gate_float * sigmoid).to(gate.dtype).float()
    grad_gate = (grad_silu * sigmoid) * (1.0 + gate_float * (1.0 - sigmoid))
    grad_gate = grad_gate.to(gate.dtype)
    grad_up = (grad_output.float() * silu).to(up.dtype)
    if limit is not None:
        grad_gate = grad_gate.masked_fill(raw_gate > limit, 0)
        grad_up = grad_up.masked_fill((raw_up < -limit) | (raw_up > limit), 0)
    return grad_gate, grad_up


def packed_apply(packed, *, limit=None):
    width = packed.shape[-1] // 2
    return apply(packed[..., :width], packed[..., width:], limit=limit)


def packed_forward(packed, *, limit=None):
    return packed_apply(packed, limit=limit)


def packed_backward(grad_output, packed, *, limit=None):
    width = packed.shape[-1] // 2
    grad_gate, grad_up = backward(
        grad_output, packed[..., :width], packed[..., width:], limit=limit
    )
    return torch.cat((grad_gate, grad_up), dim=-1)


IMPLEMENTATIONS = tuple(
    register_implementation(implementation)
    for implementation in (
        Implementation(
            "swiglu",
            "native_torch.swiglu",
            "native_torch",
            0,
            True,
            _supports,
            apply,
            forward,
            backward,
        ),
        Implementation(
            "packed_swiglu",
            "native_torch.packed_swiglu",
            "native_torch",
            0,
            True,
            _supports_packed,
            packed_apply,
            packed_forward,
            packed_backward,
        ),
    )
)

__all__ = ["IMPLEMENTATIONS"]
