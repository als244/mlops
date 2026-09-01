"""Ordinary native-PyTorch two-input and packed SwiGLU graphs."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

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
    if not isinstance(packed, torch.Tensor) or packed.ndim != 2:
        return SupportResult.no("packed must be a rank-two tensor")
    if packed.shape[-1] % 2:
        return SupportResult.no("packed width must be even")
    return SupportResult.yes()


def apply(gate, up):
    return functional.silu(gate.float()).to(gate.dtype) * up


def forward(gate, up):
    return apply(gate, up)


def backward(grad_output, gate, up):
    gate_float = gate.float()
    sigmoid = torch.sigmoid(gate_float)
    grad_silu = (grad_output * up).to(gate.dtype).float()
    silu = (gate_float * sigmoid).to(gate.dtype).float()
    grad_gate = (grad_silu * sigmoid) * (1.0 + gate_float * (1.0 - sigmoid))
    return grad_gate.to(gate.dtype), (grad_output.float() * silu).to(up.dtype)


def packed_apply(packed):
    width = packed.shape[-1] // 2
    return apply(packed[..., :width], packed[..., width:])


def packed_forward(packed):
    return packed_apply(packed)


def packed_backward(grad_output, packed):
    width = packed.shape[-1] // 2
    grad_gate, grad_up = backward(
        grad_output, packed[..., :width], packed[..., width:]
    )
    return torch.cat((grad_gate, grad_up), dim=-1)


IMPLEMENTATIONS = tuple(
    register_implementation(implementation)
    for implementation in (
        Implementation(
            "swiglu", "native_torch.swiglu", "native_torch", 0, True,
            _supports, apply, forward, backward,
        ),
        Implementation(
            "packed_swiglu", "native_torch.packed_swiglu", "native_torch", 0, True,
            _supports_packed, packed_apply, packed_forward, packed_backward,
        ),
    )
)

__all__ = ["IMPLEMENTATIONS"]
