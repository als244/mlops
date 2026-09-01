"""Builtin Triton two-input and packed SwiGLU implementations."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.swiglu import (
    swiglu_backward,
    swiglu_forward,
    swiglu_packed_backward,
    swiglu_packed_forward,
)


def _supports(gate, up, *, surface, **_kwargs):
    del surface
    if not isinstance(gate, torch.Tensor) or not isinstance(up, torch.Tensor):
        return SupportResult.no("gate and up must be tensors")
    if gate.shape != up.shape:
        return SupportResult.no("gate and up must have equal shape")
    if not all(value.is_cuda and value.is_contiguous() for value in (gate, up)):
        return SupportResult.no("requires contiguous CUDA tensors")
    return SupportResult.yes()


def _supports_packed(packed, *, surface, **_kwargs):
    del surface
    if not isinstance(packed, torch.Tensor):
        return SupportResult.no("packed must be a tensor")
    if packed.ndim != 2 or packed.shape[-1] % 2:
        return SupportResult.no("packed must have shape [rows, 2 * width]")
    if not packed.is_cuda or not packed.is_contiguous():
        return SupportResult.no("requires a contiguous CUDA tensor")
    return SupportResult.yes()


def forward(gate, up):
    return swiglu_forward(gate, up)


def backward(grad_output, gate, up):
    return swiglu_backward(grad_output, gate, up, variant="triton")


def legacy_backward(grad_output, gate, up):
    return swiglu_backward(grad_output, gate, up, variant="triton_legacy")


@torch.library.custom_op(
    "mlops::swiglu_builtin_triton_fwd", mutates_args=()
)
def _forward_op(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return forward(gate, up)


@_forward_op.register_fake
def _forward_fake(gate, up):
    del up
    return torch.empty_like(gate)


@torch.library.custom_op(
    "mlops::swiglu_builtin_triton_bwd", mutates_args=()
)
def _backward_op(
    grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return backward(grad_output, gate, up)


@_backward_op.register_fake
def _backward_fake(grad_output, gate, up):
    del grad_output
    return torch.empty_like(gate), torch.empty_like(up)


def _setup_context(ctx, inputs, output):
    del output
    gate, up = inputs
    ctx.save_for_backward(gate, up)


def _autograd_backward(ctx, grad_output):
    gate, up = ctx.saved_tensors
    return _backward_op(grad_output, gate, up)


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)


@torch.library.custom_op(
    "mlops::swiglu_builtin_triton_legacy_fwd", mutates_args=()
)
def _legacy_forward_op(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return forward(gate, up)


@_legacy_forward_op.register_fake
def _legacy_forward_fake(gate, up):
    del up
    return torch.empty_like(gate)


@torch.library.custom_op(
    "mlops::swiglu_builtin_triton_legacy_bwd", mutates_args=()
)
def _legacy_backward_op(
    grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return legacy_backward(grad_output, gate, up)


@_legacy_backward_op.register_fake
def _legacy_backward_fake(grad_output, gate, up):
    del grad_output
    return torch.empty_like(gate), torch.empty_like(up)


def _legacy_autograd_backward(ctx, grad_output):
    gate, up = ctx.saved_tensors
    return _legacy_backward_op(grad_output, gate, up)


_legacy_forward_op.register_autograd(
    _legacy_autograd_backward, setup_context=_setup_context
)


def apply(gate, up):
    return _forward_op(gate, up)



def legacy_apply(gate, up):
    return _legacy_forward_op(gate, up)


def packed_forward(packed):
    return swiglu_packed_forward(packed)


def packed_backward(grad_output, packed):
    return swiglu_packed_backward(grad_output, packed, variant="triton")


def packed_legacy_backward(grad_output, packed):
    return swiglu_packed_backward(grad_output, packed, variant="triton_legacy")


@torch.library.custom_op(
    "mlops::packed_swiglu_builtin_triton_fwd", mutates_args=()
)
def _packed_forward_op(packed: torch.Tensor) -> torch.Tensor:
    return packed_forward(packed)


@_packed_forward_op.register_fake
def _packed_forward_fake(packed):
    return packed.new_empty((packed.shape[0], packed.shape[1] // 2))


@torch.library.custom_op(
    "mlops::packed_swiglu_builtin_triton_bwd", mutates_args=()
)
def _packed_backward_op(
    grad_output: torch.Tensor, packed: torch.Tensor
) -> torch.Tensor:
    return packed_backward(grad_output, packed)


@_packed_backward_op.register_fake
def _packed_backward_fake(grad_output, packed):
    del grad_output
    return torch.empty_like(packed)


def _setup_packed_context(ctx, inputs, output):
    del output
    (packed,) = inputs
    ctx.save_for_backward(packed)


def _packed_autograd_backward(ctx, grad_output):
    (packed,) = ctx.saved_tensors
    return _packed_backward_op(grad_output, packed)


_packed_forward_op.register_autograd(
    _packed_autograd_backward, setup_context=_setup_packed_context
)


@torch.library.custom_op(
    "mlops::packed_swiglu_builtin_triton_legacy_fwd",
    mutates_args=(),
)
def _packed_legacy_forward_op(packed: torch.Tensor) -> torch.Tensor:
    return packed_forward(packed)


@_packed_legacy_forward_op.register_fake
def _packed_legacy_forward_fake(packed):
    return packed.new_empty((packed.shape[0], packed.shape[1] // 2))


@torch.library.custom_op(
    "mlops::packed_swiglu_builtin_triton_legacy_bwd",
    mutates_args=(),
)
def _packed_legacy_backward_op(
    grad_output: torch.Tensor, packed: torch.Tensor
) -> torch.Tensor:
    return packed_legacy_backward(grad_output, packed)


@_packed_legacy_backward_op.register_fake
def _packed_legacy_backward_fake(grad_output, packed):
    del grad_output
    return torch.empty_like(packed)


def _packed_legacy_autograd_backward(ctx, grad_output):
    (packed,) = ctx.saved_tensors
    return _packed_legacy_backward_op(grad_output, packed)


_packed_legacy_forward_op.register_autograd(
    _packed_legacy_autograd_backward, setup_context=_setup_packed_context
)


def packed_apply(packed):
    return _packed_forward_op(packed)


def packed_legacy_apply(packed):
    return _packed_legacy_forward_op(packed)


IMPLEMENTATIONS = tuple(
    register_implementation(implementation)
    for implementation in (
        Implementation(
            "swiglu", "builtin.swiglu.triton", "builtin", 100, True,
            _supports, apply, forward, backward,
        ),
        Implementation(
            "swiglu", "builtin.swiglu.triton_legacy", "builtin", 10, True,
            _supports, legacy_apply, forward, legacy_backward,
        ),
        Implementation(
            "packed_swiglu", "builtin.packed_swiglu.triton", "builtin", 100, True,
            _supports_packed, packed_apply, packed_forward, packed_backward,
        ),
        Implementation(
            "packed_swiglu", "builtin.packed_swiglu.triton_legacy", "builtin", 10, True,
            _supports_packed, packed_legacy_apply, packed_forward,
            packed_legacy_backward,
        ),
    )
)

__all__ = ["IMPLEMENTATIONS"]
