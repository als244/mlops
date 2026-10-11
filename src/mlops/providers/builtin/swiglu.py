"""Builtin Triton two-input and packed SwiGLU implementations."""

from __future__ import annotations

import torch

from ...dispatch import logical_costs as logical
from ...dispatch.costs import flop_formula
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
    if packed.ndim < 1 or packed.shape[-1] % 2:
        return SupportResult.no("packed final dimension must contain gate and up")
    if not packed.is_cuda or not packed.is_contiguous():
        return SupportResult.no("requires a contiguous CUDA tensor")
    return SupportResult.yes()


def forward(gate, up, *, limit=None):
    return swiglu_forward(gate, up, limit=limit)


def backward(grad_output, gate, up, *, limit=None):
    return swiglu_backward(grad_output, gate, up, variant="triton", limit=limit)


def legacy_backward(grad_output, gate, up, *, limit=None):
    return swiglu_backward(grad_output, gate, up, variant="triton_legacy", limit=limit)


@torch.library.custom_op("mlops::swiglu_builtin_triton_fwd", mutates_args=())
def _forward_op(
    gate: torch.Tensor, up: torch.Tensor, limit: float | None = None
) -> torch.Tensor:
    return forward(gate, up, limit=limit)


@_forward_op.register_fake
def _forward_fake(gate, up, limit=None):
    del up
    return torch.empty_like(gate)


@torch.library.custom_op("mlops::swiglu_builtin_triton_bwd", mutates_args=())
def _backward_op(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    limit: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return backward(grad_output, gate, up, limit=limit)


@_backward_op.register_fake
def _backward_fake(grad_output, gate, up, limit=None):
    del grad_output
    return torch.empty_like(gate), torch.empty_like(up)


def _setup_context(ctx, inputs, output):
    del output
    gate, up, ctx.limit = inputs
    ctx.save_for_backward(gate, up)


def _autograd_backward(ctx, grad_output):
    gate, up = ctx.saved_tensors
    return (*_backward_op(grad_output, gate, up, ctx.limit), None)


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)


@torch.library.custom_op("mlops::swiglu_builtin_triton_legacy_fwd", mutates_args=())
def _legacy_forward_op(
    gate: torch.Tensor, up: torch.Tensor, limit: float | None = None
) -> torch.Tensor:
    return forward(gate, up, limit=limit)


@_legacy_forward_op.register_fake
def _legacy_forward_fake(gate, up, limit=None):
    del up
    return torch.empty_like(gate)


@torch.library.custom_op("mlops::swiglu_builtin_triton_legacy_bwd", mutates_args=())
def _legacy_backward_op(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    limit: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return legacy_backward(grad_output, gate, up, limit=limit)


@_legacy_backward_op.register_fake
def _legacy_backward_fake(grad_output, gate, up, limit=None):
    del grad_output
    return torch.empty_like(gate), torch.empty_like(up)


def _legacy_autograd_backward(ctx, grad_output):
    gate, up = ctx.saved_tensors
    return (*_legacy_backward_op(grad_output, gate, up, ctx.limit), None)


_legacy_forward_op.register_autograd(
    _legacy_autograd_backward, setup_context=_setup_context
)


def apply(gate, up, *, limit=None):
    return _forward_op(gate, up, limit)


def legacy_apply(gate, up, *, limit=None):
    return _legacy_forward_op(gate, up, limit)


def packed_forward(packed, *, limit=None):
    return swiglu_packed_forward(packed, limit=limit)


def packed_backward(grad_output, packed, *, limit=None):
    return swiglu_packed_backward(grad_output, packed, variant="triton", limit=limit)


def packed_legacy_backward(grad_output, packed, *, limit=None):
    return swiglu_packed_backward(
        grad_output, packed, variant="triton_legacy", limit=limit
    )


@torch.library.custom_op("mlops::packed_swiglu_builtin_triton_fwd", mutates_args=())
def _packed_forward_op(
    packed: torch.Tensor, limit: float | None = None
) -> torch.Tensor:
    return packed_forward(packed, limit=limit)


@_packed_forward_op.register_fake
def _packed_forward_fake(packed, limit=None):
    return packed.new_empty((*packed.shape[:-1], packed.shape[-1] // 2))


@torch.library.custom_op("mlops::packed_swiglu_builtin_triton_bwd", mutates_args=())
def _packed_backward_op(
    grad_output: torch.Tensor, packed: torch.Tensor, limit: float | None = None
) -> torch.Tensor:
    return packed_backward(grad_output, packed, limit=limit)


@_packed_backward_op.register_fake
def _packed_backward_fake(grad_output, packed, limit=None):
    del grad_output
    return torch.empty_like(packed)


def _setup_packed_context(ctx, inputs, output):
    del output
    packed, ctx.limit = inputs
    ctx.save_for_backward(packed)


def _packed_autograd_backward(ctx, grad_output):
    (packed,) = ctx.saved_tensors
    return _packed_backward_op(grad_output, packed, ctx.limit), None


_packed_forward_op.register_autograd(
    _packed_autograd_backward, setup_context=_setup_packed_context
)


@torch.library.custom_op(
    "mlops::packed_swiglu_builtin_triton_legacy_fwd",
    mutates_args=(),
)
def _packed_legacy_forward_op(
    packed: torch.Tensor, limit: float | None = None
) -> torch.Tensor:
    return packed_forward(packed, limit=limit)


@_packed_legacy_forward_op.register_fake
def _packed_legacy_forward_fake(packed, limit=None):
    return packed.new_empty((*packed.shape[:-1], packed.shape[-1] // 2))


@torch.library.custom_op(
    "mlops::packed_swiglu_builtin_triton_legacy_bwd",
    mutates_args=(),
)
def _packed_legacy_backward_op(
    grad_output: torch.Tensor, packed: torch.Tensor, limit: float | None = None
) -> torch.Tensor:
    return packed_legacy_backward(grad_output, packed, limit=limit)


@_packed_legacy_backward_op.register_fake
def _packed_legacy_backward_fake(grad_output, packed, limit=None):
    del grad_output
    return torch.empty_like(packed)


def _packed_legacy_autograd_backward(ctx, grad_output):
    (packed,) = ctx.saved_tensors
    return _packed_legacy_backward_op(grad_output, packed, ctx.limit), None


_packed_legacy_forward_op.register_autograd(
    _packed_legacy_autograd_backward, setup_context=_setup_packed_context
)


def packed_apply(packed, *, limit=None):
    return _packed_forward_op(packed, limit)


def packed_legacy_apply(packed, *, limit=None):
    return _packed_legacy_forward_op(packed, limit)


# The legacy kernels compute the same function; both variants of each pair
# share one formula.
@flop_formula(_forward_op, _legacy_forward_op)
def _forward_flops(gate, up, limit=None, *, out_val=None, **_kwargs):
    del out_val
    return logical.swiglu(gate, up, limit=limit, entrypoint="forward").logical_flops


@flop_formula(_backward_op, _legacy_backward_op)
def _backward_flops(grad_output, gate, up, limit=None, *, out_val=None, **_kwargs):
    del grad_output, out_val
    return logical.swiglu(gate, up, limit=limit, entrypoint="backward").logical_flops


@flop_formula(_packed_forward_op, _packed_legacy_forward_op)
def _packed_forward_flops(packed, limit=None, *, out_val=None, **_kwargs):
    del out_val
    return logical.packed_swiglu(
        packed, limit=limit, entrypoint="forward"
    ).logical_flops


@flop_formula(_packed_backward_op, _packed_legacy_backward_op)
def _packed_backward_flops(grad_output, packed, limit=None, *, out_val=None, **_kwargs):
    del grad_output, out_val
    return logical.packed_swiglu(
        packed, limit=limit, entrypoint="backward"
    ).logical_flops


IMPLEMENTATIONS = tuple(
    register_implementation(implementation)
    for implementation in (
        Implementation(
            "swiglu",
            "builtin.swiglu.triton",
            "builtin",
            100,
            True,
            _supports,
            apply,
            forward,
            backward,
        ),
        Implementation(
            "swiglu",
            "builtin.swiglu.triton_legacy",
            "builtin",
            10,
            True,
            _supports,
            legacy_apply,
            forward,
            legacy_backward,
        ),
        Implementation(
            "packed_swiglu",
            "builtin.packed_swiglu.triton",
            "builtin",
            100,
            True,
            _supports_packed,
            packed_apply,
            packed_forward,
            packed_backward,
        ),
        Implementation(
            "packed_swiglu",
            "builtin.packed_swiglu.triton_legacy",
            "builtin",
            10,
            True,
            _supports_packed,
            packed_legacy_apply,
            packed_forward,
            packed_legacy_backward,
        ),
    )
)

__all__ = ["IMPLEMENTATIONS"]
