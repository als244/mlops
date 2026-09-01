"""ATen tanh-GELU with an explicit registered VJP boundary."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.gelu import gelu_backward, gelu_forward


def _supports(x, *, surface, **_kwargs):
    del surface
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        return SupportResult.no("x must be a floating-point tensor")
    return SupportResult.yes()


def forward(x):
    return gelu_forward(x)


def backward(grad_output, x):
    return gelu_backward(grad_output, x)


@torch.library.custom_op(
    "mlops::gelu_builtin_aten_explicit_backward_fwd",
    mutates_args=(),
)
def _forward_op(x: torch.Tensor) -> torch.Tensor:
    return forward(x)


@_forward_op.register_fake
def _forward_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op(
    "mlops::gelu_builtin_aten_explicit_backward_bwd",
    mutates_args=(),
)
def _backward_op(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return backward(grad_output, x)


@_backward_op.register_fake
def _backward_fake(grad_output, x):
    del grad_output
    return torch.empty_like(x)


def _setup_context(ctx, inputs, output):
    del output
    (x,) = inputs
    ctx.save_for_backward(x)


def _autograd_backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    return _backward_op(grad_output, x)


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def apply(x):
    return _forward_op(x)


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="gelu",
        implementation_id="builtin.gelu.aten_explicit_backward",
        provider="builtin",
        priority=100,
        deterministic=True,
        supports=_supports,
        apply=apply,
        forward=forward,
        backward=backward,
    )
)

__all__ = ["IMPLEMENTATION", "apply", "backward", "forward"]
