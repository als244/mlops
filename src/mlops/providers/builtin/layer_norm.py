"""Builtin Triton LayerNorm implementation adapter."""

from __future__ import annotations

import torch

from ...dispatch import logical_costs as logical
from ...dispatch.context import weight_gradient_dtype
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.layer_norm import layer_norm_backward, layer_norm_forward


def _supports(x, weight, bias=None, _eps=1e-5, *, surface, **_kwargs):
    del surface
    tensors = (x, weight) if bias is None else (x, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return SupportResult.no("x, weight, and optional bias must be tensors")
    if not all(value.is_cuda and value.is_contiguous() for value in tensors):
        return SupportResult.no("requires contiguous CUDA tensors")
    if x.ndim < 1 or weight.ndim != 1 or x.shape[-1] != weight.numel():
        return SupportResult.no("weight must be rank one and match x.shape[-1]")
    if bias is not None and bias.shape != weight.shape:
        return SupportResult.no("bias must match weight")
    return SupportResult.yes()


def forward(x, weight, bias=None, eps=1e-5):
    return layer_norm_forward(x, weight, bias, float(eps))


def backward(grad_output, x, weight, mean, rstd, weight_grad_dtype=None):
    return layer_norm_backward(grad_output, x, weight, mean, rstd, weight_grad_dtype)


# The forward takes the weight-gradient dtype too, so a captured forward
# records the one its backward will use.
@torch.library.custom_op(
    "mlops::layer_norm_builtin_triton_fwd",
    mutates_args=(),
)
def _forward_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    weight_grad_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del weight_grad_dtype
    return forward(x, weight, bias, float(eps))


@_forward_op.register_fake
def _forward_fake(x, weight, bias, eps, weight_grad_dtype):
    del weight, bias, eps, weight_grad_dtype
    rows = x.numel() // x.shape[-1]
    statistics = torch.empty(rows, dtype=torch.float32, device=x.device)
    return torch.empty_like(x), statistics, torch.empty_like(statistics)


@torch.library.custom_op(
    "mlops::layer_norm_builtin_triton_bwd",
    mutates_args=(),
)
def _backward_op(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight_grad_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return backward(grad_output, x, weight, mean, rstd, weight_grad_dtype)


@_backward_op.register_fake
def _backward_fake(grad_output, x, weight, mean, rstd, weight_grad_dtype):
    del grad_output, mean, rstd
    return (
        torch.empty_like(x),
        torch.empty_like(weight, dtype=weight_grad_dtype),
        torch.empty_like(weight, dtype=weight_grad_dtype),
    )


@flop_formula(_forward_op)
def _forward_flops(x, weight, bias, *_rest, out_val=None, **_kwargs):
    del out_val
    return logical.layer_norm(x, weight, bias, entrypoint="forward").logical_flops


@flop_formula(_backward_op)
def _backward_flops(grad_output, x, weight, *_rest, out_val=None, **_kwargs):
    del grad_output, out_val
    return logical.layer_norm(x, weight, entrypoint="backward").logical_flops


def _setup_context(ctx, inputs, output):
    x, weight, bias, _eps, weight_grad_dtype = inputs
    _normalized, mean, rstd = output
    ctx.save_for_backward(x, weight, mean, rstd)
    ctx.has_bias = bias is not None
    ctx.weight_grad_dtype = weight_grad_dtype
    ctx.mark_non_differentiable(mean, rstd)


def _autograd_backward(ctx, grad_output, _grad_mean, _grad_rstd):
    x, weight, mean, rstd = ctx.saved_tensors
    grad_x, grad_weight, grad_bias = _backward_op(
        grad_output, x, weight, mean, rstd, ctx.weight_grad_dtype
    )
    return grad_x, grad_weight, grad_bias if ctx.has_bias else None, None, None


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def apply(x, weight, bias=None, eps=1e-5):
    output, _mean, _rstd = _forward_op(
        x, weight, bias, float(eps), weight_gradient_dtype()
    )
    return output


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="layer_norm",
        implementation_id="builtin.layer_norm.triton",
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
