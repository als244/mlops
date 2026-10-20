"""Portable native-PyTorch RMSNorm graph and explicit VJP."""

from __future__ import annotations

import torch

from ...dispatch.costs import CostHints
from ...dispatch.registry import Implementation, SupportResult, register_implementation


def _supports(x, weight, _eps=1e-5, *, surface, **_kwargs) -> SupportResult:
    del surface
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return SupportResult.no("x and weight must be tensors")
    if x.ndim < 1 or weight.ndim != 1 or x.shape[-1] != weight.numel():
        return SupportResult.no("weight must be rank one and match x.shape[-1]")
    if x.device != weight.device:
        return SupportResult.no("x and weight must share a device")
    if not x.is_floating_point() or not weight.is_floating_point():
        return SupportResult.no("x and weight must be floating-point")
    return SupportResult.yes()


def _estimate(x, weight, _eps=1e-5, *, entrypoint, **_kwargs) -> CostHints:
    del x, weight, entrypoint, _kwargs
    return CostHints(
        notes=(
            "native-Torch physical traffic and workspace depend on compiler decomposition",
        )
    )


def _compute(value):
    # Preserve float64 only for numerical gradcheck. Production float16,
    # bfloat16, and float32 calls retain the operation's fp32 reduction.
    return value if value.dtype == torch.float64 else value.float()


def apply(x, weight, eps=1e-5, zero_centered=False):
    """Normalize in FP32, round to the input dtype, then apply the gain; a
    zero-centered weight is an offset from one, ``1 + weight`` applied to the
    FP32 normalized value before the one cast."""
    compute = _compute(x)
    rstd = torch.rsqrt(compute.square().mean(-1, keepdim=True) + float(eps))
    if zero_centered:
        return (compute * rstd * (1 + _compute(weight))).to(x.dtype)
    return ((compute * rstd).to(x.dtype) * weight).to(x.dtype)


def forward(x, weight, eps=1e-5, zero_centered=False):
    """Return native-PyTorch ``(output, rstd)`` without building autograd state."""
    width = x.shape[-1]
    rows = x.numel() // width
    x_rows = x.reshape(rows, width)
    compute = _compute(x_rows)
    rstd = torch.rsqrt(compute.square().mean(-1) + float(eps))
    if zero_centered:
        normalized = compute * rstd.unsqueeze(-1) * (1 + _compute(weight))
        return normalized.to(x.dtype).reshape_as(x), rstd
    x_hat = (compute * rstd.unsqueeze(-1)).to(x.dtype)
    output = (x_hat * weight).to(x.dtype).reshape_as(x)
    return output, rstd


def backward(grad_output, x, weight, rstd, zero_centered=False):
    """Return native-PyTorch ``(grad_x, grad_weight)`` explicit VJP."""
    width = x.shape[-1]
    rows = x.numel() // width
    grad_output_rows = grad_output.contiguous().reshape(rows, width)
    x_rows = x.reshape(rows, width)
    compute = _compute(x_rows)
    x_hat = compute * rstd.unsqueeze(-1)
    if zero_centered:
        # The forward is FP32 until its one cast, so its VJP is too.
        grad_rows = _compute(grad_output_rows)
        grad_hat = grad_rows * (1 + _compute(weight))
        correction = (grad_hat * x_hat).mean(-1, keepdim=True)
        grad_x = rstd.unsqueeze(-1) * (grad_hat - x_hat * correction)
        grad_weight = (grad_rows * x_hat).sum(0).to(weight.dtype)
        return grad_x.to(x.dtype).reshape_as(x), grad_weight
    x_hat_storage = x_hat.to(x.dtype)
    grad_hat_storage = (grad_output_rows * weight).to(x.dtype)
    grad_hat = (
        grad_hat_storage
        if x.dtype == torch.float64
        else grad_hat_storage.float()
    )
    correction = (grad_hat * x_hat).mean(-1, keepdim=True)
    grad_x = (
        rstd.unsqueeze(-1) * (grad_hat - x_hat * correction)
    ).to(x.dtype).reshape_as(x)
    grad_weight = (grad_output_rows * x_hat_storage).sum(0).to(weight.dtype)
    return grad_x, grad_weight


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="rms_norm",
        implementation_id="native_torch.rms_norm",
        provider="native_torch",
        priority=0,
        deterministic=True,
        supports=_supports,
        apply=apply,
        forward=forward,
        backward=backward,
        estimate=_estimate,
    )
)


__all__ = ["IMPLEMENTATION", "apply", "backward", "forward"]
