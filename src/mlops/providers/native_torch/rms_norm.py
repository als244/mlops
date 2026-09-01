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


def apply(x, weight, eps=1e-5):
    """Apply the ordinary PyTorch RMSNorm graph with Llama cast semantics."""
    # Preserve float64 only for numerical gradcheck. Production float16,
    # bfloat16, and float32 calls retain the reference model's fp32 reduction.
    compute = x if x.dtype == torch.float64 else x.float()
    rstd = torch.rsqrt(compute.square().mean(-1, keepdim=True) + float(eps))
    return ((compute * rstd).to(x.dtype) * weight).to(x.dtype)


def forward(x, weight, eps=1e-5):
    """Return native-PyTorch ``(output, rstd)`` without building autograd state."""
    width = x.shape[-1]
    rows = x.numel() // width
    x_rows = x.reshape(rows, width)
    compute = x_rows if x.dtype == torch.float64 else x_rows.float()
    rstd = torch.rsqrt(compute.square().mean(-1) + float(eps))
    x_hat = (compute * rstd.unsqueeze(-1)).to(x.dtype)
    output = (x_hat * weight).to(x.dtype).reshape_as(x)
    return output, rstd


def backward(grad_output, x, weight, rstd):
    """Return native-PyTorch ``(grad_x, grad_weight)`` explicit VJP."""
    width = x.shape[-1]
    rows = x.numel() // width
    grad_output_rows = grad_output.contiguous().reshape(rows, width)
    x_rows = x.reshape(rows, width)
    compute = x_rows if x.dtype == torch.float64 else x_rows.float()
    x_hat = compute * rstd.unsqueeze(-1)
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
