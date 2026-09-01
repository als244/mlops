"""Portable native-PyTorch LayerNorm graph and explicit VJP."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation


def _supports(x, weight, bias=None, _eps=1e-5, *, surface, **_kwargs):
    del surface
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return SupportResult.no("x and weight must be tensors")
    if x.ndim < 1 or weight.ndim != 1 or x.shape[-1] != weight.numel():
        return SupportResult.no("weight must be rank one and match x.shape[-1]")
    if bias is not None and (not isinstance(bias, torch.Tensor) or bias.shape != weight.shape):
        return SupportResult.no("bias must be None or match weight")
    return SupportResult.yes()


def _statistics(x, eps):
    compute = x if x.dtype == torch.float64 else x.float()
    mean = compute.mean(-1, keepdim=True)
    rstd = torch.rsqrt((compute - mean).square().mean(-1, keepdim=True) + float(eps))
    return compute, mean, rstd


def apply(x, weight, bias=None, eps=1e-5):
    compute, mean, rstd = _statistics(x, eps)
    output = ((compute - mean) * rstd).to(x.dtype) * weight
    if bias is not None:
        output = output + bias
    return output.to(x.dtype)


def forward(x, weight, bias=None, eps=1e-5):
    width = x.shape[-1]
    rows = x.numel() // width
    compute, mean, rstd = _statistics(x.reshape(rows, width), eps)
    normalized = ((compute - mean) * rstd).to(x.dtype)
    output = normalized * weight
    if bias is not None:
        output = output + bias
    return output.to(x.dtype).reshape_as(x), mean.flatten(), rstd.flatten()


def backward(grad_output, x, weight, mean, rstd):
    width = x.shape[-1]
    rows = x.numel() // width
    x_rows = x.reshape(rows, width)
    compute = x_rows if x.dtype == torch.float64 else x_rows.float()
    normalized = (compute - mean.unsqueeze(-1)) * rstd.unsqueeze(-1)
    normalized_storage = normalized.to(x.dtype)
    grad_rows = grad_output.contiguous().reshape(rows, width)
    grad_storage = (grad_rows * weight).to(x.dtype)
    grad_normalized = grad_storage if x.dtype == torch.float64 else grad_storage.float()
    grad_x = rstd.unsqueeze(-1) * (
        grad_normalized
        - grad_normalized.mean(-1, keepdim=True)
        - normalized * (grad_normalized * normalized).mean(-1, keepdim=True)
    )
    return (
        grad_x.to(x.dtype).reshape_as(x),
        (grad_rows * normalized_storage).sum(0).to(weight.dtype),
        grad_rows.sum(0).to(weight.dtype),
    )


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="layer_norm",
        implementation_id="native_torch.layer_norm",
        provider="native_torch",
        priority=0,
        deterministic=True,
        supports=_supports,
        apply=apply,
        forward=forward,
        backward=backward,
    )
)

__all__ = ["IMPLEMENTATION", "apply", "backward", "forward"]
