"""Gated RMSNorm provider with a standalone torch fallback."""

from __future__ import annotations

import torch


def gated_rms_norm_forward(x, gate, weight, eps):
    x_float = x.float()
    rstd = torch.rsqrt(x_float.square().mean(-1) + float(eps))
    output = (
        torch.nn.functional.silu(gate.float())
        * (x_float * rstd.unsqueeze(-1))
        * weight.float()
    ).to(x.dtype)
    return output, rstd


def fla_gated_rms_norm_forward(x, gate, weight, eps):
    from fla.modules.fused_norm_gate import layer_norm_gated_fwd

    shape = x.shape
    output, _mean, rstd, _residual = layer_norm_gated_fwd(
        x.reshape(-1, shape[-1]).contiguous(),
        gate.reshape(-1, shape[-1]).contiguous(),
        weight,
        bias=None,
        activation="swish",
        eps=float(eps),
        is_rms_norm=True,
    )
    return output.reshape(shape), rstd


def fla_gated_rms_norm_backward(grad_output, x, gate, weight, rstd, eps):
    from fla.modules.fused_norm_gate import layer_norm_gated_bwd

    shape = x.shape
    grad_x, grad_gate, grad_weight, _bias, _residual = layer_norm_gated_bwd(
        dy=grad_output.reshape(-1, shape[-1]).contiguous(),
        x=x.reshape(-1, shape[-1]).contiguous(),
        g=gate.reshape(-1, shape[-1]).contiguous(),
        weight=weight,
        bias=None,
        activation="swish",
        eps=float(eps),
        rstd=rstd,
        is_rms_norm=True,
    )
    return grad_x.reshape(shape), grad_gate.reshape(shape), grad_weight
