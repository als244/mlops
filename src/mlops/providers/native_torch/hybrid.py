"""Ordinary native-PyTorch graphs for hybrid operation families."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.causal_conv import causal_conv_forward
from ...kernels.gated_rms_norm import gated_rms_norm_forward
from ...kernels.l2_norm import l2_norm_forward
from ...kernels.linear_attention import linear_attention_forward


def _supports(first, *_args, surface, **_kwargs):
    if surface == "explicit":
        return SupportResult.no("native hybrid graphs are apply-only")
    if not isinstance(first, torch.Tensor):
        return SupportResult.no("the primary input must be a tensor")
    return SupportResult.yes()


def causal_apply(x, weight, lengths, cumulative, chunk_indices=None):
    del cumulative, chunk_indices
    flat = x.reshape(-1, x.shape[-1])
    return causal_conv_forward(flat, weight, lengths).reshape_as(x)


def l2_apply(x, eps=1e-6):
    return l2_norm_forward(x, float(eps))[0]


def gated_apply(x, gate, weight, eps=1e-5):
    return gated_rms_norm_forward(x, gate, weight, float(eps))[0]


def linear_apply(
    q,
    k,
    v,
    beta,
    a,
    a_log,
    dt_bias,
    lengths,
    cumulative,
    chunk_indices,
    *,
    scale=None,
):
    del cumulative, chunk_indices
    resolved_scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    return linear_attention_forward(
        q, k, v, beta, a, a_log, dt_bias, lengths, resolved_scale
    )


IMPLEMENTATIONS = tuple(
    register_implementation(implementation)
    for implementation in (
        Implementation(
            operation="causal_conv_silu",
            implementation_id="native_torch.causal_conv_silu",
            provider="native_torch", priority=0, deterministic=True,
            supports=_supports, apply=causal_apply,
        ),
        Implementation(
            operation="l2_norm", implementation_id="native_torch.l2_norm",
            provider="native_torch", priority=0, deterministic=True,
            supports=_supports, apply=l2_apply,
        ),
        Implementation(
            operation="gated_rms_norm",
            implementation_id="native_torch.gated_rms_norm",
            provider="native_torch", priority=0, deterministic=True,
            supports=_supports, apply=gated_apply,
        ),
        Implementation(
            operation="linear_attention",
            implementation_id="native_torch.linear_attention",
            provider="native_torch", priority=0, deterministic=True,
            supports=_supports, apply=linear_apply,
        ),
    )
)

__all__ = ["IMPLEMENTATIONS"]
