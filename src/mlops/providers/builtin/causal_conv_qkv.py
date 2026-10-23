"""Portable concatenate/convolve/split reference for the split-QKV boundary."""

import torch

from ...causal_conv import causal_conv_silu
from ...dispatch.registry import Implementation, SupportResult, register_implementation


def _supports(q, k, v, weight, cumulative, chunks=None, *, surface, **_kwargs):
    if surface != "semantic":
        return SupportResult.no("split-QKV convolution is a semantic composition")
    return SupportResult.yes()


def apply(q, k, v, weight, cumulative, chunks=None):
    out = causal_conv_silu(torch.cat((q, k, v), dim=-1), weight, cumulative, chunks)
    return out.split((q.shape[-1], k.shape[-1], v.shape[-1]), dim=-1)


IMPLEMENTATION = register_implementation(
    Implementation(
        "causal_conv_silu_qkv",
        "builtin.causal_conv_silu_qkv.composed",
        "builtin",
        0,
        True,
        _supports,
        apply,
    )
)
