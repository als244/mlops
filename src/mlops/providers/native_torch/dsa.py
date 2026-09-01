"""Ordinary native-PyTorch sparse-attention and indexer graphs."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.dsa_indexer import index_scores as raw_index_scores
from ...kernels.dsa_indexer import selection_mask


def _attention_support(q, k, v, indices, lengths, *, surface, **_kwargs):
    if surface == "explicit":
        return SupportResult.no("native DSA attention is apply-only")
    if sum(int(length) for length in lengths) != q.shape[0]:
        return SupportResult.no("lengths must cover the query token axis")
    return SupportResult.yes()


def attention_apply(q, k, v, indices, lengths, *, reference_pad_value=False):
    tokens = q.shape[0]
    mask = torch.full((tokens, tokens), float("-inf"), dtype=torch.float32, device=q.device)
    start = 0
    for length in lengths:
        stop = start + int(length)
        mask[start:stop, start:stop] = selection_mask(indices, start, stop, start, stop)
        start = stop
    value = v
    if reference_pad_value and v.shape[-1] < q.shape[-1]:
        value = torch.cat((v, v.new_zeros(*v.shape[:-1], q.shape[-1] - v.shape[-1])), dim=-1)
    output = functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        attn_mask=mask[None, None].to(q.dtype),
    )
    return output.squeeze(0).transpose(0, 1)[..., : v.shape[-1]]


def _key_norm_support(x, weight, bias, _eps=1e-5, *, surface, **_kwargs):
    if surface == "explicit":
        return SupportResult.no("native key norm is apply-only")
    return SupportResult.yes()


def key_norm_apply(x, weight, bias, eps=1e-5):
    return functional.layer_norm(
        x.float(), (x.shape[-1],), weight.float(), bias.float(), float(eps)
    )


def _score_support(q, k, weights, lengths, *, surface, **_kwargs):
    if surface == "explicit":
        return SupportResult.no("native index scoring is apply-only")
    return SupportResult.yes()


def score_apply(q, k, weights, lengths):
    return raw_index_scores(q, k, weights, lengths, reduction="multiply_sum")


IMPLEMENTATIONS = tuple(
    register_implementation(implementation)
    for implementation in (
        Implementation(
            operation="dsa_attention", implementation_id="native_torch.dsa_attention",
            provider="native_torch", priority=0, deterministic=True,
            supports=_attention_support, apply=attention_apply,
        ),
        Implementation(
            operation="dsa_indexer_key_norm",
            implementation_id="native_torch.dsa_indexer_key_norm",
            provider="native_torch", priority=0, deterministic=True,
            supports=_key_norm_support, apply=key_norm_apply,
        ),
        Implementation(
            operation="index_scores", implementation_id="native_torch.index_scores",
            provider="native_torch", priority=0, deterministic=True,
            supports=_score_support, apply=score_apply,
        ),
    )
)

__all__ = ["IMPLEMENTATIONS"]
