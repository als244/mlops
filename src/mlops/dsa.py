"""Selected DeepSeek Sparse Attention semantic operation."""

from __future__ import annotations

import torch

from .dispatch import resolve_implementation


def dsa_attention(
    q,
    k,
    v,
    indices,
    lengths,
    *,
    reference_pad_value=False,
):
    """Apply selected sparse attention to packed token/head inputs."""
    lengths = tuple(int(length) for length in lengths)
    if sum(lengths) != q.shape[0]:
        raise ValueError("sequence lengths do not cover the DSA token axis")
    if q.ndim != 3 or q.shape != k.shape or q.shape[:2] != v.shape[:2]:
        raise ValueError("DSA expects compatible flat q/k/v tensors")
    implementation = resolve_implementation(
        "dsa_attention",
        q,
        k,
        v,
        indices,
        lengths,
        surface="semantic",
        reference_pad_value=bool(reference_pad_value),
    )
    return implementation.apply(
        q,
        k,
        v,
        indices,
        lengths,
        reference_pad_value=bool(reference_pad_value),
    )


def _validate_sparse_latent(q, kv, indices):
    if q.ndim != 3 or kv.shape != (q.shape[0], q.shape[-1]):
        raise ValueError("Expected Q [tokens, heads, latent] and KV [tokens, latent]")
    if q.shape[1] % 16 or q.shape[-1] not in (64, 128, 256, 512):
        raise ValueError(
            "Expected heads divisible by 16 and latent width 64/128/256/512"
        )
    if (
        indices.ndim != 2
        or indices.shape[0] != q.shape[0]
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected integer selected indices [tokens, selected_keys]")
    if q.dtype != torch.bfloat16 or kv.dtype != q.dtype:
        raise ValueError("Sparse TileLang attention currently requires BF16 Q and KV")
    if not all(t.is_cuda and t.device == q.device for t in (q, kv, indices)):
        raise ValueError("Sparse attention inputs must share one CUDA device")


def sparse_latent_attention(q, kv, indices, *, scale):
    """Selected attention with one latent KV tensor shared across query heads."""
    _validate_sparse_latent(q, kv, indices)
    implementation = resolve_implementation(
        "sparse_latent_attention",
        q,
        kv,
        indices,
        scale=float(scale),
        surface="semantic",
    )
    return implementation.apply(q, kv, indices, scale=float(scale))


__all__ = ["dsa_attention", "sparse_latent_attention"]
