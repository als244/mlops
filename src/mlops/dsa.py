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


def _validate_sparse_latent_rope(q, kv, indices, q_rope, k_rope):
    _validate_sparse_latent(q, kv, indices)
    if q_rope is None or k_rope is None:
        raise ValueError("Both Q and K positional tails are required")
    if q_rope.shape != (*q.shape[:2], 64) or k_rope.shape != (q.shape[0], 64):
        raise ValueError("Expected positional Q [tokens, heads, 64] and K [tokens, 64]")
    if any(t.dtype != q.dtype or t.device != q.device for t in (q_rope, k_rope)):
        raise ValueError("Positional tails must share Q's dtype and device")


def _positional_arguments(q, kv, indices, q_rope, k_rope):
    if q_rope is None and k_rope is None:
        return {}
    _validate_sparse_latent_rope(q, kv, indices, q_rope, k_rope)
    return {"q_rope": q_rope, "k_rope": k_rope}


def sparse_latent_attention(q, kv, indices, *, scale, q_rope=None, k_rope=None):
    """Selected attention with a shared latent KV bank and optional RoPE tails.

    Already-rotated tails contribute only to attention scores. Values and the
    output retain the latent width. Autograd includes both positional gradients.
    """
    _validate_sparse_latent(q, kv, indices)
    positional = _positional_arguments(q, kv, indices, q_rope, k_rope)
    implementation = resolve_implementation(
        "sparse_latent_attention",
        q,
        kv,
        indices,
        scale=float(scale),
        surface="semantic",
        **positional,
    )
    return implementation.apply(q, kv, indices, scale=float(scale), **positional)


__all__ = ["dsa_attention", "sparse_latent_attention"]
