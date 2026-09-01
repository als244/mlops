"""Selected DeepSeek Sparse Attention semantic operation."""

from __future__ import annotations

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


__all__ = ["dsa_attention"]
