"""MLA attention implementations behind one model-facing operation."""

from __future__ import annotations

from .dispatch import resolve_implementation


def mla_attention(q, k, v, lengths):
    """Attention core for assembled MLA heads.

    Low-rank projections, norms, and decoupled RoPE remain visible native
    PyTorch/custom operations in the surrounding forward-only module;
    this seam owns the assembled latent-attention launch and backward.
    """
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    normalized_lengths = tuple(int(length) for length in lengths)
    implementation = resolve_implementation(
        "mla_attention", q, k, v, normalized_lengths, surface="semantic"
    )
    return implementation.apply(q, k, v, normalized_lengths)


__all__ = ["mla_attention"]
