"""MLA attention implementations behind one model-facing operation."""

from __future__ import annotations

from .dispatch import deterministic_required, resolve_implementation


def mla_attention(q, k, v, lengths, *, deterministic: bool | None = None):
    """Attention core for assembled MLA heads.

    Low-rank projections, norms, and decoupled RoPE remain visible native
    PyTorch/custom operations in the surrounding forward-only module;
    this seam owns the assembled latent-attention launch and backward.

    ``deterministic`` carries the same meaning as it does for
    ``flash_attention``, which this operation launches.
    """
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    normalized_lengths = tuple(int(length) for length in lengths)
    required = deterministic_required() if deterministic is None else bool(deterministic)
    implementation = resolve_implementation(
        "mla_attention",
        q,
        k,
        v,
        normalized_lengths,
        surface="semantic",
        deterministic=required,
    )
    return implementation.apply(
        q, k, v, normalized_lengths, deterministic=required
    )


__all__ = ["mla_attention"]
