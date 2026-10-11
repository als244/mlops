"""MLA attention implementations behind one model-facing operation."""

from __future__ import annotations

import torch

from .dispatch import deterministic_required, resolve_implementation
from .dsa import sparse_latent_attention


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
    required = (
        deterministic_required() if deterministic is None else bool(deterministic)
    )
    implementation = resolve_implementation(
        "mla_attention",
        q,
        k,
        v,
        normalized_lengths,
        surface="semantic",
        deterministic=required,
    )
    return implementation.apply(q, k, v, normalized_lengths, deterministic=required)


def sparse_mla(query, latent, key_projection, value_projection, indices, *, scale):
    """Apply MLA without expanding per-token keys/values for every head.

    query [T,H,Dqk], latent [T,L], key_projection [H,Dqk,L],
    value_projection [H,Dv,L]. Scale is based on Dqk, not latent width L.
    """
    if key_projection.shape != (*query.shape[1:], latent.shape[-1]):
        raise ValueError("Key projection must be [heads, query_width, latent_width]")
    if (
        value_projection.ndim != 3
        or value_projection.shape[0] != query.shape[1]
        or value_projection.shape[-1] != latent.shape[-1]
    ):
        raise ValueError("Value projection must be [heads, value_width, latent_width]")
    q_latent = torch.einsum("thd,hdl->thl", query, key_projection)
    attended = sparse_latent_attention(q_latent, latent, indices, scale=scale)
    return torch.einsum("thl,hdl->thd", attended, value_projection)


__all__ = ["mla_attention", "sparse_mla"]
