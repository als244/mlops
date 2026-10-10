"""Caller-side NoPE MLA composition; projections are outside the kernel operation."""

import torch

from .attention import sparse_latent_attention


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
