"""Stateless selected-attention entry points for one shared latent KV bank."""

import torch

from ..dispatch import resolve_implementation
from ..dsa import _positional_arguments, _validate_sparse_latent


def forward(q, kv, indices, *, scale, q_rope=None, k_rope=None):
    """Return (output, logsumexp); both have a row for each query head."""
    _validate_sparse_latent(q, kv, indices)
    positional = _positional_arguments(q, kv, indices, q_rope, k_rope)
    implementation = resolve_implementation(
        "sparse_latent_attention",
        q,
        kv,
        indices,
        scale=scale,
        surface="explicit",
        **positional,
    )
    with torch.no_grad():
        return implementation.forward(q, kv, indices, scale=scale, **positional)


def backward(dy, q, kv, indices, output, lse, *, scale, q_rope=None, k_rope=None):
    """Return (dQ, dKV), plus (dQ_rope, dK_rope) when positional tails are used."""
    _validate_sparse_latent(q, kv, indices)
    positional = _positional_arguments(q, kv, indices, q_rope, k_rope)
    implementation = resolve_implementation(
        "sparse_latent_attention",
        q,
        kv,
        indices,
        scale=scale,
        surface="explicit",
        **positional,
    )
    with torch.no_grad():
        return implementation.backward(
            dy, q, kv, indices, output, lse, scale=scale, **positional
        )


__all__ = ["backward", "forward"]
