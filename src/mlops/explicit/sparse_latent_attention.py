"""Stateless selected-attention entry points for one shared latent KV bank."""

import torch

from ..dispatch import resolve_implementation
from ..dsa import _validate_sparse_latent


def forward(q, kv, indices, *, scale):
    """Return (output, logsumexp); both have a row for each query head."""
    _validate_sparse_latent(q, kv, indices)
    implementation = resolve_implementation(
        "sparse_latent_attention", q, kv, indices, scale=scale, surface="explicit"
    )
    with torch.no_grad():
        return implementation.forward(q, kv, indices, scale=scale)


def backward(dy, q, kv, indices, output, lse, *, scale):
    """Return (dQ, dKV); dKV combines key and value uses of the same bank."""
    _validate_sparse_latent(q, kv, indices)
    implementation = resolve_implementation(
        "sparse_latent_attention", q, kv, indices, scale=scale, surface="explicit"
    )
    with torch.no_grad():
        return implementation.backward(dy, q, kv, indices, output, lse, scale=scale)


__all__ = ["backward", "forward"]
