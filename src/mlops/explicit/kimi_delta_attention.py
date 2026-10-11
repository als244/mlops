"""Stateless KDA forward and VJP; all sequence metadata is explicit."""

import torch

from ..dispatch import resolve_implementation
from ..kda import _validate


def forward(q, k, v, g, beta, cumulative, chunks):
    """Return output, cumulative gates, and two intra-chunk matrices."""
    _validate(q, k, v, g, beta, cumulative, chunks)
    implementation = resolve_implementation(
        "kimi_delta_attention",
        q,
        k,
        v,
        g,
        beta,
        cumulative,
        chunks,
        surface="explicit",
    )
    with torch.no_grad():
        return implementation.forward(q, k, v, g, beta, cumulative, chunks)


def backward(dy, q, k, v, g, beta, gates, aqk, akk, cumulative, chunks):
    """Return dQ, dK, dV, dG, dBeta using the caller's forward residuals."""
    _validate(q, k, v, g, beta, cumulative, chunks)
    implementation = resolve_implementation(
        "kimi_delta_attention",
        q,
        k,
        v,
        g,
        beta,
        cumulative,
        chunks,
        surface="explicit",
    )
    with torch.no_grad():
        return implementation.backward(
            dy, q, k, v, g, beta, gates, aqk, akk, cumulative, chunks
        )


__all__ = ["backward", "forward"]
