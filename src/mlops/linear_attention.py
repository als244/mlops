"""Model-facing gated delta-rule semantic façade."""

from __future__ import annotations

from collections.abc import Sequence

from .dispatch import resolve_implementation


def linear_attention(
    q,
    k,
    v,
    beta,
    a,
    a_log,
    dt_bias,
    lengths: Sequence[int],
    cumulative,
    chunk_indices,
    *,
    scale=None,
):
    """Run packed Gated DeltaNet attention and return values shaped like ``v``.

    ``cumulative`` and ``chunk_indices`` are caller-owned round metadata from
    :func:`mlops.prepare_packed_sequence_metadata`.  Requiring them at this
    boundary prevents a provider from deriving host counts from CUDA tensors
    (a hidden device synchronization) or retaining an internal cache.
    """
    lengths = tuple(int(length) for length in lengths)
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    implementation = resolve_implementation(
        "linear_attention",
        q,
        k,
        v,
        beta,
        a,
        a_log,
        dt_bias,
        lengths,
        cumulative,
        chunk_indices,
        surface="semantic",
        scale=scale,
    )
    return implementation.apply(
        q,
        k,
        v,
        beta,
        a,
        a_log,
        dt_bias,
        lengths,
        cumulative,
        chunk_indices,
        scale=scale,
    )


__all__ = ["linear_attention"]
