"""Explicit Gated DeltaNet core forward and VJP entry points."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..dispatch import resolve_implementation


def forward(
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
    """Return ``(output, gate, matrix)`` using caller-owned round metadata."""
    normalized = tuple(int(length) for length in lengths)
    resolved_scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    implementation = resolve_implementation(
        "linear_attention",
        q,
        k,
        v,
        beta,
        a,
        a_log,
        dt_bias,
        normalized,
        cumulative,
        chunk_indices,
        surface="explicit",
        scale=resolved_scale,
    )
    with torch.no_grad():
        return implementation.forward(
            q,
            k,
            v,
            beta,
            a,
            a_log,
            dt_bias,
            normalized,
            cumulative,
            chunk_indices,
            scale=resolved_scale,
        )


def backward(
    grad_output,
    q,
    k,
    v,
    beta,
    a,
    a_log,
    dt_bias,
    gate,
    matrix,
    cumulative,
    chunk_indices,
    *,
    scale=None,
):
    """Return VJPs for ``q, k, v, beta, a, a_log, dt_bias`` in order."""
    resolved_scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    implementation = resolve_implementation(
        "linear_attention",
        q,
        k,
        v,
        beta,
        a,
        a_log,
        dt_bias,
        (),
        cumulative,
        chunk_indices,
        surface="explicit",
        scale=resolved_scale,
    )
    with torch.no_grad():
        return implementation.backward(
            grad_output,
            q,
            k,
            v,
            beta,
            a,
            a_log,
            dt_bias,
            gate,
            matrix,
            cumulative,
            chunk_indices,
            scale=resolved_scale,
        )


__all__ = ["backward", "forward"]
