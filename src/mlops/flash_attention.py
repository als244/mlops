"""Model-facing variable-length attention semantic façade."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .dispatch import deterministic_required, resolve_implementation


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: Sequence[int],
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
    deterministic: bool | None = None,
) -> torch.Tensor:
    """Apply an exact variable-length attention implementation.

    ``deterministic`` asks for a backward whose accumulation order is fixed,
    so one step from one seed always lands on the same gradients.  It costs
    throughput, so it defaults to whatever ``deterministic_kernels`` is in
    effect, which is off.
    """
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    normalized_lengths = tuple(int(length) for length in lengths)
    # Resolve the request here rather than inside the implementation: the
    # answer is baked into any graph captured from this call, and a context
    # variable read at replay time would report the wrong era.
    required = deterministic_required() if deterministic is None else bool(deterministic)
    implementation = resolve_implementation(
        "flash_attention",
        q,
        k,
        v,
        normalized_lengths,
        surface="semantic",
        causal=bool(causal),
        softmax_scale=softmax_scale,
        deterministic=required,
    )
    return implementation.apply(
        q,
        k,
        v,
        normalized_lengths,
        causal=bool(causal),
        softmax_scale=softmax_scale,
        deterministic=required,
    )


__all__ = ["flash_attention"]
