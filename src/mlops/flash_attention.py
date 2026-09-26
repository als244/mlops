"""Model-facing variable-length attention semantic façade."""

from __future__ import annotations

import torch

from .dispatch import deterministic_required, resolve_implementation


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
    deterministic: bool | None = None,
) -> torch.Tensor:
    """Apply an exact variable-length attention implementation.

    ``cu_seqlens`` are the cumulative sequence offsets along the token axis, a
    one-dimensional integer tensor from 0 to the token count, and
    ``max_seqlen`` bounds the longest sequence. The offsets are data: a
    captured graph takes them as an input, so one graph serves every packing
    of the same tokens. A repeated trailing offset is an empty sequence, which
    pads the tensor to a fixed size.

    ``deterministic`` asks for a backward whose accumulation order is fixed,
    so one step from one seed always lands on the same gradients.  It costs
    throughput, so it defaults to whatever ``deterministic_kernels`` is in
    effect, which is off.
    """
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    # Resolve the request here rather than inside the implementation: the
    # answer is baked into any graph captured from this call, and a context
    # variable read at replay time would report the wrong era.
    required = deterministic_required() if deterministic is None else bool(deterministic)
    implementation = resolve_implementation(
        "flash_attention",
        q,
        k,
        v,
        cu_seqlens,
        int(max_seqlen),
        surface="semantic",
        causal=bool(causal),
        softmax_scale=softmax_scale,
        deterministic=required,
    )
    return implementation.apply(
        q,
        k,
        v,
        cu_seqlens,
        int(max_seqlen),
        causal=bool(causal),
        softmax_scale=softmax_scale,
        deterministic=required,
    )


__all__ = ["flash_attention"]
