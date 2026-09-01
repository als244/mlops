"""Explicit variable-length FlashAttention forward and VJP entry points."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..dispatch import resolve_implementation


def _validate(q, k, v, cu_seqlens, max_seqlen, lengths):
    normalized = tuple(int(length) for length in lengths)
    if not normalized or any(length <= 0 for length in normalized):
        raise ValueError("lengths must contain positive sequence lengths")
    if sum(normalized) != q.shape[0]:
        raise ValueError("sequence lengths do not cover the query token axis")
    if cu_seqlens.dtype != torch.int32 or cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be a one-dimensional INT32 tensor")
    if cu_seqlens.numel() != len(normalized) + 1:
        raise ValueError("cu_seqlens size does not match lengths")
    if int(max_seqlen) != max(normalized):
        raise ValueError("max_seqlen does not match lengths")
    if not all(tensor.is_contiguous() for tensor in (q, k, v, cu_seqlens)):
        raise ValueError("explicit FlashAttention inputs must be contiguous")
    return normalized


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    lengths: Sequence[int],
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(output, lse)`` for caller-supplied varlen metadata."""
    normalized = _validate(q, k, v, cu_seqlens, max_seqlen, lengths)
    implementation = resolve_implementation(
        "flash_attention",
        q,
        k,
        v,
        normalized,
        surface="explicit",
        causal=bool(causal),
        softmax_scale=softmax_scale,
    )
    with torch.no_grad():
        return implementation.forward(
            q,
            k,
            v,
            cu_seqlens,
            int(max_seqlen),
            normalized,
            causal=bool(causal),
            softmax_scale=softmax_scale,
        )


def backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    lengths: Sequence[int],
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(grad_q, grad_k, grad_v)`` without invoking autograd."""
    normalized = _validate(q, k, v, cu_seqlens, max_seqlen, lengths)
    implementation = resolve_implementation(
        "flash_attention",
        q,
        k,
        v,
        normalized,
        surface="explicit",
        causal=bool(causal),
        softmax_scale=softmax_scale,
    )
    with torch.no_grad():
        return implementation.backward(
            grad_output,
            q,
            k,
            v,
            output,
            lse,
            cu_seqlens,
            int(max_seqlen),
            normalized,
            causal=bool(causal),
            softmax_scale=softmax_scale,
        )


__all__ = ["backward", "forward"]
