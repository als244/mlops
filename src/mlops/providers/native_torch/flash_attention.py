"""Ordinary native-PyTorch variable-length SDPA graph."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.flash_attention import native_torch_attention


def _supports(q, k, v, cu_seqlens, max_seqlen, *, surface, **_kwargs):
    del max_seqlen
    if surface == "explicit":
        return SupportResult.no("native SDPA is apply-only; use autograd for its VJP")
    if not all(isinstance(value, torch.Tensor) for value in (q, k, v, cu_seqlens)):
        return SupportResult.no("q, k, v, and cu_seqlens must be tensors")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
        return SupportResult.no("cu_seqlens must be one-dimensional int32")
    if cu_seqlens.device != q.device:
        return SupportResult.no("cu_seqlens must be on the queries' device")
    return SupportResult.yes()


def apply(
    q, k, v, cu_seqlens, max_seqlen, *, causal=True, softmax_scale=None, deterministic=False
):
    # Dense SDPA per sequence is already order-stable; the registration says
    # so, and the request needs no kernel change here.
    del max_seqlen, deterministic
    return native_torch_attention(
        q, k, v, cu_seqlens, causal=causal, softmax_scale=softmax_scale
    )


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="flash_attention",
        implementation_id="native_torch.flash_attention",
        provider="native_torch",
        priority=0,
        deterministic=True,
        supports=_supports,
        apply=apply,
    )
)

__all__ = ["IMPLEMENTATION", "apply"]
