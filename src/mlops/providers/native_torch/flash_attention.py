"""Ordinary native-PyTorch variable-length SDPA graph."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.flash_attention import native_torch_attention


def _supports(q, k, v, lengths, *, surface, **_kwargs):
    if surface == "explicit":
        return SupportResult.no("native SDPA is apply-only; use autograd for its VJP")
    if not all(isinstance(value, torch.Tensor) for value in (q, k, v)):
        return SupportResult.no("q, k, and v must be tensors")
    normalized = tuple(int(length) for length in lengths)
    if not normalized or any(length <= 0 for length in normalized):
        return SupportResult.no("lengths must contain positive values")
    if sum(normalized) != q.shape[0]:
        return SupportResult.no("lengths must cover the query token axis")
    return SupportResult.yes()


def apply(q, k, v, lengths, *, causal=True, softmax_scale=None):
    return native_torch_attention(
        q, k, v, lengths, causal=causal, softmax_scale=softmax_scale
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
