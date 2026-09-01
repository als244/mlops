"""MLA core backed by the builtin variable-length FlashAttention operation."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.flash_attention import native_flash_attention_supported
from .flash_attention import apply as flash_apply


def _supports(q, k, v, lengths, *, surface, **_kwargs):
    if surface == "explicit":
        return SupportResult.no("MLA uses the flash-attention explicit boundary")
    if not all(isinstance(value, torch.Tensor) for value in (q, k, v)):
        return SupportResult.no("q, k, and v must be tensors")
    padded_width = q.shape[-1]
    if v.shape[-1] > padded_width:
        return SupportResult.no("MLA value width cannot exceed q/k width")
    probe = v if v.shape[-1] == padded_width else v.new_empty((*v.shape[:-1], padded_width))
    if not native_flash_attention_supported(q, k, probe):
        return SupportResult.no("PyTorch native variable-length flash is unsupported")
    return SupportResult.yes()


def apply(q, k, v, lengths):
    value_width = v.shape[-1]
    if value_width == q.shape[-1]:
        return flash_apply(q, k, v, lengths)
    padded = torch.cat(
        (v, v.new_zeros(*v.shape[:-1], q.shape[-1] - value_width)), dim=-1
    )
    return flash_apply(q, k, padded, lengths)[..., :value_width]


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="mla_attention",
        implementation_id="builtin.mla_attention.flash",
        provider="builtin",
        priority=100,
        deterministic=False,
        supports=_supports,
        apply=apply,
    )
)

__all__ = ["IMPLEMENTATION", "apply"]
