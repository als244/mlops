"""Ordinary native-PyTorch MLA attention core."""

from __future__ import annotations

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.flash_attention import native_torch_attention


def _supports(q, k, v, lengths, *, surface, **_kwargs):
    if surface == "explicit":
        return SupportResult.no("native MLA attention is apply-only")
    if v.shape[-1] > q.shape[-1]:
        return SupportResult.no("MLA value width cannot exceed q/k width")
    if sum(int(length) for length in lengths) != q.shape[0]:
        return SupportResult.no("lengths must cover the query token axis")
    return SupportResult.yes()


def apply(q, k, v, lengths, *, deterministic=False):
    # Dense SDPA per sequence is already order-stable; the registration says
    # so, and the request needs no kernel change here.
    del deterministic
    return native_torch_attention(q, k, v, lengths)


IMPLEMENTATION = register_implementation(
    Implementation(
        "mla_attention", "native_torch.mla_attention", "native_torch", 0, True,
        _supports, apply,
    )
)

__all__ = ["IMPLEMENTATION", "apply"]
