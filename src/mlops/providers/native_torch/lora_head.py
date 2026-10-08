"""Full-logits PyTorch reference for a low-rank output head."""

from __future__ import annotations

from torch.nn import functional

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ..builtin.lora_head import _scale
from ..builtin.lora_head import _supports as _common_support
from .head import _normalizer


def _supports(*args, surface, **kwargs):
    if surface == "explicit":
        return SupportResult.no("full-logits PyTorch head is apply-only")
    return _common_support(*args, surface=surface, **kwargs)


def apply(
    hidden,
    head_weight,
    lora_a,
    lora_b,
    targets,
    *,
    scale=1.0,
    chunk_size=None,
    valid_rows=None,
    reduction="mean",
):
    del chunk_size
    normalizer = _normalizer(hidden, valid_rows, reduction)
    x = hidden.reshape(-1, hidden.shape[-1])
    a, b = lora_a.to(hidden.dtype), lora_b.to(hidden.dtype)
    logits = x @ head_weight.T + _scale(scale) * ((x @ a.T) @ b.T)
    labels = targets.reshape(-1).long()
    labels = labels.masked_fill(labels < 0, -100)
    return (
        functional.cross_entropy(logits.float(), labels, reduction="sum") / normalizer
    )


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="lora_head_loss",
        implementation_id="native_torch.lora_head_loss",
        provider="native_torch",
        priority=0,
        deterministic=True,
        supports=_supports,
        apply=apply,
    )
)

__all__ = ["IMPLEMENTATION", "apply"]
