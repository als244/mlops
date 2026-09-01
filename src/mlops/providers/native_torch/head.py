"""Ordinary full-logits native-PyTorch language-model head loss."""

from __future__ import annotations

import torch.nn.functional as functional

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ..builtin.head import _common_support


def _supports(
    hidden,
    head_weight,
    targets,
    *,
    surface,
    chunk_size=None,
    valid_rows=None,
    **_kwargs,
):
    del chunk_size, valid_rows
    if surface == "explicit":
        return SupportResult.no("native full-logits head is apply-only")
    return _common_support(hidden, head_weight, targets)


def _normalizer(hidden, valid_rows):
    rows = hidden.numel() // hidden.shape[-1]
    normalizer = rows if valid_rows is None else int(valid_rows)
    if not 0 < normalizer <= rows:
        raise ValueError(f"valid_rows must be in [1, {rows}]; got {normalizer}")
    return normalizer


def apply(
    hidden,
    head_weight,
    targets,
    *,
    chunk_size=None,
    valid_rows=None,
):
    """Materialize full logits and return the canonical scalar objective."""
    del chunk_size
    normalizer = _normalizer(hidden, valid_rows)
    hidden_2d = hidden.reshape(-1, hidden.shape[-1])
    logits = hidden_2d @ head_weight.T
    return functional.cross_entropy(
        logits.float(),
        targets.reshape(-1).long(),
        reduction="sum",
    ) / normalizer


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="head_loss",
        implementation_id="native_torch.head_loss",
        provider="native_torch",
        priority=0,
        deterministic=True,
        supports=_supports,
        apply=apply,
    )
)


__all__ = ["IMPLEMENTATION", "apply"]
