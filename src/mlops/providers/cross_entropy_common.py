"""Provider-independent validation and reduction for cross entropy."""

from __future__ import annotations

import torch

from ..dispatch.registry import SupportResult


REDUCTIONS = frozenset({"mean", "none", "sum"})


def validate_cross_entropy(
    logits,
    targets,
    weight=None,
    ignore_index=-100,
    reduction="mean",
) -> SupportResult:
    """Validate the public metadata shared by every implementation."""
    del ignore_index
    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        return SupportResult.no("logits and targets must be tensors")
    if logits.ndim != 2:
        return SupportResult.no("logits must have shape [rows, vocabulary]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        return SupportResult.no("targets must have shape [rows]")
    if logits.shape[1] == 0:
        return SupportResult.no("vocabulary dimension must be nonempty")
    if not logits.is_floating_point():
        return SupportResult.no("logits must be floating-point")
    if targets.dtype not in {torch.int32, torch.int64}:
        return SupportResult.no("targets must have dtype int32 or int64")
    if targets.device != logits.device:
        return SupportResult.no("logits and targets must share a device")
    if reduction not in REDUCTIONS:
        return SupportResult.no("reduction must be 'none', 'sum', or 'mean'")
    if weight is not None:
        if not isinstance(weight, torch.Tensor):
            return SupportResult.no("weight must be a tensor or None")
        if weight.ndim != 1 or weight.numel() != logits.shape[1]:
            return SupportResult.no("weight must have shape [vocabulary]")
        if weight.device != logits.device or not weight.is_floating_point():
            return SupportResult.no(
                "weight must be floating-point and share the logits device"
            )
        if weight.requires_grad:
            return SupportResult.no("class weight is non-differentiable")
    return SupportResult.yes()


def reduce_losses(
    losses: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None,
    ignore_index: int,
    reduction: str,
) -> torch.Tensor:
    """Apply the canonical reduction to per-row weighted losses."""
    if reduction == "none":
        return losses
    total = losses.sum()
    if reduction == "sum":
        return total
    valid = targets != int(ignore_index)
    if weight is None:
        denominator = valid.sum().to(losses.dtype)
    else:
        safe_targets = targets.masked_fill(~valid, 0).long()
        denominator = (
            weight.to(losses.dtype).gather(0, safe_targets) * valid
        ).sum()
    return total / denominator


__all__ = ["REDUCTIONS", "reduce_losses", "validate_cross_entropy"]
