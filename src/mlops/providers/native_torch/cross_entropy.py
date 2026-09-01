"""Portable native-PyTorch cross entropy and explicit logits VJP."""

from __future__ import annotations

import torch

from ...dispatch.costs import CostHints
from ...dispatch.registry import Implementation, register_implementation
from ..cross_entropy_common import reduce_losses, validate_cross_entropy


def _supports(
    logits,
    targets,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    *,
    surface,
    **_kwargs,
):
    del surface
    return validate_cross_entropy(
        logits, targets, weight, ignore_index, reduction
    )


def _estimate(
    logits,
    targets,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    *,
    entrypoint,
    **_kwargs,
) -> CostHints:
    del logits, targets, weight, ignore_index, reduction, entrypoint, _kwargs
    return CostHints(
        notes=(
            "native-Torch physical traffic and workspace depend on compiler decomposition",
        )
    )


def _compute_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dtype == torch.float64 else tensor.float()


def forward(logits, targets, weight=None, ignore_index=-100):
    """Return native-PyTorch per-row ``(losses, logsumexp)``."""
    compute = _compute_tensor(logits)
    logsumexp = torch.logsumexp(compute, dim=-1)
    valid = targets != int(ignore_index)
    safe_targets = targets.masked_fill(~valid, 0).long()
    target_logits = compute.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    losses = logsumexp - target_logits
    if weight is not None:
        losses = losses * _compute_tensor(weight).gather(0, safe_targets)
    losses = torch.where(valid, losses, torch.zeros_like(losses))
    return losses, logsumexp


def backward(
    grad_losses,
    logits,
    targets,
    logsumexp,
    weight=None,
    ignore_index=-100,
):
    """Return the native-PyTorch logits VJP for per-row cotangents."""
    compute = _compute_tensor(logits)
    probability = torch.exp(compute - logsumexp.unsqueeze(1))
    valid = targets != int(ignore_index)
    safe_targets = targets.masked_fill(~valid, 0).long()
    one_hot = torch.zeros_like(probability).scatter(
        1, safe_targets.unsqueeze(1), 1.0
    )
    row_scale = _compute_tensor(grad_losses)
    if weight is not None:
        row_scale = row_scale * _compute_tensor(weight).gather(0, safe_targets)
    gradient = (probability - one_hot) * row_scale.unsqueeze(1)
    gradient = gradient * valid.unsqueeze(1)
    return gradient.to(logits.dtype)


def apply(
    logits,
    targets,
    weight=None,
    ignore_index=-100,
    reduction="mean",
):
    """Apply the differentiable native-PyTorch cross-entropy graph."""
    losses, _logsumexp = forward(logits, targets, weight, ignore_index)
    return reduce_losses(losses, targets, weight, ignore_index, reduction)


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="cross_entropy",
        implementation_id="native_torch.cross_entropy",
        provider="native_torch",
        priority=0,
        deterministic=True,
        supports=_supports,
        apply=apply,
        forward=forward,
        backward=backward,
        estimate=_estimate,
    )
)


__all__ = ["IMPLEMENTATION", "apply", "backward", "forward"]
