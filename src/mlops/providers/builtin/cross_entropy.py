"""Builtin Triton cross entropy and opaque PyTorch training boundary."""

from __future__ import annotations

import torch

from ...dispatch.costs import CostHints, register_operation_estimator
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.cross_entropy import (
    cross_entropy_backward_triton,
    cross_entropy_forward_triton,
    standalone_cross_entropy_available,
)
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
) -> SupportResult:
    del surface
    common = validate_cross_entropy(
        logits, targets, weight, ignore_index, reduction
    )
    if not common:
        return common
    if weight is not None:
        return SupportResult.no("builtin Triton cross entropy has no class weights")
    if not logits.is_cuda:
        return SupportResult.no("requires CUDA logits and targets")
    if logits.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return SupportResult.no("requires float16, bfloat16, or float32 logits")
    if not standalone_cross_entropy_available():
        return SupportResult.no("Triton is unavailable")
    return SupportResult.yes()


def _logical_cost(
    logits,
    targets,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    *,
    entrypoint,
    **_kwargs,
) -> CostHints:
    del ignore_index, reduction, _kwargs
    rows, vocabulary = logits.shape
    elements = logits.numel()
    logits_bytes = elements * logits.element_size()
    targets_bytes = targets.numel() * targets.element_size()
    weight_bytes = 0 if weight is None else weight.numel() * weight.element_size()
    if entrypoint == "forward":
        return CostHints(
            logical_flops=4 * elements + 2 * rows,
            logical_bytes_accessed=(
                logits_bytes + targets_bytes + weight_bytes + 4 * rows
            ),
            notes=(
                f"logical estimate assumes {rows} rows and vocabulary {vocabulary}",
                "comparisons and max reductions are excluded from the FLOP count",
            ),
        )
    return CostHints(
        logical_flops=4 * elements,
        logical_bytes_accessed=(
            2 * logits_bytes + targets_bytes + weight_bytes + 8 * rows
        ),
        notes=(
            "backward bytes include logits, per-row cotangent/LSE, and logits VJP",
        ),
    )


register_operation_estimator("cross_entropy", _logical_cost)


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
    del weight, ignore_index, reduction, _kwargs
    rows = logits.shape[0]
    elements = logits.numel()
    logits_bytes = elements * logits.element_size()
    targets_bytes = targets.numel() * targets.element_size()
    layout_workspace = (
        (0 if logits.is_contiguous() else logits_bytes)
        + (0 if targets.is_contiguous() else targets_bytes)
    )
    if entrypoint == "forward":
        return CostHints(
            implementation_flops=4 * elements + 2 * rows,
            implementation_bytes_accessed=logits_bytes + targets_bytes + 8 * rows,
            workspace_bytes=layout_workspace,
            notes=(
                "losses and logsumexp are outputs/residuals, not workspace",
            ),
        )
    return CostHints(
        implementation_flops=4 * elements,
        implementation_bytes_accessed=(
            2 * logits_bytes + targets_bytes + 8 * rows
        ),
        workspace_bytes=layout_workspace,
        notes=("source-level traffic excludes allocator and cache effects",),
    )


def forward(logits, targets, weight=None, ignore_index=-100):
    """Return builtin Triton per-row ``(losses, logsumexp)``."""
    if weight is not None:
        raise ValueError("builtin Triton cross entropy has no class weights")
    return cross_entropy_forward_triton(logits, targets, int(ignore_index))


def backward(
    grad_losses,
    logits,
    targets,
    logsumexp,
    weight=None,
    ignore_index=-100,
):
    """Return the builtin Triton logits VJP for per-row cotangents."""
    if weight is not None:
        raise ValueError("builtin Triton cross entropy has no class weights")
    return cross_entropy_backward_triton(
        grad_losses,
        logits,
        targets,
        logsumexp,
        int(ignore_index),
    )


@torch.library.custom_op(
    "mlops::cross_entropy_builtin_triton_fwd",
    mutates_args=(),
)
def _forward_op(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return forward(logits, targets, None, ignore_index)


@_forward_op.register_fake
def _forward_fake(logits, targets, ignore_index):
    del targets, ignore_index
    rows = logits.shape[0]
    losses = torch.empty(rows, dtype=torch.float32, device=logits.device)
    return losses, torch.empty_like(losses)


@torch.library.custom_op(
    "mlops::cross_entropy_builtin_triton_bwd",
    mutates_args=(),
)
def _backward_op(
    grad_losses: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    logsumexp: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    return backward(
        grad_losses,
        logits,
        targets,
        logsumexp,
        None,
        ignore_index,
    )


@_backward_op.register_fake
def _backward_fake(grad_losses, logits, targets, logsumexp, ignore_index):
    del grad_losses, targets, logsumexp, ignore_index
    return torch.empty_like(logits)


def _setup_context(ctx, inputs, output):
    logits, targets, ignore_index = inputs
    _losses, logsumexp = output
    ctx.save_for_backward(logits, targets, logsumexp)
    ctx.ignore_index = int(ignore_index)
    ctx.mark_non_differentiable(logsumexp)


def _autograd_backward(ctx, grad_losses, _grad_logsumexp):
    logits, targets, logsumexp = ctx.saved_tensors
    grad_logits = _backward_op(
        grad_losses,
        logits,
        targets,
        logsumexp,
        ctx.ignore_index,
    )
    return grad_logits, None, None


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def apply(
    logits,
    targets,
    weight=None,
    ignore_index=-100,
    reduction="mean",
):
    """Apply autograd-enabled builtin Triton cross entropy."""
    if weight is not None:
        raise ValueError("builtin Triton cross entropy has no class weights")
    losses, _logsumexp = _forward_op(logits, targets, int(ignore_index))
    return reduce_losses(losses, targets, None, ignore_index, reduction)


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="cross_entropy",
        implementation_id="builtin.cross_entropy.triton",
        provider="builtin",
        priority=100,
        deterministic=True,
        supports=_supports,
        apply=apply,
        forward=forward,
        backward=backward,
        estimate=_estimate,
    )
)


__all__ = ["IMPLEMENTATION", "apply", "backward", "forward"]
