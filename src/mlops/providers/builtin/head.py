"""Package-owned chunked language-model head loss implementation."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.cross_entropy import cross_entropy_fwd_bwd
from ...kernels.head import default_head_chunk_size


def _common_support(hidden, head_weight, targets):
    if not all(
        isinstance(value, torch.Tensor)
        for value in (hidden, head_weight, targets)
    ):
        return SupportResult.no("head inputs must be tensors")
    if hidden.ndim < 1 or head_weight.ndim != 2:
        return SupportResult.no("hidden must have rows and head_weight must be rank two")
    if head_weight.shape[-1] != hidden.shape[-1]:
        return SupportResult.no("hidden and head widths are incompatible")
    if targets.numel() != hidden.numel() // hidden.shape[-1]:
        return SupportResult.no("targets must contain one label per hidden row")
    if hidden.device != head_weight.device or hidden.device != targets.device:
        return SupportResult.no("hidden, head_weight, and targets must share a device")
    return SupportResult.yes()


def _supports(
    hidden,
    head_weight,
    targets=None,
    *,
    surface,
    chunk_size=None,
    valid_rows=None,
    _entrypoint="forward",
    **_kwargs,
):
    del surface, chunk_size, valid_rows
    if _entrypoint == "backward":
        if not all(isinstance(value, torch.Tensor) for value in (hidden, head_weight)):
            return SupportResult.no("explicit backward seeds must be tensors")
        return SupportResult.yes()
    return _common_support(hidden, head_weight, targets)


def _policy(hidden, head_weight, chunk_size, valid_rows):
    rows = hidden.numel() // hidden.shape[-1]
    chunk = (
        default_head_chunk_size(head_weight.shape[0])
        if chunk_size is None
        else int(chunk_size)
    )
    normalizer = rows if valid_rows is None else int(valid_rows)
    if chunk <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk}")
    if not 0 < normalizer <= rows:
        raise ValueError(f"valid_rows must be in [1, {rows}]; got {normalizer}")
    return chunk, normalizer


def forward(
    hidden,
    head_weight,
    targets,
    *,
    chunk_size=None,
    valid_rows=None,
):
    """Return loss and seed-one hidden/head VJPs with bounded logits."""
    chunk, normalizer = _policy(hidden, head_weight, chunk_size, valid_rows)
    rows = hidden.numel() // hidden.shape[-1]
    with torch.no_grad():
        hidden_2d = hidden.reshape(rows, hidden.shape[-1])
        targets_1d = targets.reshape(rows)
        loss = torch.zeros((), dtype=torch.float32, device=hidden.device)
        grad_hidden = torch.empty_like(hidden_2d)
        grad_head = torch.zeros_like(head_weight)
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            hidden_chunk = hidden_2d[start:stop]
            logits = hidden_chunk @ head_weight.T
            partial, grad_logits = cross_entropy_fwd_bwd(
                logits,
                targets_1d[start:stop],
                total_rows=normalizer,
            )
            loss += partial
            grad_head.add_((grad_logits.T @ hidden_chunk).to(grad_head.dtype))
            grad_hidden[start:stop].copy_(grad_logits @ head_weight)
    return loss, grad_hidden.reshape_as(hidden), grad_head


def backward(grad_loss, grad_hidden_seed, grad_head_seed):
    """Scale immutable seed-one VJPs by an arbitrary scalar cotangent."""
    with torch.no_grad():
        return (
            grad_hidden_seed * grad_loss.to(grad_hidden_seed.dtype),
            grad_head_seed * grad_loss.to(grad_head_seed.dtype),
        )


@torch.library.custom_op(
    "mlops::head_loss_builtin_chunked_fwd",
    mutates_args=(),
)
def _forward_op(
    hidden: torch.Tensor,
    head_weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
    valid_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss, grad_hidden, grad_head = forward(
        hidden,
        head_weight,
        targets,
        chunk_size=chunk_size,
        valid_rows=valid_rows,
    )
    return loss, grad_hidden.reshape(-1, hidden.shape[-1]), grad_head


@_forward_op.register_fake
def _forward_fake(hidden, head_weight, targets, chunk_size, valid_rows):
    del targets, chunk_size, valid_rows
    rows = hidden.numel() // hidden.shape[-1]
    return (
        hidden.new_empty((), dtype=torch.float32),
        hidden.new_empty((rows, hidden.shape[-1])),
        torch.empty_like(head_weight),
    )


def _setup_context(ctx, inputs, output):
    hidden, _head_weight, _targets, *_policy_args = inputs
    _loss, grad_hidden, grad_head = output
    ctx.save_for_backward(grad_hidden, grad_head)
    ctx.hidden_shape = hidden.shape
    ctx.mark_non_differentiable(grad_hidden, grad_head)


def _autograd_backward(ctx, grad_loss, _grad_hidden_output, _grad_head_output):
    grad_hidden, grad_head = ctx.saved_tensors
    grad_hidden, grad_head = backward(grad_loss, grad_hidden, grad_head)
    return grad_hidden.reshape(ctx.hidden_shape), grad_head, None, None, None


_forward_op.register_autograd(
    _autograd_backward,
    setup_context=_setup_context,
)


def apply(
    hidden,
    head_weight,
    targets,
    *,
    chunk_size=None,
    valid_rows=None,
):
    """Apply the autograd-enabled bounded-logits head loss."""
    chunk, normalizer = _policy(hidden, head_weight, chunk_size, valid_rows)
    loss, *_seeds = _forward_op(
        hidden,
        head_weight,
        targets,
        chunk,
        normalizer,
    )
    return loss


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="head_loss",
        implementation_id="builtin.head_loss.chunked",
        provider="builtin",
        priority=100,
        deterministic=True,
        supports=_supports,
        apply=apply,
        forward=forward,
        backward=backward,
    )
)


__all__ = ["IMPLEMENTATION", "apply", "backward", "forward"]
