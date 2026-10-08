"""Chunked LoRA head projection, cross entropy, and requested seed VJPs."""

from __future__ import annotations

import math

import torch

from ...dispatch import logical_costs as logical
from ...dispatch.context import weight_gradient_dtype
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.cross_entropy import cross_entropy_fwd_bwd
from ...kernels.matmul import add_product_
from .head import _common_support, _policy


def _supports(
    hidden,
    head_weight,
    lora_a,
    lora_b,
    targets=None,
    *,
    surface,
    _entrypoint="forward",
    **_kwargs,
):
    del surface
    if _entrypoint == "backward":
        return SupportResult.yes()
    common = _common_support(hidden, head_weight, targets)
    if not common:
        return common
    if not all(isinstance(t, torch.Tensor) and t.ndim == 2 for t in (lora_a, lora_b)):
        return SupportResult.no("LoRA factors must be rank-two tensors")
    rank, width = lora_a.shape
    if rank <= 0 or width != hidden.shape[-1]:
        return SupportResult.no("lora_a must have positive rank and match hidden width")
    if lora_b.shape != (head_weight.shape[0], rank):
        return SupportResult.no("lora_b must have shape [vocabulary, rank]")
    if any(t.device != hidden.device for t in (lora_a, lora_b)):
        return SupportResult.no("LoRA factors must share the hidden device")
    if any(not t.is_floating_point() for t in (hidden, head_weight, lora_a, lora_b)):
        return SupportResult.no(
            "hidden, weight and LoRA factors must be floating point"
        )
    if head_weight.dtype != hidden.dtype:
        return SupportResult.no("hidden and head_weight must share a dtype")
    return SupportResult.yes()


def _scale(scale):
    scale = float(scale)
    if not -math.inf < scale < math.inf:
        raise ValueError(f"scale must be finite; got {scale}")
    return scale


def _run(
    hidden,
    head_weight,
    lora_a,
    lora_b,
    targets,
    scale,
    chunk,
    normalizer,
    weight_grad_dtype,
    need_hidden_grad,
    need_head_grad,
    need_lora_a_grad,
    need_lora_b_grad,
):
    rows = hidden.numel() // hidden.shape[-1]
    with torch.no_grad():
        x = hidden.reshape(rows, hidden.shape[-1])
        labels = targets.reshape(rows)
        loss = hidden.new_zeros((), dtype=torch.float32)
        dx = x.new_empty(x.shape) if need_hidden_grad else None
        dw = (
            torch.zeros_like(head_weight, dtype=weight_grad_dtype)
            if need_head_grad
            else None
        )
        da = (
            torch.zeros_like(lora_a, dtype=weight_grad_dtype)
            if need_lora_a_grad
            else None
        )
        db = (
            torch.zeros_like(lora_b, dtype=weight_grad_dtype)
            if need_lora_b_grad
            else None
        )
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            xc = x[start:stop]
            reduced = xc @ lora_a.T
            logits = xc @ head_weight.T
            logits.addmm_(reduced, lora_b.T, alpha=scale)
            partial, dz = cross_entropy_fwd_bwd(
                logits,
                labels[start:stop],
                total_rows=normalizer,
            )
            loss.add_(partial)
            if dw is not None:
                add_product_(dw, dz.T, xc)
            if db is not None:
                add_product_(db, dz.T, reduced, alpha=scale)
            if dx is not None or da is not None:
                dr = (dz @ lora_b).mul_(scale)
                if da is not None:
                    add_product_(da, dr.T, xc)
                if dx is not None:
                    dx[start:stop].copy_(dz @ head_weight)
                    dx[start:stop].addmm_(dr, lora_a)
        return loss, None if dx is None else dx.reshape_as(hidden), dw, da, db


def forward(
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
    weight_grad_dtype=None,
    need_hidden_grad=True,
    need_head_grad=False,
    need_lora_a_grad=True,
    need_lora_b_grad=True,
):
    """Return loss and requested unit-cotangent VJPs; no autograd state."""
    chunk, normalizer = _policy(hidden, head_weight, chunk_size, valid_rows, reduction)
    return _run(
        hidden,
        head_weight,
        lora_a.to(hidden.dtype),
        lora_b.to(hidden.dtype),
        targets,
        _scale(scale),
        chunk,
        normalizer,
        weight_grad_dtype,
        need_hidden_grad,
        need_head_grad,
        need_lora_a_grad,
        need_lora_b_grad,
    )


def backward(
    grad_loss, grad_hidden_seed, grad_head_seed, grad_lora_a_seed, grad_lora_b_seed
):
    """Scale seed VJPs out of place; omitted gradients remain None."""
    with torch.no_grad():
        return tuple(
            None if seed is None else seed * grad_loss.to(seed.dtype)
            for seed in (
                grad_hidden_seed,
                grad_head_seed,
                grad_lora_a_seed,
                grad_lora_b_seed,
            )
        )


@torch.library.custom_op(
    "mlops::lora_head_loss_builtin_chunked_fwd",
    mutates_args=(),
    schema=(
        "(Tensor hidden, Tensor head_weight, Tensor lora_a, Tensor lora_b, "
        "Tensor targets, float scale, SymInt chunk_size, SymInt normalizer, "
        "ScalarType? weight_grad_dtype, bool need_hidden_grad, bool need_head_grad, "
        "bool need_lora_a_grad, bool need_lora_b_grad) "
        "-> (Tensor, Tensor?, Tensor?, Tensor?, Tensor?)"
    ),
)
def _forward_op(
    hidden: torch.Tensor,
    head_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    targets: torch.Tensor,
    scale: float,
    chunk_size: int,
    normalizer: int,
    weight_grad_dtype: torch.dtype | None,
    need_hidden_grad: bool,
    need_head_grad: bool,
    need_lora_a_grad: bool,
    need_lora_b_grad: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    return _run(
        hidden,
        head_weight,
        lora_a,
        lora_b,
        targets,
        scale,
        chunk_size,
        normalizer,
        weight_grad_dtype,
        need_hidden_grad,
        need_head_grad,
        need_lora_a_grad,
        need_lora_b_grad,
    )


@_forward_op.register_fake
def _forward_fake(
    hidden,
    head_weight,
    lora_a,
    lora_b,
    targets,
    scale,
    chunk_size,
    normalizer,
    weight_grad_dtype,
    need_hidden_grad,
    need_head_grad,
    need_lora_a_grad,
    need_lora_b_grad,
):
    del targets, scale, chunk_size, normalizer
    return (
        hidden.new_empty((), dtype=torch.float32),
        hidden.new_empty(hidden.shape) if need_hidden_grad else None,
        torch.empty_like(head_weight, dtype=weight_grad_dtype)
        if need_head_grad
        else None,
        torch.empty_like(lora_a, dtype=weight_grad_dtype) if need_lora_a_grad else None,
        torch.empty_like(lora_b, dtype=weight_grad_dtype) if need_lora_b_grad else None,
    )


@flop_formula(_forward_op)
def _forward_flops(
    hidden,
    head_weight,
    lora_a,
    lora_b,
    targets,
    scale,
    chunk_size,
    normalizer,
    weight_grad_dtype,
    need_hidden_grad,
    need_head_grad,
    need_lora_a_grad,
    need_lora_b_grad,
    *,
    out_val=None,
    **_kwargs,
):
    del scale, chunk_size, normalizer, out_val
    return logical.lora_head_loss(
        hidden,
        head_weight,
        lora_a,
        lora_b,
        targets,
        weight_grad_dtype=weight_grad_dtype,
        need_hidden_grad=need_hidden_grad,
        need_head_grad=need_head_grad,
        need_lora_a_grad=need_lora_a_grad,
        need_lora_b_grad=need_lora_b_grad,
    ).logical_flops


def _setup_context(ctx, inputs, output):
    del inputs
    seeds = output[1:]
    ctx.save_for_backward(*seeds)
    ctx.mark_non_differentiable(*(seed for seed in seeds if seed is not None))


def _autograd_backward(ctx, grad_loss, *_grad_seed_outputs):
    gradients = backward(grad_loss, *ctx.saved_tensors)
    return (*gradients, *((None,) * 9))


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)


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
    chunk, normalizer = _policy(hidden, head_weight, chunk_size, valid_rows, reduction)
    factors = lora_a.to(hidden.dtype), lora_b.to(hidden.dtype)
    needs = tuple(
        torch.is_grad_enabled() and t.requires_grad
        for t in (hidden, head_weight, *factors)
    )
    loss, *_seeds = _forward_op(
        hidden,
        head_weight,
        *factors,
        targets,
        _scale(scale),
        chunk,
        normalizer,
        weight_gradient_dtype(),
        *needs,
    )
    return loss


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="lora_head_loss",
        implementation_id="builtin.lora_head_loss.chunked",
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
