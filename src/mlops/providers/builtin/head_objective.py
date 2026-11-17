"""One projection per chunk, with objective-specific unit gradients saved."""

from __future__ import annotations

from functools import cache
from importlib import import_module

import torch
from torch.nn import functional as F

from ...dispatch import logical_costs as logical
from ...dispatch.context import weight_gradient_dtype
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.cross_entropy import (
    cross_entropy_backward_triton,
    cross_entropy_forward_triton,
)
from ...kernels.head import default_head_chunk_size
from ...kernels.matmul import add_product_
from ...kernels.objectives.head import fused_head_objective
from ...objectives._head import BUILTINS, evaluate, validate
from .head import _common_support
from .lora_head import _scale


@cache
def objective_function(name):
    module, separator, attribute = name.partition(":")
    if not separator or not module or not attribute.isidentifier():
        raise ValueError("head objective must name an installed module:function")
    function = getattr(import_module(module), attribute)
    if not callable(function):
        raise TypeError("head objective must be callable")
    return function


@torch.compiler.assume_constant_result
def _objective_name(function):
    # Only the installed callable identity is static. Row data and controls
    # remain ordinary tensor arguments to the captured custom operation.
    if isinstance(function, str):
        if function not in BUILTINS:
            raise ValueError(f"unknown built-in head objective: {function}")
        return function
    if not callable(function) or getattr(function, "__qualname__", None) != getattr(
        function, "__name__", ""
    ):
        raise ValueError("head objective must be an installed module-level function")
    name = f"{function.__module__}:{function.__name__}"
    if objective_function(name) is not function:
        raise ValueError("head objective differs from its installed definition")
    return name


def _supports(hidden, head_weight, targets, *, surface, **_kwargs):
    if surface == "explicit":
        return SupportResult.no(
            "head objective exposes its saved-gradient autograd path"
        )
    common = _common_support(hidden, head_weight, targets)
    if not common:
        return common
    if not hidden.is_floating_point() or hidden.dtype != head_weight.dtype:
        return SupportResult.no("hidden and head_weight must share a floating dtype")
    if targets.dtype not in (torch.int32, torch.int64):
        return SupportResult.no("targets must be int32 or int64")
    return common


def _policy(hidden, head_weight, targets, inputs, controls, lora_a, lora_b, chunk_size):
    chunk = (
        default_head_chunk_size(head_weight.shape[0])
        if chunk_size is None
        else chunk_size
    )
    if type(chunk) is not int or chunk <= 0:
        raise ValueError("head chunk_size must be a positive integer")
    for value in inputs:
        if value.shape != targets.shape:
            raise ValueError("head row inputs must have the target shape")
    for value in controls:
        if value.ndim != 0:
            raise ValueError("head controls must be scalar tensors")
    for value in (*inputs, *controls):
        if value.device != hidden.device or value.requires_grad:
            raise ValueError(
                "head inputs/controls must share its device and not require gradients"
            )
    if (lora_a is None) != (lora_b is None):
        raise ValueError("head LoRA requires both factors")
    if lora_a is not None and (
        lora_a.ndim != 2
        or lora_b.ndim != 2
        or lora_a.shape[0] <= 0
        or lora_a.shape[1] != hidden.shape[-1]
        or lora_b.shape != (head_weight.shape[0], lora_a.shape[0])
    ):
        raise ValueError("head LoRA factor shapes differ from the projection")
    for factor in (lora_a, lora_b):
        if factor is not None and (
            factor.device != hidden.device or not factor.is_floating_point()
        ):
            raise ValueError("head LoRA factors must be floating tensors on its device")
    return chunk


def _logprobs(logits, targets):
    if logits.is_cuda:
        losses, lse = cross_entropy_forward_triton(logits, targets)
    else:
        values = logits.float()
        losses = F.cross_entropy(values, targets, reduction="none")
        lse = values.logsumexp(-1)
    return -losses, lse


def _logits_gradient(logits, targets, lse, gradient):
    if logits.is_cuda:
        return cross_entropy_backward_triton(-gradient, logits, targets, lse)
    probability = (logits.float() - lse[:, None]).exp()
    valid = targets != -100
    probability.scatter_add_(
        1, targets.clamp_min(0)[:, None], -valid[:, None].to(probability.dtype)
    )
    return (-probability * torch.where(valid, gradient, 0.0)[:, None]).to(logits.dtype)


def _run(
    hidden,
    weight,
    targets,
    a,
    b,
    scale,
    objective,
    inputs,
    controls,
    chunk,
    grad_dtype,
    needs,
):
    rows, width = targets.numel(), hidden.shape[-1]
    x, labels = hidden.reshape(rows, width), targets.reshape(rows).long()
    fields = [value.reshape(rows) for value in inputs]
    loss = hidden.new_zeros((), dtype=torch.float32)
    logprobs = hidden.new_empty((rows,), dtype=torch.float32)
    tensors = (hidden, weight, a, b)
    seeds = [
        (
            value.new_empty(value.shape)
            if index == 0
            else torch.zeros_like(value, dtype=grad_dtype)
        )
        if needed
        else None
        for index, (value, needed) in enumerate(zip(tensors, needs, strict=True))
    ]
    dx, dw, da, db = seeds
    if dx is not None:
        dx = dx.reshape(rows, width)
    training = any(needs)
    builtin = objective in BUILTINS
    fused = builtin and hidden.is_cuda
    function = None if builtin else objective_function(objective)
    row_losses = torch.empty_like(logprobs) if fused else None
    with torch.no_grad():
        for start in range(0, rows, chunk):
            stop = min(rows, start + chunk)
            part = x[start:stop]
            logits = part @ weight.T
            reduced = None
            if a is not None:
                reduced = part @ a.T
                logits.add_((reduced @ b.T) * scale)
            row_inputs = [value[start:stop] for value in fields]
            if fused:
                dz = fused_head_objective(
                    logits,
                    labels[start:stop],
                    row_inputs,
                    controls,
                    objective=objective,
                    need_gradient=training,
                    logprobs=logprobs[start:stop],
                    losses=row_losses[start:stop],
                )
            else:
                current, lse = _logprobs(logits, labels[start:stop])
                partial, gradient = (
                    evaluate(objective, current, row_inputs, controls)
                    if builtin
                    else function(current, row_inputs, controls)
                )
                if partial.ndim != 0 or gradient.shape != current.shape:
                    raise ValueError(
                        "head objective must return a scalar sum and per-row derivative"
                    )
                loss.add_(partial)
                logprobs[start:stop].copy_(current)
                dz = (
                    _logits_gradient(logits, labels[start:stop], lse, gradient)
                    if training
                    else None
                )
            if training:
                if dw is not None:
                    add_product_(dw, dz.T, part)
                if dx is not None:
                    dx[start:stop].copy_(dz @ weight)
                if a is not None:
                    # Ordinary LoRA scales the output in activation precision.
                    # Apply that same cast before the two factor VJPs.
                    du = dz * scale
                    if db is not None:
                        add_product_(db, du.T, reduced)
                    if da is not None or dx is not None:
                        dr = du @ b
                        if da is not None:
                            add_product_(da, dr.T, part)
                        if dx is not None:
                            dx[start:stop].add_(dr @ a)
                    del du
            # Do not keep the preceding vocabulary-sized buffers alive while
            # allocating the next chunk. No logits survive this forward.
            del logits, dz
    if row_losses is not None:
        loss = row_losses.sum()
    return loss, logprobs.reshape_as(targets), *seeds


@torch.library.custom_op(
    "mlops::head_objective_builtin_chunked_fwd",
    mutates_args=(),
    schema=(
        "(Tensor hidden, Tensor weight, Tensor targets, Tensor? a, Tensor? b, "
        "float scale, str objective, Tensor[] inputs, Tensor[] controls, "
        "SymInt chunk, ScalarType? grad_dtype, bool[] needs) "
        "-> (Tensor, Tensor, Tensor?, Tensor?, Tensor?, Tensor?)"
    ),
)
def _forward_op(
    hidden,
    weight,
    targets,
    a,
    b,
    scale,
    objective,
    inputs,
    controls,
    chunk,
    grad_dtype,
    needs,
):
    return _run(
        hidden,
        weight,
        targets,
        a,
        b,
        scale,
        objective,
        inputs,
        controls,
        chunk,
        grad_dtype,
        needs,
    )


@_forward_op.register_fake
def _forward_fake(
    hidden,
    weight,
    targets,
    a,
    b,
    scale,
    objective,
    inputs,
    controls,
    chunk,
    grad_dtype,
    needs,
):
    return (
        hidden.new_empty((), dtype=torch.float32),
        targets.new_empty(targets.shape, dtype=torch.float32),
        *(
            (
                value.new_empty(value.shape)
                if index == 0
                else torch.empty_like(value, dtype=grad_dtype)
            )
            if needed
            else None
            for index, (value, needed) in enumerate(
                zip((hidden, weight, a, b), needs, strict=True)
            )
        ),
    )


def _setup_context(ctx, inputs, output):
    _loss, logprobs, *seeds = output
    ctx.save_for_backward(*seeds)
    ctx.input_count, ctx.control_count = len(inputs[7]), len(inputs[8])
    ctx.mark_non_differentiable(logprobs, *(seed for seed in seeds if seed is not None))
    # These auxiliary outputs have no cotangents. The default autograd
    # materialization would allocate head-sized zero buffers only to discard
    # them in backward.
    ctx.set_materialize_grads(False)


@flop_formula(_forward_op)
def _forward_flops(
    hidden,
    weight,
    targets,
    a,
    b,
    scale,
    objective,
    inputs,
    controls,
    chunk,
    grad_dtype,
    needs,
    out_val=None,
    **_kwargs,
):
    return logical.head_objective(
        hidden,
        weight,
        targets,
        lora_a=a,
        lora_b=b,
        need_hidden_grad=needs[0],
        need_head_grad=needs[1],
        need_lora_a_grad=needs[2],
        need_lora_b_grad=needs[3],
        weight_grad_dtype=grad_dtype,
    ).logical_flops


def _backward(ctx, grad_loss, _grad_logprobs, *_grad_seeds):
    dx, dw, da, db = (
        None if seed is None or grad_loss is None else seed * grad_loss.to(seed.dtype)
        for seed in ctx.saved_tensors
    )
    return (
        dx,
        dw,
        None,
        da,
        db,
        None,
        None,
        [None] * ctx.input_count,
        [None] * ctx.control_count,
        None,
        None,
        None,
    )


_forward_op.register_autograd(_backward, setup_context=_setup_context)


def apply(
    hidden,
    head_weight,
    targets,
    *,
    objective,
    inputs=(),
    controls=(),
    lora_a=None,
    lora_b=None,
    scale=1.0,
    chunk_size=None,
):
    objective = _objective_name(objective)
    if objective in BUILTINS:
        validate(objective, inputs, controls)
    chunk = _policy(
        hidden, head_weight, targets, inputs, controls, lora_a, lora_b, chunk_size
    )
    a = None if lora_a is None else lora_a.to(hidden.dtype)
    b = None if lora_b is None else lora_b.to(hidden.dtype)
    needs = [
        torch.is_grad_enabled() and value is not None and value.requires_grad
        for value in (hidden, head_weight, a, b)
    ]
    loss, logprobs, *_seeds = _forward_op(
        hidden,
        head_weight,
        targets,
        a,
        b,
        _scale(scale),
        objective,
        list(inputs),
        list(controls),
        chunk,
        weight_gradient_dtype(),
        needs,
    )
    return loss, logprobs


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="head_objective",
        implementation_id="builtin.head_objective.chunked",
        provider="builtin",
        priority=100,
        deterministic=True,
        supports=_supports,
        apply=apply,
    )
)
