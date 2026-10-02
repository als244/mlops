"""Autograd, physical tensor inputs and ordered operators for AOT capture."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .config import _configuration, config_signature
from .parameters.bf16 import BF16ComputeWeight
from .parameters.fp8 import QuackFP8Weight
from .registry import _runtime


def _fake_plan_state(x, c):
    p, q, e, g = c.dispatched_rows, c.local_experts, c.num_experts, c.ep_size
    shapes = [
        (c.tokens_per_rank * c.top_k,),
        (g, q),
        (e + q, 2),
        (2,),
        (p, 3),
        (p,),
        (2,),
    ]
    # MoonEP vectorizes planning writes in groups of four int32 values. Its
    # logical outputs are views of these padded allocations; fake tensors must
    # expose that backing extent too, for compiler memory accounting.
    result = []
    for shape in shapes:
        count = math.prod(shape)
        storage = x.new_empty(((count + 3) // 4 * 4,), dtype=torch.int32)
        result.append(storage[:count].view(shape))
    return result


def _checked(handle, spec):
    runtime = _runtime(handle)
    if config_signature(runtime.cfg) != spec:
        raise ValueError("Compiled QuACK signature does not match runtime")
    return runtime


def _fake_state(x, c):
    from dataclasses import replace

    chunk = replace(c, tokens_per_rank=c.tokens_per_rank // c.num_chunks)
    return [value for _ in range(c.num_chunks) for value in _fake_chunk_state(x, chunk)]


def _fake_chunk_state(x, c):
    quantized = c.activation_transport == "fp8"
    state = _fake_plan_state(x, c) + [
        x.new_empty((2 * c.local_experts + 1,), dtype=torch.int32),
        x.new_empty(
            (c.dispatched_rows, c.feature_dim),
            dtype=torch.float8_e4m3fn if quantized else x.dtype,
        ),
        x.new_empty((c.dispatched_rows,), dtype=torch.float32),
        x.new_empty((c.dispatched_rows, 2 * c.expert_hidden_dim)),
    ]
    if quantized:
        state.append(x.new_empty((c.dispatched_rows,), dtype=torch.float32))
    return state


@torch.library.custom_op("mlops_ep_quack_v1::forward", mutates_args=())
def _forward(
    x: Tensor,
    p: Tensor,
    ids: Tensor,
    weights: list[Tensor],
    hist: Tensor,
    handle: int,
    spec: str,
) -> tuple[Tensor, list[Tensor]]:
    return _checked(handle, spec).forward(x, p, ids, hist, *weights)


@_forward.register_fake
def _f_fake(x, p, ids, weights, hist, handle, spec):
    return torch.empty_like(x), _fake_state(x, _configuration(spec))


@torch.library.custom_op("mlops_ep_quack_v1::backward", mutates_args=())
def _backward_op(
    dy: Tensor, weights: list[Tensor], state: list[Tensor], handle: int, spec: str
) -> tuple[Tensor, Tensor, list[Tensor]]:
    return _checked(handle, spec).backward(dy, *weights, state)


@_backward_op.register_fake
def _b_fake(dy, weights, state, handle, spec):
    c = _configuration(spec)
    return (
        torch.empty_like(dy),
        dy.new_empty((c.tokens_per_rank, c.top_k), dtype=torch.float32),
        [dy.new_empty(w.shape, dtype=c.weight_grad_dtype) for w in weights],
    )


@torch.library.custom_op("mlops_ep_quack_v1::fp8_forward", mutates_args=())
def _fp8_forward(
    x: Tensor,
    p: Tensor,
    ids: Tensor,
    weights: list[Tensor],
    hist: Tensor,
    handle: int,
    spec: str,
) -> tuple[Tensor, list[Tensor]]:
    return _checked(handle, spec).forward(x, p, ids, hist, weights[:4], weights[4:])


_fp8_forward.register_fake(_f_fake)


@torch.library.custom_op("mlops_ep_quack_v1::fp8_backward", mutates_args=())
def _fp8_backward(
    dy: Tensor, weights: list[Tensor], state: list[Tensor], handle: int, spec: str
) -> tuple[Tensor, Tensor, list[Tensor]]:
    return _checked(handle, spec).backward(dy, weights[:4], weights[4:], state)


@_fp8_backward.register_fake
def _fp8_b_fake(dy, weights, state, handle, spec):
    c = _configuration(spec)
    return (
        torch.empty_like(dy),
        dy.new_empty((c.tokens_per_rank, c.top_k), dtype=torch.float32),
        [
            dy.new_empty(shape, dtype=c.weight_grad_dtype)
            for shape in (
                (c.local_experts, 2 * c.expert_hidden_dim, c.feature_dim),
                (c.local_experts, c.feature_dim, c.expert_hidden_dim),
            )
        ],
    )


def _fp8_dispatch(cls, func, types, args, kwargs):
    if kwargs:
        raise ValueError("Use positional operator arguments")
    if func is torch.ops.mlops_ep_quack_v1.forward.default:
        x, p, ids, weights, hist, handle, spec = args
        components = [v for w in weights for v in w.components()]
        return _fp8_forward(x, p, ids, components, hist, handle, spec)
    dy, weights, state, handle, spec = args
    components = [v for w in weights for v in w.components()]
    return _fp8_backward(dy, components, state, handle, spec)


@torch.library.custom_op("mlops_ep_quack_v1::bf16_forward", mutates_args=())
def _physical_forward(
    x: Tensor,
    p: Tensor,
    ids: Tensor,
    weights: list[Tensor],
    hist: Tensor,
    handle: int,
    spec: str,
) -> tuple[Tensor, list[Tensor]]:
    return _checked(handle, spec).forward(x, p, ids, hist, *weights)


_physical_forward.register_fake(_f_fake)


@torch.library.custom_op("mlops_ep_quack_v1::bf16_backward", mutates_args=())
def _physical_backward(
    dy: Tensor, weights: list[Tensor], state: list[Tensor], handle: int, spec: str
) -> tuple[Tensor, Tensor, list[Tensor]]:
    return _checked(handle, spec).backward(dy, *weights, state)


_physical_backward.register_fake(_b_fake)


def _dispatch(cls, func, types, args, kwargs):
    if kwargs:
        raise ValueError("Use positional operator arguments")
    if func is torch.ops.mlops_ep_quack_v1.forward.default:
        x, p, ids, weights, hist, handle, spec = args
        return _physical_forward(
            x, p, ids, [w._data for w in weights], hist, handle, spec
        )
    dy, weights, state, handle, spec = args
    return _physical_backward(dy, [w._data for w in weights], state, handle, spec)


def _setup(ctx, inputs, output):
    _, _, _, weights, _, ctx.handle, ctx.spec = inputs
    _, state = output
    ctx.save_for_backward(*weights, *state)
    ctx.mark_non_differentiable(*state)
    ctx.set_materialize_grads(False)


def _autograd(ctx, dy, _state):
    saved = ctx.saved_tensors
    weights = list(saved[:2])
    state = list(saved[2:])
    dx, dp, grads = _backward_op(dy.contiguous(), weights, state, ctx.handle, ctx.spec)
    return dx, dp, None, grads, None, None, None


_forward.register_autograd(_autograd, setup_context=_setup)
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

_EFFECT_HANDLES = []
for op in (_forward, _backward_op):
    op.register_torch_dispatch(BF16ComputeWeight, _dispatch)
    op.register_torch_dispatch(QuackFP8Weight, _fp8_dispatch)
for name in (
    "forward",
    "backward",
    "bf16_forward",
    "bf16_backward",
    "fp8_forward",
    "fp8_backward",
):
    _EFFECT_HANDLES.append(
        _register_effectful_op(
            getattr(torch.ops.mlops_ep_quack_v1, name).default, _EffectType.ORDERED
        )
    )
