"""Experimental routed + shared operator with explicit physical inputs and saved tensors."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import _configuration
from .operators import _checked, _fake_state
from .parameters.bf16 import BF16ComputeWeight
from .parameters.fp8 import QuackFP8Weight


@torch.library.custom_op("mlops_ep_quack_shared_v1::forward", mutates_args=())
def _forward(
    x: Tensor,
    p: Tensor,
    ids: Tensor,
    weights: list[Tensor],
    hist: Tensor,
    shared_weights: list[Tensor],
    handle: int,
    spec: str,
) -> tuple[Tensor, list[Tensor]]:
    return _checked(handle, spec).forward(
        x, p, ids, hist, *weights, shared_weights=shared_weights
    )


@_forward.register_fake
def _f_fake(x, p, ids, weights, hist, shared_weights, handle, spec):
    c = _configuration(spec)
    return torch.empty_like(x), _fake_state(x, c) + [
        x.new_empty((c.tokens_per_rank, c.shared_width)) for _ in range(3)
    ]


@torch.library.custom_op("mlops_ep_quack_shared_v1::backward", mutates_args=())
def _backward_op(
    dy: Tensor,
    weights: list[Tensor],
    state: list[Tensor],
    x: Tensor,
    shared_weights: list[Tensor],
    handle: int,
    spec: str,
) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:
    return _checked(handle, spec).backward(
        dy, *weights, state, x=x, shared_weights=shared_weights
    )


@_backward_op.register_fake
def _b_fake(dy, weights, state, x, shared_weights, handle, spec):
    c = _configuration(spec)
    return (
        torch.empty_like(dy),
        dy.new_empty((c.tokens_per_rank, c.top_k), dtype=torch.float32),
        [dy.new_empty(w.shape, dtype=c.weight_grad_dtype) for w in weights],
        [torch.empty_like(w) for w in shared_weights],
    )


@torch.library.custom_op("mlops_ep_quack_shared_v1::fp8_forward", mutates_args=())
def _fp8_forward(
    x: Tensor,
    p: Tensor,
    ids: Tensor,
    weights: list[Tensor],
    hist: Tensor,
    shared_weights: list[Tensor],
    handle: int,
    spec: str,
) -> tuple[Tensor, list[Tensor]]:
    return _checked(handle, spec).forward(
        x, p, ids, hist, weights[:4], weights[4:], shared_weights=shared_weights
    )


_fp8_forward.register_fake(_f_fake)


@torch.library.custom_op("mlops_ep_quack_shared_v1::fp8_backward", mutates_args=())
def _fp8_backward(
    dy: Tensor,
    weights: list[Tensor],
    state: list[Tensor],
    x: Tensor,
    shared_weights: list[Tensor],
    handle: int,
    spec: str,
) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:
    return _checked(handle, spec).backward(
        dy, weights[:4], weights[4:], state, x=x, shared_weights=shared_weights
    )


@_fp8_backward.register_fake
def _fp8_b_fake(dy, weights, state, x, shared_weights, handle, spec):
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
        [torch.empty_like(w) for w in shared_weights],
    )


def _fp8_dispatch(cls, func, types, args, kwargs):
    if kwargs:
        raise ValueError("Use positional operator arguments")
    if func is torch.ops.mlops_ep_quack_shared_v1.forward.default:
        x, p, ids, weights, hist, shared_weights, handle, spec = args
        components = [v for w in weights for v in w.components()]
        return _fp8_forward(x, p, ids, components, hist, shared_weights, handle, spec)
    dy, weights, state, x, shared_weights, handle, spec = args
    components = [v for w in weights for v in w.components()]
    return _fp8_backward(dy, components, state, x, shared_weights, handle, spec)


@torch.library.custom_op("mlops_ep_quack_shared_v1::bf16_forward", mutates_args=())
def _physical_forward(
    x: Tensor,
    p: Tensor,
    ids: Tensor,
    weights: list[Tensor],
    hist: Tensor,
    shared_weights: list[Tensor],
    handle: int,
    spec: str,
) -> tuple[Tensor, list[Tensor]]:
    return _checked(handle, spec).forward(
        x, p, ids, hist, *weights, shared_weights=shared_weights
    )


_physical_forward.register_fake(_f_fake)


@torch.library.custom_op("mlops_ep_quack_shared_v1::bf16_backward", mutates_args=())
def _physical_backward(
    dy: Tensor,
    weights: list[Tensor],
    state: list[Tensor],
    x: Tensor,
    shared_weights: list[Tensor],
    handle: int,
    spec: str,
) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:
    return _checked(handle, spec).backward(
        dy, *weights, state, x=x, shared_weights=shared_weights
    )


_physical_backward.register_fake(_b_fake)


def _dispatch(cls, func, types, args, kwargs):
    if kwargs:
        raise ValueError("Use positional operator arguments")
    if func is torch.ops.mlops_ep_quack_shared_v1.forward.default:
        x, p, ids, weights, hist, shared_weights, handle, spec = args
        return _physical_forward(
            x, p, ids, [w._data for w in weights], hist, shared_weights, handle, spec
        )
    dy, weights, state, x, shared_weights, handle, spec = args
    return _physical_backward(
        dy, [w._data for w in weights], state, x, shared_weights, handle, spec
    )


def _setup(ctx, inputs, output):
    x, _, _, weights, _, shared_weights, ctx.handle, ctx.spec = inputs
    _, state = output
    ctx.num_shared = len(shared_weights)
    ctx.save_for_backward(x, *weights, *shared_weights, *state)
    ctx.mark_non_differentiable(*state)
    ctx.set_materialize_grads(False)


def _autograd(ctx, dy, _state):
    saved = ctx.saved_tensors
    x, weights = saved[0], list(saved[1:3])
    shared_weights = list(saved[3 : 3 + ctx.num_shared])
    state = list(saved[3 + ctx.num_shared :])
    dx, dp, grads, shared_grads = _backward_op(
        dy.contiguous(), weights, state, x, shared_weights, ctx.handle, ctx.spec
    )
    return dx, dp, None, grads, None, shared_grads, None, None


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
            getattr(torch.ops.mlops_ep_quack_shared_v1, name).default,
            _EffectType.ORDERED,
        )
    )
