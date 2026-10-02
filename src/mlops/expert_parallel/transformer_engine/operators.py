"""Autograd, FakeTensor shapes and ordered custom operators for AOT capture."""

from __future__ import annotations

import itertools
import math

import torch
from torch import Tensor

from .config import _config_from_signature, config_signature
from .registry import _runtime


def _checked_runtime(handle: int, spec: str):
    r = _runtime(handle)
    if config_signature(r.cfg) != spec:
        raise RuntimeError(
            "Compiled MoE signature does not match the runtime configuration"
        )
    return r


def _fake_state(x, c):
    P, q, E, G = c.dispatched_rows, c.local_experts, c.num_experts, c.ep_size
    shapes = [
        (c.tokens_per_rank * c.top_k,),
        (G, q),
        (E + q, 2),
        (2,),
        (P, 3),
        (P,),
        (2,),
    ]
    # MoonEP's vectorized planning writes use four-int32 padded allocations.
    # Preserve their physical extent as well as each output's logical shape.
    state = []
    for shape in shapes:
        count = math.prod(shape)
        storage = x.new_empty(((count + 3) // 4 * 4,), dtype=torch.int32)
        state.append(storage[:count].view(shape))
    state += [
        x.new_empty((2 * q,), dtype=torch.int64),
        x.new_empty((P, c.feature_dim)),
        x.new_empty((P,), dtype=torch.float32),
    ]
    state += [
        x.new_empty((P, c.expert_hidden_dim)),
        x.new_empty((P, c.expert_hidden_dim)),
        x.new_empty((P, c.feature_dim)),
    ]
    return state


def _split_weights(weights, c):
    if c.compute_precision == "bf16":
        if len(weights) != 3:
            raise ValueError("Expected three BF16 expert tensors")
        return tuple(weights)
    q = c.local_experts
    if len(weights) != 3 * q:
        raise ValueError("Expected three FP8 projections per local expert")
    return tuple(weights[index * q : (index + 1) * q] for index in range(3))


def _split_fp8_components(components, c):
    from mlops.expert_parallel.transformer_engine.parameters.components import (
        component_names,
    )

    n = len(component_names(c.compute_precision))
    if len(components) != 3 * c.local_experts * n:
        raise ValueError("Incomplete explicit FP8 component list")
    weights = [
        tuple(components[index : index + n]) for index in range(0, len(components), n)
    ]
    return _split_weights(weights, c)


@torch.library.custom_op("mlops_ep_te_v1::fp8_forward", mutates_args=())
def _fp8_forward_op(
    x: Tensor, p: Tensor, ids: Tensor, components: list[Tensor], handle: int, spec: str
) -> tuple[Tensor, list[Tensor]]:
    runtime = _checked_runtime(handle, spec)
    return runtime.forward(x, p, ids, *_split_fp8_components(components, runtime.cfg))


@_fp8_forward_op.register_fake
def _fp8_f_fake(x, p, ids, components, handle, spec):
    c = _config_from_signature(spec)
    _split_fp8_components(components, c)
    return torch.empty_like(x), _fake_state(x, c)


@torch.library.custom_op("mlops_ep_te_v1::fp8_backward", mutates_args=())
def _fp8_backward_op(
    dy: Tensor, components: list[Tensor], state: list[Tensor], handle: int, spec: str
) -> tuple[Tensor, Tensor, list[Tensor]]:
    runtime = _checked_runtime(handle, spec)
    params = _split_fp8_components(components, runtime.cfg)
    runtime._check_fp8_components(params)
    dx, dp, *grads = runtime.backward(dy, *params, state)
    return dx, dp, grads


@_fp8_backward_op.register_fake
def _fp8_b_fake(dy, components, state, handle, spec):
    c = _config_from_signature(spec)
    _split_fp8_components(components, c)
    shapes = [(c.local_experts, c.expert_hidden_dim, c.feature_dim)] * 2
    shapes += [(c.local_experts, c.feature_dim, c.expert_hidden_dim)]
    return (
        torch.empty_like(dy),
        dy.new_empty((c.tokens_per_rank, c.top_k), dtype=torch.float32),
        [dy.new_empty(shape, dtype=c.weight_grad_dtype) for shape in shapes],
    )


@torch.library.custom_op("mlops_ep_te_v1::bf16_forward", mutates_args=())
def _bf16_forward_op(
    x: Tensor, p: Tensor, ids: Tensor, components: list[Tensor], handle: int, spec: str
) -> tuple[Tensor, list[Tensor]]:
    runtime = _checked_runtime(handle, spec)
    return runtime.forward(x, p, ids, *_split_weights(components, runtime.cfg))


@_bf16_forward_op.register_fake
def _bf16_f_fake(x, p, ids, components, handle, spec):
    return torch.empty_like(x), _fake_state(x, _config_from_signature(spec))


@torch.library.custom_op("mlops_ep_te_v1::bf16_backward", mutates_args=())
def _bf16_backward_op(
    dy: Tensor, components: list[Tensor], state: list[Tensor], handle: int, spec: str
) -> tuple[Tensor, Tensor, list[Tensor]]:
    runtime = _checked_runtime(handle, spec)
    dx, dp, *grads = runtime.backward(
        dy, *_split_weights(components, runtime.cfg), state
    )
    return dx, dp, grads


@_bf16_backward_op.register_fake
def _bf16_b_fake(dy, components, state, handle, spec):
    c = _config_from_signature(spec)
    return (
        torch.empty_like(dy),
        dy.new_empty((c.tokens_per_rank, c.top_k), dtype=torch.float32),
        [dy.new_empty(w.shape, dtype=c.weight_grad_dtype) for w in components],
    )


@torch.library.custom_op("mlops_ep_te_v1::forward", mutates_args=())
def _forward_op(
    x: Tensor, p: Tensor, ids: Tensor, weights: list[Tensor], handle: int, spec: str
) -> tuple[Tensor, list[Tensor]]:
    runtime = _checked_runtime(handle, spec)
    return runtime.forward(x, p, ids, *_split_weights(weights, runtime.cfg))


@_forward_op.register_fake
def _f_fake(x, p, ids, weights, handle, spec):
    return torch.empty_like(x), _fake_state(x, _config_from_signature(spec))


@torch.library.custom_op("mlops_ep_te_v1::backward", mutates_args=())
def _backward_op(
    dy: Tensor, weights: list[Tensor], state: list[Tensor], handle: int, spec: str
) -> tuple[Tensor, Tensor, list[Tensor]]:
    runtime = _checked_runtime(handle, spec)
    dx, dp, *grads = runtime.backward(dy, *_split_weights(weights, runtime.cfg), state)
    if runtime.cfg.compute_precision != "bf16":
        grads = list(itertools.chain.from_iterable(grads))
    return dx, dp, grads


@_backward_op.register_fake
def _b_fake(dy, weights, state, handle, spec):
    c = _config_from_signature(spec)
    dtype = c.weight_grad_dtype
    return (
        torch.empty_like(dy),
        dy.new_empty((c.tokens_per_rank, c.top_k), dtype=torch.float32),
        [torch.empty(w.shape, device=dy.device, dtype=dtype) for w in weights],
    )


def _setup(ctx, inputs, output):
    _, _, _, weights, handle, spec = inputs
    _, state = output
    ctx.handle, ctx.spec, ctx.nweights = handle, spec, len(weights)
    ctx.save_for_backward(*weights, *state)
    ctx.mark_non_differentiable(*state)
    ctx.set_materialize_grads(False)


def _autograd_backward(ctx, dy, _aux):
    if dy is None:
        raise RuntimeError(
            "Every EP rank must participate in backward; use a zero loss, not a missing backward"
        )
    saved = ctx.saved_tensors
    weights, state = list(saved[: ctx.nweights]), list(saved[ctx.nweights :])
    dx, dp, grads = _backward_op(dy.contiguous(), weights, state, ctx.handle, ctx.spec)
    return dx, dp, None, grads, None, None


_forward_op.register_autograd(_autograd_backward, setup_context=_setup)
from mlops.expert_parallel.transformer_engine.parameters.bf16 import BF16ComputeWeight
from mlops.expert_parallel.transformer_engine.parameters.components import (
    FP8_TYPES,
    explicit_weight_components,
)


def _fp8_dispatch(cls, func, types, args, kwargs):
    # The logical op owns autograd for full floating-point weight gradients.
    # Its physical implementation exposes every quantized component to AOT.
    if kwargs:
        raise TypeError("FP8 logical operators use their positional schema")
    if func is torch.ops.mlops_ep_te_v1.forward.default:
        x, p, ids, weights, handle, spec = args
        c = _config_from_signature(spec)
        components = explicit_weight_components(weights, c.compute_precision)
        return _fp8_forward_op(x, p, ids, components, handle, spec)
    if func is torch.ops.mlops_ep_te_v1.backward.default:
        dy, weights, state, handle, spec = args
        c = _config_from_signature(spec)
        components = explicit_weight_components(weights, c.compute_precision)
        dx, dp, banks = _fp8_backward_op(dy, components, state, handle, spec)
        return dx, dp, [grad for bank in banks for grad in bank.unbind(0)]
    raise RuntimeError(f"Unexpected FP8 logical operator {func}")


for _op in (_forward_op, _backward_op):
    for _cls in FP8_TYPES:
        _op.register_torch_dispatch(_cls, _fp8_dispatch)


def _bf16_dispatch(cls, func, types, args, kwargs):
    if kwargs:
        raise TypeError("BF16 logical operators use their positional schema")
    if func is torch.ops.mlops_ep_te_v1.forward.default:
        x, p, ids, weights, handle, spec = args
        return _bf16_forward_op(x, p, ids, [w._data for w in weights], handle, spec)
    if func is torch.ops.mlops_ep_te_v1.backward.default:
        dy, weights, state, handle, spec = args
        return _bf16_backward_op(dy, [w._data for w in weights], state, handle, spec)
    raise RuntimeError(f"Unexpected BF16 logical operator {func}")


for _op in (_forward_op, _backward_op):
    _op.register_torch_dispatch(BF16ComputeWeight, _bf16_dispatch)

_EFFECT_ERROR = None
_EFFECT_HANDLES = []
try:
    from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

    _EFFECT_HANDLES += [
        _register_effectful_op(
            torch.ops.mlops_ep_te_v1.forward.default, _EffectType.ORDERED
        ),
        _register_effectful_op(
            torch.ops.mlops_ep_te_v1.backward.default, _EffectType.ORDERED
        ),
        _register_effectful_op(
            torch.ops.mlops_ep_te_v1.bf16_forward.default, _EffectType.ORDERED
        ),
        _register_effectful_op(
            torch.ops.mlops_ep_te_v1.bf16_backward.default, _EffectType.ORDERED
        ),
        _register_effectful_op(
            torch.ops.mlops_ep_te_v1.fp8_forward.default, _EffectType.ORDERED
        ),
        _register_effectful_op(
            torch.ops.mlops_ep_te_v1.fp8_backward.default, _EffectType.ORDERED
        ),
    ]
except (ImportError, AttributeError, TypeError) as exc:
    _EFFECT_ERROR = exc
