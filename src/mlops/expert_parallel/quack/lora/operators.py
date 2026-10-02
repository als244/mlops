"""Explicit LoRA parameters, saved tensors, and ordered AOT operator boundaries."""

import torch
from torch import Tensor

from ...lora import parse_signature, signature
from ...parameters import BF16ComputeWeight
from ..config import _configuration, config_signature
from ..operators import _fake_state as _base_fake_state
from ..parameters.components import COMPUTE_WEIGHT_TYPES
from ..registry import _runtime


def _checked(handle, spec):
    runtime = _runtime(handle)
    if signature(config_signature(runtime.cfg), runtime.lora) != spec:
        raise ValueError("Compiled LoRA signature does not match the runtime")
    return runtime


def _config(spec):
    base, lora = parse_signature(spec)
    return _configuration(base), lora


def _base_components(base, spec):
    c, _ = _config(spec)
    if c.compute_precision == "bf16":
        return [w._data if isinstance(w, BF16ComputeWeight) else w for w in base]
    return [component for w in base for component in w.components()]


def _factor_components(factors):
    return [w._data if isinstance(w, BF16ComputeWeight) else w for w in factors]


def _fake_state(x, spec, shared):
    c, _ = _config(spec)
    state = _base_fake_state(x, c)
    if shared:
        state += [x.new_empty((c.tokens_per_rank, c.shared_width)) for _ in range(3)]
    return state


@torch.library.custom_op("mlops_ep_quack_lora::physical_forward", mutates_args=())
def _physical_forward(
    x: Tensor,
    p: Tensor,
    ids: Tensor,
    base: list[Tensor],
    factors: list[Tensor],
    hist: Tensor,
    shared: list[Tensor],
    handle: int,
    spec: str,
) -> tuple[Tensor, list[Tensor]]:
    return _checked(handle, spec).lora_forward(x, p, ids, base, factors, hist, shared)


@_physical_forward.register_fake
def _f_fake(x, p, ids, base, factors, hist, shared, handle, spec):
    return torch.empty_like(x), _fake_state(x, spec, shared)


@torch.library.custom_op("mlops_ep_quack_lora::physical_backward", mutates_args=())
def _physical_backward(
    dy: Tensor,
    base: list[Tensor],
    factors: list[Tensor],
    state: list[Tensor],
    x: Tensor | None,
    shared: list[Tensor],
    handle: int,
    spec: str,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    return _checked(handle, spec).lora_backward(dy, base, factors, state, x, shared)


@_physical_backward.register_fake
def _b_fake(dy, base, factors, state, x, shared, handle, spec):
    c, lora = _config(spec)
    return (
        torch.empty_like(dy),
        dy.new_empty((c.tokens_per_rank, c.top_k), dtype=torch.float32),
        [dy.new_empty(w.shape, dtype=lora.gradient_dtype) for w in factors],
    )


@torch.library.custom_op("mlops_ep_quack_lora::forward", mutates_args=())
def forward(
    x: Tensor,
    p: Tensor,
    ids: Tensor,
    base: list[Tensor],
    factors: list[Tensor],
    hist: Tensor,
    shared: list[Tensor],
    handle: int,
    spec: str,
) -> tuple[Tensor, list[Tensor]]:
    return _physical_forward(
        x,
        p,
        ids,
        _base_components(base, spec),
        _factor_components(factors),
        hist,
        shared,
        handle,
        spec,
    )


forward.register_fake(_f_fake)


@torch.library.custom_op("mlops_ep_quack_lora::backward", mutates_args=())
def backward(
    dy: Tensor,
    base: list[Tensor],
    factors: list[Tensor],
    state: list[Tensor],
    x: Tensor | None,
    shared: list[Tensor],
    handle: int,
    spec: str,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    return _physical_backward(
        dy,
        _base_components(base, spec),
        _factor_components(factors),
        state,
        x,
        shared,
        handle,
        spec,
    )


backward.register_fake(_b_fake)


def _dispatch(cls, func, types, args, kwargs):
    if kwargs:
        raise TypeError("LoRA operators use positional arguments")
    if func is torch.ops.mlops_ep_quack_lora.forward.default:
        x, p, ids, base, factors, hist, shared, handle, spec = args
        return _physical_forward(
            x,
            p,
            ids,
            _base_components(base, spec),
            _factor_components(factors),
            hist,
            shared,
            handle,
            spec,
        )
    dy, base, factors, state, x, shared, handle, spec = args
    return _physical_backward(
        dy,
        _base_components(base, spec),
        _factor_components(factors),
        state,
        x,
        shared,
        handle,
        spec,
    )


def _setup(ctx, inputs, output):
    x, _, _, base, factors, _, shared, ctx.handle, ctx.spec = inputs
    ctx.nbase, ctx.nfactors, ctx.nshared = len(base), len(factors), len(shared)
    state = output[1]
    ctx.save_for_backward(*base, *factors, *shared, *([x] if shared else []), *state)
    ctx.mark_non_differentiable(*state)
    ctx.set_materialize_grads(False)


def _autograd(ctx, dy, _state):
    values = list(ctx.saved_tensors)
    base, values = values[: ctx.nbase], values[ctx.nbase :]
    factors, values = values[: ctx.nfactors], values[ctx.nfactors :]
    shared, values = values[: ctx.nshared], values[ctx.nshared :]
    x = values.pop(0) if shared else None
    if dy is None:
        return (
            None,
            None,
            None,
            [None] * ctx.nbase,
            [None] * ctx.nfactors,
            None,
            [None] * ctx.nshared,
            None,
            None,
        )
    dx, dp, grads = backward(
        dy.contiguous(), base, factors, values, x, shared, ctx.handle, ctx.spec
    )
    return (
        dx,
        dp,
        None,
        [None] * ctx.nbase,
        grads,
        None,
        [None] * ctx.nshared,
        None,
        None,
    )


forward.register_autograd(_autograd, setup_context=_setup)
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

_EFFECT_HANDLES = []
for op in (forward, backward):
    for cls in COMPUTE_WEIGHT_TYPES:
        op.register_torch_dispatch(cls, _dispatch)
for name in ("forward", "backward", "physical_forward", "physical_backward"):
    _EFFECT_HANDLES.append(
        _register_effectful_op(
            getattr(torch.ops.mlops_ep_quack_lora, name).default, _EffectType.ORDERED
        )
    )
