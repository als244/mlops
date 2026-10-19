"""Sparse latent attention whose positional tails affect keys but not values."""

from importlib.util import find_spec

import torch

from ...dispatch import attention_costs as costs
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...dsa import _validate_sparse_latent_rope
from .sparse_latent import _run_backward, _run_forward


@torch.library.custom_op("mlops::sparse_latent_rope_tilelang_fwd", mutates_args=())
def _forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_forward(q, kv, indices, scale, q_rope, k_rope)


@_forward.register_fake
def _fake(q, kv, q_rope, k_rope, indices, scale):
    return q.new_empty(q.shape), q.new_empty(q.shape[:2], dtype=torch.float32)


@torch.library.custom_op("mlops::sparse_latent_rope_tilelang_bwd", mutates_args=())
def _backward(
    dy: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dkv = _run_backward(dy, q, kv, indices, output, lse, scale, q_rope, k_rope)
    width = q.shape[-1]
    return (
        dq[..., :width].contiguous(),
        dkv[..., :width].contiguous(),
        dq[..., width:].contiguous(),
        dkv[..., width:].contiguous(),
    )


@_backward.register_fake
def _fake_bwd(dy, q, kv, q_rope, k_rope, indices, output, lse, scale):
    return tuple(x.new_empty(x.shape) for x in (q, kv, q_rope, k_rope))


def _setup(ctx, inputs, output):
    q, kv, q_rope, k_rope, indices, ctx.scale = inputs
    result, lse = output
    ctx.save_for_backward(q, kv, q_rope, k_rope, indices, result, lse)
    ctx.mark_non_differentiable(lse)


def _autograd(ctx, dy, _):
    return (*_backward(dy, *ctx.saved_tensors, ctx.scale), None, None)


_forward.register_autograd(_autograd, setup_context=_setup)


def _supports(q, kv, indices, *, scale, surface, q_rope=None, k_rope=None, **_kwargs):
    try:
        _validate_sparse_latent_rope(q, kv, indices, q_rope, k_rope)
    except ValueError as error:
        return SupportResult.no(str(error))
    if find_spec("tilelang") is None:
        return SupportResult.no("Sparse latent attention requires mlops[tilelang]")
    return SupportResult.yes()


def forward(q, kv, indices, *, scale, q_rope, k_rope):
    return _forward(
        q.contiguous(),
        kv.contiguous(),
        q_rope.contiguous(),
        k_rope.contiguous(),
        indices.contiguous(),
        float(scale),
    )


def backward(dy, q, kv, indices, output, lse, *, scale, q_rope, k_rope):
    return _backward(dy, q, kv, q_rope, k_rope, indices, output, lse, float(scale))


def apply(q, kv, indices, *, scale, q_rope, k_rope):
    return forward(q, kv, indices, scale=scale, q_rope=q_rope, k_rope=k_rope)[0]


@flop_formula(_forward)
def _forward_flops(q, kv, q_rope, k_rope, indices, scale, **_kwargs):
    return costs.sparse_latent_attention(
        q, kv, indices, scale=scale, q_rope=q_rope, k_rope=k_rope
    ).logical_flops


@flop_formula(_backward)
def _backward_flops(dy, q, kv, q_rope, k_rope, indices, output, lse, scale, **_kwargs):
    return costs.sparse_latent_attention(
        q,
        kv,
        indices,
        scale=scale,
        q_rope=q_rope,
        k_rope=k_rope,
        entrypoint="backward",
    ).logical_flops


IMPLEMENTATION = register_implementation(
    Implementation(
        "sparse_latent_attention",
        "builtin.sparse_latent_attention.rope.tilelang",
        "builtin",
        100,
        False,
        _supports,
        apply,
        forward,
        backward,
    )
)
