"""TileLang sparse attention with shared latent KV and a combined KV gradient."""

from importlib.util import find_spec

import torch
from torch.nn import functional as F

from ...dispatch import attention_costs as costs
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...dsa import _validate_sparse_latent


def _physical(q, kv, indices, q_rope=None, k_rope=None):
    # The reused upstream kernels have separate value and RoPE-tail operands.
    # A zero tail preserves the non-positional latent attention math; it is temporary kernel workspace.
    query = F.pad(q, (0, 64)) if q_rope is None else torch.cat((q, q_rope), -1)
    values = F.pad(kv, (0, 64)) if k_rope is None else torch.cat((kv, k_rope), -1)
    query = query.unsqueeze(0).contiguous()
    values = values.unsqueeze(0).unsqueeze(2).contiguous()
    ids = F.pad(indices, (0, (-indices.shape[-1]) % 32), value=-1)
    ids = torch.where(ids >= 0, ids, q.shape[0]).to(torch.int32)
    return query, values, ids[None, :, None, :].contiguous()


@torch.library.custom_op("mlops::sparse_latent_attention_tilelang_fwd", mutates_args=())
def _forward(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_forward(q, kv, indices, scale)


def _run_forward(q, kv, indices, scale, q_rope=None, k_rope=None):
    from ...kernels.sparse_mla.forward import sparse_mla_fwd

    query, values, ids = _physical(q, kv, indices, q_rope, k_rope)
    output, lse = sparse_mla_fwd(
        query,
        values,
        ids,
        q.shape[1],
        q.shape[-1],
        64,
        ids.shape[-1],
        sm_scale=scale,
        block_I=32,
        threads=128,
        head_tile=16,
    )
    return output.squeeze(0), lse.squeeze(0)


@_forward.register_fake
def _fake(q, kv, indices, scale):
    return torch.empty_like(q), q.new_empty(q.shape[:2], dtype=torch.float32)


@torch.library.custom_op("mlops::sparse_latent_attention_tilelang_bwd", mutates_args=())
def _backward(
    dy: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dq, dkv = _run_backward(dy, q, kv, indices, output, lse, scale)
    return dq[..., : q.shape[-1]].contiguous(), dkv[..., : kv.shape[-1]].contiguous()


def _run_backward(dy, q, kv, indices, output, lse, scale, q_rope=None, k_rope=None):
    from ...kernels.sparse_mla.backward import bwd, postprocess, preprocess

    query, values, ids = _physical(q, kv, indices, q_rope, k_rope)
    delta = preprocess(output.unsqueeze(0), dy.contiguous().unsqueeze(0))
    # Contributions from different queries/heads share KV; accumulate in FP32.
    dkv = torch.zeros_like(values, dtype=torch.float32)
    dq = bwd(
        query,
        values,
        dy.contiguous().unsqueeze(0),
        ids,
        lse.unsqueeze(0),
        delta,
        dkv,
        q.shape[1],
        q.shape[-1],
        64,
        ids.shape[-1],
        sm_scale=scale,
        head_tile=16,
    )
    dkv = postprocess(dkv, q.shape[-1], 64)
    return dq[0], dkv[0, :, 0]


@_backward.register_fake
def _fake_bwd(dy, q, kv, indices, output, lse, scale):
    return torch.empty_like(q), torch.empty_like(kv)


def _setup(ctx, inputs, output):
    q, kv, indices, ctx.scale = inputs
    result, lse = output
    ctx.save_for_backward(q, kv, indices, result, lse)
    ctx.mark_non_differentiable(lse)


def _autograd(ctx, dy, _):
    return (*_backward(dy, *ctx.saved_tensors, ctx.scale), None, None)


_forward.register_autograd(_autograd, setup_context=_setup)


def _supports(q, kv, indices, *, scale, surface, q_rope=None, k_rope=None, **_kwargs):
    if q_rope is not None or k_rope is not None:
        return SupportResult.no("Use the positional-tail implementation for RoPE")
    try:
        _validate_sparse_latent(q, kv, indices)
    except ValueError as error:
        return SupportResult.no(str(error))
    if find_spec("tilelang") is None:
        return SupportResult.no("Sparse latent attention requires mlops[tilelang]")
    return SupportResult.yes()


def forward(q, kv, indices, *, scale):
    return _forward(q.contiguous(), kv.contiguous(), indices.contiguous(), float(scale))


def backward(dy, q, kv, indices, output, lse, *, scale):
    return _backward(dy, q, kv, indices, output, lse, float(scale))


def apply(q, kv, indices, *, scale):
    return forward(q, kv, indices, scale=scale)[0]


@flop_formula(_forward)
def _forward_flops(q, kv, indices, scale, **_kwargs):
    return costs.sparse_latent_attention(q, kv, indices, scale=scale).logical_flops


@flop_formula(_backward)
def _backward_flops(dy, q, kv, indices, output, lse, scale, **_kwargs):
    return costs.sparse_latent_attention(
        q, kv, indices, scale=scale, entrypoint="backward"
    ).logical_flops


IMPLEMENTATION = register_implementation(
    Implementation(
        "sparse_latent_attention",
        "builtin.sparse_latent_attention.tilelang",
        "builtin",
        100,
        False,
        _supports,
        apply,
        forward,
        backward,
    )
)
