"""Sparse latent attention using adapted upstream TileLang training kernels.

Q is [T,H,L], KV is [T,L] shared across heads, and values equal latent keys.
The model supplies the attention scale before latent projection absorption.
Indices are unique causal keys per row, with -1 padding and at least one key.
"""

import torch
from torch.nn import functional as F


def _physical(q, kv, indices):
    # The reused upstream kernels have separate value and RoPE-tail operands.
    # A zero tail preserves GLM's NoPE math; it is temporary kernel workspace.
    query = F.pad(q, (0, 64)).unsqueeze(0).contiguous()
    values = F.pad(kv, (0, 64)).unsqueeze(0).unsqueeze(2).contiguous()
    ids = F.pad(indices, (0, (-indices.shape[-1]) % 32), value=-1)
    ids = torch.where(ids >= 0, ids, q.shape[0]).to(torch.int32)
    return query, values, ids[None, :, None, :].contiguous()


@torch.library.custom_op("mlops_glm::sparse_latent_forward", mutates_args=())
def _forward(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    from .kernels.sparse_mla_forward import sparse_mla_fwd

    query, values, ids = _physical(q, kv, indices)
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


@torch.library.custom_op("mlops_glm::sparse_latent_backward", mutates_args=())
def _backward(
    dy: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .kernels.sparse_mla_backward import bwd, postprocess, preprocess

    query, values, ids = _physical(q, kv, indices)
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
    return dq[0, :, :, : q.shape[-1]].contiguous(), dkv[
        0, :, 0, : kv.shape[-1]
    ].contiguous()


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


def sparse_latent_attention(q, kv, indices, *, scale):
    if q.ndim != 3 or kv.shape != (q.shape[0], q.shape[-1]):
        raise ValueError("Expected Q [tokens, heads, latent] and KV [tokens, latent]")
    if q.shape[1] % 16 or q.shape[-1] not in (64, 128, 256, 512):
        raise ValueError(
            "Expected heads divisible by 16 and latent width 64/128/256/512"
        )
    if (
        indices.ndim != 2
        or indices.shape[0] != q.shape[0]
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected integer selected indices [tokens, selected_keys]")
    if q.dtype != torch.bfloat16 or kv.dtype != q.dtype:
        raise ValueError("Sparse TileLang attention currently requires BF16 Q and KV")
    if not all(t.is_cuda and t.device == q.device for t in (q, kv, indices)):
        raise ValueError("Sparse attention inputs must share one CUDA device")
    return _forward(
        q.contiguous(), kv.contiguous(), indices.contiguous(), float(scale)
    )[0]
