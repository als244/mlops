"""Builtin sparse-attention and DSA-indexer implementations."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.dsa import sparse_attention_backward, sparse_attention_forward
from ...kernels.dsa_indexer import index_scores as raw_index_scores
from ...layer_norm import layer_norm


def _attention_support(q, k, v, indices, lengths, *, surface, reference_pad_value=False, **_kwargs):
    del surface, reference_pad_value
    if not all(isinstance(value, torch.Tensor) for value in (q, k, v, indices)):
        return SupportResult.no("q, k, v, and indices must be tensors")
    if not all(value.is_cuda for value in (q, k, v, indices)):
        return SupportResult.no("builtin sparse DSA requires CUDA tensors")
    if sum(int(length) for length in lengths) != q.shape[0]:
        return SupportResult.no("lengths must cover the query token axis")
    return SupportResult.yes()


def attention_forward(q, k, v, indices, lengths):
    with torch.no_grad():
        return sparse_attention_forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            indices.contiguous(),
            [int(length) for length in lengths],
        )


def attention_backward(grad_output, q, k, v, indices, lse, lengths):
    with torch.no_grad():
        return sparse_attention_backward(
            grad_output.contiguous(),
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            indices.contiguous(),
            lse.contiguous(),
            [int(length) for length in lengths],
        )


@torch.library.custom_op(
    "mlops::dsa_attention_builtin_sparse_fwd",
    mutates_args=(),
)
def _attention_forward_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    return attention_forward(q, k, v, indices, lengths)


@_attention_forward_op.register_fake
def _attention_forward_fake(q, k, v, indices, lengths):
    del k, indices, lengths
    return (
        torch.empty_like(v),
        torch.empty((q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device),
    )


@torch.library.custom_op(
    "mlops::dsa_attention_builtin_sparse_bwd",
    mutates_args=(),
)
def _attention_backward_op(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return attention_backward(grad_output, q, k, v, indices, lse, lengths)


@_attention_backward_op.register_fake
def _attention_backward_fake(grad_output, q, k, v, indices, lse, lengths):
    del grad_output, indices, lse, lengths
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_attention_context(ctx, inputs, output):
    q, k, v, indices, lengths = inputs
    _result, lse = output
    ctx.save_for_backward(q, k, v, indices, lse)
    ctx.lengths = tuple(lengths)
    ctx.mark_non_differentiable(lse)


def _attention_autograd_backward(ctx, grad_output, _grad_lse):
    q, k, v, indices, lse = ctx.saved_tensors
    grad_q, grad_k, grad_v = _attention_backward_op(
        grad_output,
        q,
        k,
        v,
        indices,
        lse,
        list(ctx.lengths),
    )
    return grad_q, grad_k, grad_v, None, None


_attention_forward_op.register_autograd(
    _attention_autograd_backward,
    setup_context=_setup_attention_context,
)


def attention_apply(q, k, v, indices, lengths, *, reference_pad_value=False):
    del reference_pad_value
    output, _lse = _attention_forward_op(
        q.contiguous(), k.contiguous(), v.contiguous(), indices.contiguous(),
        list(lengths),
    )
    return output


def _key_norm_support(x, weight, bias, _eps=1e-5, *, surface, **_kwargs):
    del surface
    if not all(isinstance(value, torch.Tensor) for value in (x, weight, bias)):
        return SupportResult.no("x, weight, and bias must be tensors")
    return SupportResult.yes()


def key_norm_apply(x, weight, bias, eps=1e-5):
    return layer_norm(x, weight, bias, float(eps))


def _score_support(q, k, weights, lengths, *, surface, **_kwargs):
    if surface == "explicit":
        return SupportResult.no("DSA index scoring is an ordinary PyTorch graph")
    if not all(isinstance(value, torch.Tensor) for value in (q, k, weights)):
        return SupportResult.no("q, k, and weights must be tensors")
    if sum(int(length) for length in lengths) != q.shape[0]:
        return SupportResult.no("lengths must cover the query token axis")
    return SupportResult.yes()


def score_apply(q, k, weights, lengths):
    return raw_index_scores(q, k, weights, lengths, reduction="einsum")


IMPLEMENTATIONS = tuple(
    register_implementation(implementation)
    for implementation in (
        Implementation(
            "dsa_attention", "builtin.dsa_attention.sparse", "builtin", 100, True,
            _attention_support, attention_apply, attention_forward,
            attention_backward,
        ),
        Implementation(
            operation="dsa_indexer_key_norm",
            implementation_id="builtin.dsa_indexer_key_norm.layer_norm",
            provider="builtin", priority=100, deterministic=True,
            supports=_key_norm_support, apply=key_norm_apply,
        ),
        Implementation(
            operation="index_scores",
            implementation_id="builtin.index_scores.einsum",
            provider="builtin", priority=100, deterministic=True,
            supports=_score_support, apply=score_apply,
        ),
    )
)

__all__ = ["IMPLEMENTATIONS"]
