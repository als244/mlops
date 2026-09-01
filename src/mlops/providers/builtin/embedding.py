"""Package-owned deterministic embedding implementation."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.embedding import embedding_backward


def _supports(tokens, weight, *, surface, **_kwargs):
    del surface
    if not isinstance(tokens, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return SupportResult.no("tokens and weight must be tensors")
    if weight.ndim != 2:
        return SupportResult.no("weight must have shape [vocabulary, width]")
    if tokens.device != weight.device:
        return SupportResult.no("tokens and weight must share a device")
    if tokens.dtype not in {torch.int32, torch.int64}:
        return SupportResult.no("tokens must use int32 or int64 indices")
    return SupportResult.yes()


def forward(tokens, weight):
    flat = torch.index_select(weight, 0, tokens.reshape(-1).long())
    return flat.reshape(*tokens.shape, weight.shape[1])


def backward(tokens, grad_output, num_embeddings):
    return embedding_backward(tokens, grad_output, int(num_embeddings))


@torch.library.custom_op(
    "mlops::embedding_builtin_deterministic_fwd",
    mutates_args=(),
)
def _forward_op(tokens: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return forward(tokens, weight)


@_forward_op.register_fake
def _forward_fake(tokens, weight):
    return weight.new_empty((*tokens.shape, weight.shape[1]))


@torch.library.custom_op(
    "mlops::embedding_builtin_deterministic_bwd",
    mutates_args=(),
)
def _backward_op(
    tokens: torch.Tensor,
    grad_output: torch.Tensor,
    num_embeddings: int,
) -> torch.Tensor:
    return backward(tokens, grad_output, int(num_embeddings))


@_backward_op.register_fake
def _backward_fake(tokens, grad_output, num_embeddings):
    del tokens
    return grad_output.new_empty((num_embeddings, grad_output.shape[-1]))


def _setup_context(ctx, inputs, output):
    tokens, weight = inputs
    ctx.save_for_backward(tokens)
    ctx.num_embeddings = weight.shape[0]


def _autograd_backward(ctx, grad_output):
    (tokens,) = ctx.saved_tensors
    return None, _backward_op(tokens, grad_output, ctx.num_embeddings)


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def apply(tokens, weight):
    return _forward_op(tokens, weight)


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="embedding",
        implementation_id="builtin.embedding.deterministic",
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
