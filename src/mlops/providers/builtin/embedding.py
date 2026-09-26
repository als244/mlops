"""Package-owned deterministic embedding implementation."""

from __future__ import annotations

import torch

from ...dispatch.context import weight_gradient_dtype
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


def backward(tokens, grad_output, num_embeddings, weight_grad_dtype=None):
    return embedding_backward(
        tokens, grad_output, int(num_embeddings), weight_grad_dtype
    )


# The forward takes the weight-gradient dtype too, so a captured forward
# records the one its backward will use.
@torch.library.custom_op(
    "mlops::embedding_builtin_deterministic_fwd",
    mutates_args=(),
)
def _forward_op(
    tokens: torch.Tensor,
    weight: torch.Tensor,
    weight_grad_dtype: torch.dtype | None,
) -> torch.Tensor:
    del weight_grad_dtype
    return forward(tokens, weight)


@_forward_op.register_fake
def _forward_fake(tokens, weight, weight_grad_dtype):
    del weight_grad_dtype
    return weight.new_empty((*tokens.shape, weight.shape[1]))


@torch.library.custom_op(
    "mlops::embedding_builtin_deterministic_bwd",
    mutates_args=(),
)
def _backward_op(
    tokens: torch.Tensor,
    grad_output: torch.Tensor,
    num_embeddings: int,
    weight_grad_dtype: torch.dtype | None,
) -> torch.Tensor:
    return backward(tokens, grad_output, int(num_embeddings), weight_grad_dtype)


@_backward_op.register_fake
def _backward_fake(tokens, grad_output, num_embeddings, weight_grad_dtype):
    del tokens
    return grad_output.new_empty(
        (num_embeddings, grad_output.shape[-1]),
        dtype=grad_output.dtype if weight_grad_dtype is None else weight_grad_dtype,
    )


def _setup_context(ctx, inputs, output):
    tokens, weight, weight_grad_dtype = inputs
    ctx.save_for_backward(tokens)
    ctx.num_embeddings = weight.shape[0]
    ctx.weight_grad_dtype = weight_grad_dtype


def _autograd_backward(ctx, grad_output):
    (tokens,) = ctx.saved_tensors
    grad_weight = _backward_op(
        tokens, grad_output, ctx.num_embeddings, ctx.weight_grad_dtype
    )
    return None, grad_weight, None


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def apply(tokens, weight):
    return _forward_op(tokens, weight, weight_gradient_dtype())


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
