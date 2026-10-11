"""Registration for non-differentiable pooled causal index selection."""

import torch

from ...dispatch import attention_costs as costs
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation


@torch.library.custom_op("mlops::pooled_topk_builtin_tiled", mutates_args=())
def _select(
    q: torch.Tensor,
    key: torch.Tensor,
    gates: torch.Tensor,
    head_weights: torch.Tensor,
    position_bias: torch.Tensor,
    boundaries: torch.Tensor,
    top_k: int,
    query_chunk: int,
    key_chunk: int,
) -> torch.Tensor:
    from ...kernels.pooled_indexer import pooled_topk

    return pooled_topk(
        q,
        key,
        gates,
        head_weights,
        position_bias,
        boundaries,
        top_k,
        query_chunk,
        key_chunk,
    )


@_select.register_fake
def _fake(
    q,
    key,
    gates,
    head_weights,
    position_bias,
    boundaries,
    top_k,
    query_chunk,
    key_chunk,
):
    return torch.empty(
        (q.shape[0], top_k + position_bias.shape[0] - 1),
        device=q.device,
        dtype=torch.int32,
    )


def _supports(
    q,
    key,
    gates,
    head_weights,
    position_bias,
    boundaries,
    *,
    top_k=2048,
    query_chunk=64,
    key_chunk=128,
    surface,
    **_kwargs,
):
    if surface == "explicit":
        return SupportResult.no("pooled selection is non-differentiable")
    if not all(t.is_cuda for t in (q, key, gates, head_weights, position_bias)):
        return SupportResult.no("pooled selection payloads require CUDA")
    if boundaries.device.type != "cpu":
        return SupportResult.no("pooled selection boundaries must be CPU metadata")
    return SupportResult.yes()


def apply(
    q,
    key,
    gates,
    head_weights,
    position_bias,
    boundaries,
    *,
    top_k=2048,
    query_chunk=64,
    key_chunk=128,
):
    return _select(
        q,
        key,
        gates,
        head_weights,
        position_bias,
        boundaries,
        top_k,
        query_chunk,
        key_chunk,
    )


@flop_formula(_select)
def _flops(
    q,
    key,
    gates,
    head_weights,
    position_bias,
    boundaries,
    top_k,
    query_chunk,
    key_chunk,
    **_kwargs,
):
    return costs.pooled_topk(
        q, key, gates, head_weights, position_bias, boundaries, top_k=top_k
    ).logical_flops


IMPLEMENTATION = register_implementation(
    Implementation(
        "pooled_topk",
        "builtin.pooled_topk.tiled",
        "builtin",
        100,
        False,
        _supports,
        apply,
    )
)
