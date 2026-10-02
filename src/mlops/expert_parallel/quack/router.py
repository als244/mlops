"""SonicMoE token-choice router and metadata, independent of expert precision.

Token rounding is deliberately absent. The upstream local gather permutations
are not a distributed MoonEP plan: MoonEP owns placement and replica expansion.
We pass SonicMoE's expert frequency directly to MoonEP, avoiding another caller
histogram. The full upstream metadata pipeline is retained for initial comparison.
"""

import torch
from torch import Tensor

from .._compat.quack_autotune import router_kernels


@torch.library.custom_op("mlops_ep_quack_router::forward", mutates_args=())
def route_op(logits: Tensor, k: int, normalize: bool) -> tuple[Tensor, Tensor, Tensor]:
    _topk_softmax_fwd, _, TC_topk_router_metadata_triton = router_kernels()

    rows, experts = logits.shape
    scores = torch.empty((rows, k), device=logits.device, dtype=torch.float32)
    ids = torch.empty((rows, k), device=logits.device, dtype=torch.int32)
    with torch.cuda.nvtx.range("moon_quack/routing/topk_softmax"):
        _topk_softmax_fwd(
            logits,
            scores,
            ids,
            experts,
            k,
            is_softmax_over_topk=normalize,
            norm_topk_probs=False,
        )
    frequency = torch.empty(experts, device=logits.device, dtype=torch.int32)
    offsets = torch.empty(experts + 1, device=logits.device, dtype=torch.int32)
    gather = torch.empty(rows * k, device=logits.device, dtype=torch.int32)
    scatter = torch.empty_like(gather)
    reverse = torch.empty_like(gather)
    with torch.cuda.nvtx.range("moon_quack/routing/metadata"):
        TC_topk_router_metadata_triton(
            ids, experts, frequency, offsets, gather, scatter, reverse
        )
    return scores, ids, frequency


@route_op.register_fake
def route_fake(logits, k, normalize):
    return (
        logits.new_empty((logits.shape[0], k), dtype=torch.float32),
        logits.new_empty((logits.shape[0], k), dtype=torch.int32),
        logits.new_empty((logits.shape[1],), dtype=torch.int32),
    )


@torch.library.custom_op("mlops_ep_quack_router::backward", mutates_args=())
def route_backward(
    logits: Tensor, scores: Tensor, ids: Tensor, dp: Tensor, normalize: bool
) -> Tensor:
    _, _topk_softmax_bwd, _ = router_kernels()

    with torch.cuda.nvtx.range("moon_quack/routing/backward"):
        out = torch.zeros_like(logits)
        _topk_softmax_bwd(
            logits,
            out,
            None,
            dp.contiguous(),
            scores,
            ids,
            logits.shape[1],
            ids.shape[1],
            is_softmax_over_topk=normalize,
            norm_topk_probs=False,
        )
    return out


@route_backward.register_fake
def route_backward_fake(logits, scores, ids, dp, normalize):
    return torch.empty_like(logits)


def _setup(ctx, inputs, output):
    logits, _, ctx.normalize = inputs
    scores, ids, frequency = output
    ctx.save_for_backward(logits, scores, ids)
    ctx.mark_non_differentiable(ids, frequency)
    ctx.set_materialize_grads(False)


def _backward(ctx, dp, _ids, _frequency):
    if dp is None:
        return None, None, None
    return route_backward(*ctx.saved_tensors, dp, ctx.normalize), None, None


route_op.register_autograd(_backward, setup_context=_setup)
