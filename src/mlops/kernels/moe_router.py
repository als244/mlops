"""Fused softmax/top-k router and analytic backward."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


_MODE = {"topk_then_softmax": 0, "softmax_then_topk": 1}


@dataclass
class _RoutingOverrides:
    decisions: tuple[torch.Tensor, ...]
    consumed: int = 0


_OVERRIDES: ContextVar[_RoutingOverrides | None] = ContextVar(
    "routing_overrides", default=None
)
_WEIGHT_PRECISION = ContextVar("route_weight_precision", default="float32")


@torch.compiler.assume_constant_result
def current_route_weight_precision() -> str:
    """Return the capture-time routing precision as an artifact constant."""
    return _WEIGHT_PRECISION.get()


@contextmanager
def route_weight_precision(name: str):
    """Select fp32 route weights or the historical input-dtype ablation."""
    if name not in {"float32", "input"}:
        raise ValueError(f"unsupported route-weight precision {name!r}")
    token = _WEIGHT_PRECISION.set(name)
    try:
        yield
    finally:
        _WEIGHT_PRECISION.reset(token)


@contextmanager
def routing_override(decisions):
    """Force successive router calls to use caller-supplied top-k indices.

    Only the discrete selection is overridden.  Route weights remain live,
    differentiable functions of the caller's own logits.  This is a
    diagnostic ablation for separating near-tie branch flips from continuous
    kernel error; it is context-local and does not alter normal model calls.
    """
    state = _RoutingOverrides(tuple(decision.detach() for decision in decisions))
    token = _OVERRIDES.set(state)
    try:
        yield
        if state.consumed != len(state.decisions):
            raise RuntimeError(
                f"routing override supplied {len(state.decisions)} decisions "
                f"but consumed {state.consumed}"
            )
    finally:
        _OVERRIDES.reset(token)


def _next_override(logits, top_k):
    state = _OVERRIDES.get()
    if state is None:
        return None
    if state.consumed >= len(state.decisions):
        raise RuntimeError("routing override exhausted before model forward ended")
    ids = state.decisions[state.consumed]
    state.consumed += 1
    expected = (logits.shape[0], int(top_k))
    if tuple(ids.shape) != expected:
        raise ValueError(
            f"routing override {state.consumed - 1} has shape {tuple(ids.shape)}; "
            f"expected {expected}"
        )
    if ids.device != logits.device:
        ids = ids.to(logits.device)
    ids = ids.long()
    if bool(((ids < 0) | (ids >= logits.shape[1])).any()):
        raise ValueError("routing override contains an out-of-range expert id")
    if bool(
        (
            torch.sort(ids, dim=-1).values[:, 1:]
            == torch.sort(ids, dim=-1).values[:, :-1]
        ).any()
    ):
        raise ValueError("routing override contains duplicate experts in one row")
    return ids


if triton is not None:

    @triton.jit
    def _topk_kernel(
        logits_ptr,
        weights_ptr,
        ids_ptr,
        tokens,
        E: tl.constexpr,
        K: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_T: tl.constexpr,
        MODE: tl.constexpr,
    ):
        token = (tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)).to(
            tl.int64
        )
        token_mask = token < tokens
        expert = tl.arange(0, BLOCK_E).to(tl.int64)
        expert_mask = expert < E
        logits = tl.load(
            logits_ptr + token[:, None] * E + expert[None, :],
            mask=token_mask[:, None] & expert_mask[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        if MODE == 1:
            maximum = tl.max(logits, axis=1)
            exponentials = tl.exp(logits - maximum[:, None])
            exponentials = tl.where(expert_mask[None, :], exponentials, 0.0)
            remaining = exponentials / tl.sum(exponentials, axis=1)[:, None]
        else:
            remaining = logits
        chosen = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.int1)
        for index in range(K):
            maximum = tl.max(remaining, axis=1)
            is_maximum = (remaining == maximum[:, None]) & expert_mask[None, :]
            candidate = tl.where(is_maximum, expert[None, :], BLOCK_E)
            expert_id = tl.minimum(tl.min(candidate, axis=1), E - 1)
            tl.store(
                ids_ptr + token * K + index,
                expert_id.to(tl.int32),
                mask=token_mask,
            )
            if MODE == 1:
                tl.store(
                    weights_ptr + token * K + index,
                    maximum.to(weights_ptr.dtype.element_ty),
                    mask=token_mask,
                )
            selected = expert[None, :] == expert_id[:, None]
            chosen |= selected
            remaining = tl.where(selected, -float("inf"), remaining)
        if MODE == 0:
            selected_logits = tl.where(chosen, logits, -float("inf"))
            maximum = tl.max(selected_logits, axis=1)
            exponentials = tl.exp(selected_logits - maximum[:, None])
            exponentials = tl.where(chosen, exponentials, 0.0)
            probabilities = exponentials / tl.sum(exponentials, axis=1)[:, None]
            for index in range(K):
                expert_id = tl.load(
                    ids_ptr + token * K + index, mask=token_mask, other=0
                ).to(tl.int64)
                hit = expert[None, :] == expert_id[:, None]
                probability = tl.sum(tl.where(hit, probabilities, 0.0), axis=1)
                tl.store(
                    weights_ptr + token * K + index,
                    probability.to(weights_ptr.dtype.element_ty),
                    mask=token_mask,
                )

    @triton.jit
    def _backward_kernel(
        grad_weights_ptr,
        weights_ptr,
        ids_ptr,
        logits_ptr,
        grad_logits_ptr,
        tokens,
        E: tl.constexpr,
        K: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
        MODE: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        k_offset = tl.arange(0, BLOCK_K)
        k_mask = k_offset < K
        expert = tl.arange(0, BLOCK_E).to(tl.int64)
        expert_mask = expert < E
        grad = tl.load(
            grad_weights_ptr + token * K + k_offset,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        ids = tl.load(ids_ptr + token * K + k_offset, mask=k_mask, other=0).to(tl.int64)
        selected = k_mask[:, None] & (expert[None, :] == ids[:, None])
        if MODE == 0:
            probability = tl.load(
                weights_ptr + token * K + k_offset,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            selected_grad = probability * (grad - tl.sum(grad * probability))
            full_grad = tl.sum(tl.where(selected, selected_grad[:, None], 0.0), axis=0)
        else:
            logits = tl.load(
                logits_ptr + token * E + expert,
                mask=expert_mask,
                other=-float("inf"),
            ).to(tl.float32)
            maximum = tl.max(logits)
            exponentials = tl.where(expert_mask, tl.exp(logits - maximum), 0.0)
            full_probability = exponentials / tl.sum(exponentials)
            selected_probability = tl.sum(
                tl.where(selected, full_probability[None, :], 0.0), axis=1
            )
            dot = tl.sum(grad * selected_probability)
            full_grad = -full_probability * dot + tl.sum(
                tl.where(
                    selected,
                    (grad * selected_probability)[:, None],
                    0.0,
                ),
                axis=0,
            )
        tl.store(
            grad_logits_ptr + token * E + expert,
            full_grad,
            mask=expert_mask,
        )


def route(
    logits,
    top_k,
    mode,
    *,
    bias=None,
    n_group=1,
    topk_group=1,
    routed_scaling=1.0,
    weight_precision=None,
    use_context_override=True,
):
    """Select route weights/IDs with explicit compiler-capture controls.

    Normal callers inherit the context-local precision and routing override.
    Provider artifact builders pass both choices explicitly so strict Export
    never observes ``ContextVar`` state.
    """
    tokens, experts = logits.shape
    # Routing is logically fp32 even though the router GEMM and expert
    # activations use the model storage dtype.  Keeping the selected weights
    # in fp32 matches the ordinary PyTorch reference: the full softmax (or
    # sigmoid normalization) is evaluated in fp32 and its selected values are
    # consumed directly by the expert combine.  Rounding this small (T, K)
    # tensor to bf16 caused an avoidable ~1.7e-3 relative error at every MoE
    # layer and amplified near-tie route instability in deep stacks.
    selected_precision = (
        _WEIGHT_PRECISION.get() if weight_precision is None else str(weight_precision)
    )
    if selected_precision not in {"float32", "input"}:
        raise ValueError(f"unsupported route-weight precision {selected_precision!r}")
    weights = torch.empty(
        (tokens, int(top_k)),
        dtype=(torch.float32 if selected_precision == "float32" else logits.dtype),
        device=logits.device,
    )
    ids = torch.empty((tokens, int(top_k)), dtype=torch.int32, device=logits.device)
    forced_ids = _next_override(logits, top_k) if use_context_override else None
    if forced_ids is not None:
        logits_float = logits.float()
        if mode == "sigmoid_noaux_tc":
            scores = torch.sigmoid(logits_float)
            picked = scores.gather(1, forced_ids)
            selected_weights = (
                picked / picked.sum(-1, keepdim=True) * float(routed_scaling)
            )
        elif mode == "softmax_then_topk":
            selected_weights = torch.softmax(logits_float, -1).gather(1, forced_ids)
        elif mode == "topk_then_softmax":
            selected_weights = torch.softmax(logits_float.gather(1, forced_ids), -1)
        else:
            raise ValueError(f"unsupported routing mode {mode!r}")
        weights.copy_(selected_weights)
        ids.copy_(forced_ids.to(torch.int32))
        return weights, ids
    if mode == "sigmoid_noaux_tc":
        scores = torch.sigmoid(logits.float())
        with torch.no_grad():
            selection = scores + (0.0 if bias is None else bias.float())
            grouped = selection.view(tokens, int(n_group), experts // int(n_group))
            grouped, _ = torch.sort(grouped, dim=-1, descending=True, stable=True)
            group_scores = grouped[..., : min(2, grouped.shape[-1])].sum(-1)
            _, group_ids = torch.sort(
                group_scores, dim=-1, descending=True, stable=True
            )
            group_mask = torch.zeros(
                tokens, int(n_group), dtype=torch.bool, device=logits.device
            )
            group_mask.scatter_(1, group_ids[:, : int(topk_group)], True)
            expert_mask = group_mask.repeat_interleave(experts // int(n_group), dim=1)
            masked = selection.masked_fill(~expert_mask, float("-inf"))
            _, selected_ids = torch.sort(masked, dim=-1, descending=True, stable=True)
            selected_ids = selected_ids[:, : int(top_k)]
        picked = scores.gather(1, selected_ids)
        selected_weights = picked / picked.sum(-1, keepdim=True) * float(routed_scaling)
        weights.copy_(selected_weights.to(weights.dtype))
        ids.copy_(selected_ids.to(torch.int32))
        return weights, ids
    if mode not in _MODE:
        raise ValueError(f"unsupported routing mode {mode!r}")
    if triton is not None and logits.is_cuda and logits.is_contiguous():
        block_tokens = 8
        _topk_kernel[(triton.cdiv(tokens, block_tokens),)](
            logits,
            weights,
            ids,
            tokens,
            E=experts,
            K=int(top_k),
            BLOCK_E=triton.next_power_of_2(experts),
            BLOCK_T=block_tokens,
            MODE=_MODE[mode],
        )
        return weights, ids
    values = (
        torch.softmax(logits.float(), -1)
        if mode == "softmax_then_topk"
        else logits.float()
    )
    values, indices = torch.sort(values, dim=-1, descending=True, stable=True)
    ids.copy_(indices[:, :top_k].to(torch.int32))
    selected = values[:, :top_k]
    if mode == "topk_then_softmax":
        selected = torch.softmax(selected, -1)
    weights.copy_(selected.to(weights.dtype))
    return weights, ids


def route_torch(
    logits,
    top_k,
    mode,
    *,
    bias=None,
    n_group=1,
    topk_group=1,
    routed_scaling=1.0,
):
    """Raw-model routing expressions with stable Torch selection/autograd."""
    logits_float = logits.float()
    forced_ids = _next_override(logits, top_k)
    if forced_ids is not None:
        selected_ids = forced_ids
    elif mode == "sigmoid_noaux_tc":
        scores = torch.sigmoid(logits_float)
        with torch.no_grad():
            selection = scores + (0.0 if bias is None else bias.float())
            tokens, experts = selection.shape
            grouped = selection.view(tokens, int(n_group), experts // int(n_group))
            grouped, _ = torch.sort(grouped, dim=-1, descending=True, stable=True)
            group_scores = grouped[..., : min(2, grouped.shape[-1])].sum(-1)
            _, group_ids = torch.sort(
                group_scores, dim=-1, descending=True, stable=True
            )
            group_mask = torch.zeros(
                tokens, int(n_group), dtype=torch.bool, device=logits.device
            )
            group_mask.scatter_(1, group_ids[:, : int(topk_group)], True)
            expert_mask = group_mask.repeat_interleave(experts // int(n_group), dim=1)
            masked = selection.masked_fill(~expert_mask, float("-inf"))
            _, selected_ids = torch.sort(masked, dim=-1, descending=True, stable=True)
            selected_ids = selected_ids[:, : int(top_k)]
    elif mode in _MODE:
        selection = (
            torch.softmax(logits_float, dim=-1)
            if mode == "softmax_then_topk"
            else logits_float
        )
        _, selected_ids = torch.sort(selection, dim=-1, descending=True, stable=True)
        selected_ids = selected_ids[:, : int(top_k)]
    else:
        raise ValueError(f"unsupported routing mode {mode!r}")

    if mode == "sigmoid_noaux_tc":
        scores = torch.sigmoid(logits_float)
        picked = scores.gather(1, selected_ids)
        selected_weights = picked / picked.sum(-1, keepdim=True) * float(routed_scaling)
    elif mode == "softmax_then_topk":
        selected_weights = torch.softmax(logits_float, dim=-1).gather(1, selected_ids)
    else:
        selected_weights = torch.softmax(logits_float.gather(1, selected_ids), dim=-1)
    return selected_weights, selected_ids.to(torch.int32)


def route_backward(grad_weights, weights, ids, logits, mode):
    grad_weights = grad_weights.contiguous()
    grad_logits = torch.empty_like(logits, dtype=torch.float32)
    if mode == "sigmoid_noaux_tc":
        selected = ids.long()
        grad = grad_weights.float()
        route_weight = weights.float()
        selected_scores = torch.sigmoid(logits.float()).gather(1, selected)
        score_sum = selected_scores.sum(-1, keepdim=True)
        scaling = route_weight.sum(-1, keepdim=True)
        dot = (grad * route_weight).sum(-1, keepdim=True)
        selected_grad = (
            ((scaling * grad - dot) / score_sum)
            * selected_scores
            * (1.0 - selected_scores)
        )
        grad_logits.zero_()
        grad_logits.scatter_(1, selected, selected_grad)
        return grad_logits
    if triton is not None and logits.is_cuda and logits.is_contiguous():
        _backward_kernel[(logits.shape[0],)](
            grad_weights,
            weights,
            ids,
            logits,
            grad_logits,
            logits.shape[0],
            E=logits.shape[1],
            K=ids.shape[1],
            BLOCK_E=triton.next_power_of_2(logits.shape[1]),
            BLOCK_K=triton.next_power_of_2(ids.shape[1]),
            MODE=_MODE[mode],
        )
        return grad_logits
    selected = ids.long()
    if mode == "topk_then_softmax":
        probability = weights.float()
        grad_selected = probability * (
            grad_weights.float()
            - (grad_weights.float() * probability).sum(-1, keepdim=True)
        )
        grad_logits.zero_()
        grad_logits.scatter_(1, selected, grad_selected)
    else:
        probability = torch.softmax(logits.float(), -1)
        top_probability = probability.gather(1, selected)
        dot = (grad_weights.float() * top_probability).sum(-1, keepdim=True)
        grad_logits.copy_(-probability * dot)
        grad_logits.scatter_add_(1, selected, grad_weights.float() * top_probability)
    return grad_logits


def add_aux_gradient_(grad_logits, logits, ids, scale):
    if scale is None:
        return grad_logits
    tokens, experts = logits.shape
    counts = torch.zeros(experts, dtype=torch.float32, device=logits.device)
    counts.scatter_add_(
        0, ids.reshape(-1).long(), torch.ones(ids.numel(), device=ids.device)
    )
    frequency = counts / ids.numel()
    probability = torch.softmax(logits.float(), -1)
    coefficient = scale.float() * experts / tokens
    grad_logits.add_(
        coefficient
        * probability
        * (frequency.unsqueeze(0) - (probability @ frequency).unsqueeze(1))
    )
    return grad_logits


def add_sequence_aux_gradient_(grad_logits, logits, ids, scale, lengths):
    if scale is None:
        return grad_logits
    scores = torch.sigmoid(logits.float())
    row_sum = scores.sum(-1, keepdim=True)
    normalized = scores / row_sum
    experts = logits.shape[1]
    start = 0
    for length in lengths:
        stop = start + int(length)
        counts = torch.zeros(experts, dtype=torch.float32, device=logits.device)
        segment_ids = ids[start:stop].reshape(-1).long()
        counts.scatter_add_(
            0, segment_ids, torch.ones(segment_ids.numel(), device=logits.device)
        )
        frequency = counts * experts / (ids.shape[1] * int(length))
        coefficient = scale.float() / (int(length) * row_sum[start:stop])
        score_gradient = coefficient * (
            frequency.unsqueeze(0) - (normalized[start:stop] @ frequency).unsqueeze(1)
        )
        grad_logits[start:stop].add_(
            score_gradient * scores[start:stop] * (1.0 - scores[start:stop])
        )
        start = stop
    return grad_logits


__all__ = [
    "add_aux_gradient_",
    "add_sequence_aux_gradient_",
    "current_route_weight_precision",
    "route",
    "route_torch",
    "route_backward",
    "route_weight_precision",
    "routing_override",
]
