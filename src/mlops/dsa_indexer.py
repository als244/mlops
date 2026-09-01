"""Semantic implementation boundary for DeepSeek sparse-attention selection."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from .kernels.dsa_indexer import (
    selection_mask,
    sequence_bounds,
    topk_indices as _topk_indices,
)
from .dispatch import resolve_implementation


@dataclass
class _SelectionOverrides:
    decisions: tuple[torch.Tensor, ...]
    consumed: int = 0


_SELECTION_OVERRIDES: ContextVar[_SelectionOverrides | None] = ContextVar(
    "dsa_selection_overrides", default=None
)


def dsa_indexer_key_norm(x, weight, bias, eps=1e-5):
    """Apply the selected DSA indexer key-normalization implementation."""
    implementation = resolve_implementation(
        "dsa_indexer_key_norm", x, weight, bias, float(eps), surface="semantic"
    )
    return implementation.apply(x, weight, bias, float(eps))


@contextmanager
def dsa_selection_override(decisions):
    """Temporarily replace discrete top-k decisions for fidelity diagnosis."""
    state = _SelectionOverrides(tuple(item.detach() for item in decisions))
    token = _SELECTION_OVERRIDES.set(state)
    try:
        yield
        if state.consumed != len(state.decisions):
            raise RuntimeError(
                f"DSA override supplied {len(state.decisions)} decisions but "
                f"consumed {state.consumed}"
            )
    finally:
        _SELECTION_OVERRIDES.reset(token)


def _next_selection_override(scores, lengths):
    state = _SELECTION_OVERRIDES.get()
    if state is None:
        return None
    if state.consumed >= len(state.decisions):
        raise RuntimeError("DSA selection override exhausted")
    indices = state.decisions[state.consumed].to(scores.device).long()
    state.consumed += 1
    if indices.ndim != 2 or indices.shape[0] != scores.shape[0]:
        raise ValueError(
            f"DSA selection override has shape {tuple(indices.shape)}; "
            f"expected ({scores.shape[0]}, selected_width)"
        )
    for start, stop in sequence_bounds(lengths):
        segment = indices[start:stop]
        if bool(((segment < start) | (segment >= stop)).any()):
            raise ValueError("DSA override selects an index outside its sequence")
    return indices.to(torch.int32)


def index_scores(q, k, weights, lengths):
    """Return causal, block-diagonal lightning-indexer scores."""
    implementation = resolve_implementation(
        "index_scores", q, k, weights, lengths, surface="semantic"
    )
    return implementation.apply(q, k, weights, lengths)


def topk_indices(scores, lengths, top_k):
    """Select DSA keys, honoring an optional diagnostic override."""
    # Overrides are verification state, not part of a frozen model artifact.
    forced = (
        None
        if torch.compiler.is_compiling()
        else _next_selection_override(scores, lengths)
    )
    return forced if forced is not None else _topk_indices(scores, lengths, top_k)


__all__ = [
    "dsa_indexer_key_norm",
    "dsa_selection_override",
    "index_scores",
    "selection_mask",
    "topk_indices",
]
