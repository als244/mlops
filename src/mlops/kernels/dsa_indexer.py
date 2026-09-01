"""Pure tensor helpers for DeepSeek sparse-attention index selection."""

from __future__ import annotations

import torch


def sequence_bounds(lengths):
    """Yield half-open token ranges for a packed sequence-length list."""
    start = 0
    for length in lengths:
        stop = start + int(length)
        yield start, stop
        start = stop


def index_scores(q, k, weights, lengths, *, reduction: str):
    """Build the block-diagonal causal indexer score matrix."""
    if reduction not in {"einsum", "multiply_sum"}:
        raise ValueError(f"unsupported DSA indexer reduction {reduction!r}")
    tokens = q.shape[0]
    scores = torch.full(
        (tokens, tokens), float("-inf"), dtype=torch.float32, device=q.device
    )
    for lo, hi in sequence_bounds(lengths):
        relation = torch.einsum(
            "thd,sd->ths", q[lo:hi].float(), k[lo:hi].float()
        ).clamp_min_(0.0)
        if reduction == "multiply_sum":
            block = (weights[lo:hi].float().unsqueeze(-1) * relation).sum(1)
        else:
            block = torch.einsum("th,ths->ts", weights[lo:hi].float(), relation)
        causal = torch.triu(
            torch.ones(hi - lo, hi - lo, dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        scores[lo:hi, lo:hi] = block.masked_fill(causal, float("-inf"))
    return scores


def topk_indices(scores, lengths, top_k):
    """Select per-sequence indices and convert them to global offsets."""
    selected = []
    for lo, hi in sequence_bounds(lengths):
        width = min(int(top_k), hi - lo)
        local = torch.topk(scores[lo:hi, lo:hi], width, dim=-1, sorted=True).indices
        if width < int(top_k):
            # A packed batch uses one rectangular index tensor. Duplicating an
            # existing selected key pads only the representation; the mask is
            # unchanged because it is formed with scatter.
            padding = local[:, :1].expand(-1, int(top_k) - width)
            local = torch.cat((local, padding), dim=-1)
        selected.append(local + lo)
    return torch.cat(selected, dim=0).to(torch.int32)


def selection_mask(indices, lo, hi, row_lo, row_hi):
    """Build an additive selected-and-causal mask for a token-row block."""
    mask = torch.full(
        (row_hi - row_lo, hi - lo),
        float("-inf"),
        dtype=torch.float32,
        device=indices.device,
    )
    local = (indices[row_lo:row_hi].long() - lo).clamp_(0, hi - lo - 1)
    mask.scatter_(-1, local, 0.0)
    rows = torch.arange(row_lo, row_hi, device=indices.device).unsqueeze(1)
    columns = torch.arange(lo, hi, device=indices.device).unsqueeze(0)
    return mask.masked_fill_(columns > rows, float("-inf"))


__all__ = [
    "index_scores",
    "selection_mask",
    "sequence_bounds",
    "topk_indices",
]
