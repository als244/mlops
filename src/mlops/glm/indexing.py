"""GLM k-pool indexing with sequence-local, bounded score tiles.

The indexer is discrete/non-differentiable, matching the published model.
Boundaries are explicit CPU metadata. No GPU scalar reads or T-by-T matrix.
"""

from itertools import pairwise

import torch


@torch.library.custom_op("mlops_glm::pooled_topk", mutates_args=())
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
    pool, width = position_bias.shape
    ends = boundaries.tolist()
    if (
        not ends
        or ends[0] != 0
        or ends[-1] != q.shape[0]
        or any(a >= b for a, b in pairwise(ends))
    ):
        raise ValueError("CPU boundaries must strictly cover all packed tokens")
    result = torch.full(
        (q.shape[0], top_k + pool - 1), -1, device=q.device, dtype=torch.int32
    )
    for start, stop in pairwise(ends):
        # Below the key budget, every causal key is selected. Ranking them
        # cannot change attention; avoid the score GEMMs entirely.
        if stop - start <= top_k:
            keys = torch.arange(start, stop, device=q.device, dtype=torch.int32)
            for first in range(start, stop, query_chunk):
                last = min(first + query_chunk, stop)
                rows = torch.arange(first, last, device=q.device)
                result[first:last, : stop - start] = (
                    keys[None, :]
                    .expand(last - first, -1)
                    .masked_fill(keys[None, :] > rows[:, None], -1)
                )
            continue
        n_pools = (stop - start) // pool
        if n_pools:
            count = n_pools * pool
            logits = (
                gates[start : start + count].reshape(n_pools, pool, width).float()
                + position_bias.float()
            )
            probabilities = logits.softmax(1).to(key.dtype)
            pooled = (
                (
                    probabilities
                    * key[start : start + count].reshape(n_pools, pool, width)
                )
                .sum(1)
                .float()
            )
        keep = min(top_k // pool, n_pools)
        for first in range(start, stop, query_chunk):
            last = min(first + query_chunk, stop)
            rows = last - first
            position = torch.arange(first - start, last - start, device=q.device)
            queries = q[first:last].float().reshape(-1, width)
            weights = (head_weights[first:last].float() * q.shape[1] ** -0.5).unsqueeze(
                1
            )
            selected_scores = torch.empty((rows, 0), device=q.device)
            selected_pools = torch.empty((rows, 0), device=q.device, dtype=torch.int64)
            for key_first in range(0, n_pools, key_chunk):
                key_last = min(key_first + key_chunk, n_pools)
                scores = queries @ pooled[key_first:key_last].T
                scores = (scores.reshape(rows, q.shape[1], -1) * width**-0.5).relu()
                scores = weights @ scores
                scores = scores.squeeze(1)
                candidates = torch.arange(key_first, key_last, device=q.device)
                visible = (candidates + 1) * pool - 1 <= position[:, None]
                scores = scores.masked_fill(~visible, float("-inf"))
                merged = torch.cat((selected_scores, scores), -1)
                merged_ids = torch.cat(
                    (selected_pools, candidates.expand(rows, -1)), -1
                )
                selected_scores, offsets = merged.topk(
                    min(keep, merged.shape[-1]), dim=-1
                )
                selected_pools = merged_ids.gather(-1, offsets)
            if keep:
                indices = selected_pools[..., None] * pool + torch.arange(
                    pool, device=q.device
                )
                indices = indices + start
                indices = indices.masked_fill(
                    ~selected_scores.isfinite().unsqueeze(-1), -1
                )
                result[first:last, : keep * pool] = indices.flatten(-2).to(torch.int32)
            # Always include the visible incomplete pool, even before the first full pool.
            if pool > 1:
                visible_count = position + 1
                tail_count = visible_count % pool
                tail_offset = torch.arange(pool - 1, device=q.device)
                tail = start + (visible_count - tail_count)[:, None] + tail_offset
                tail = tail.masked_fill(tail_offset >= tail_count[:, None], -1)
                result[first:last, keep * pool : keep * pool + pool - 1] = tail.to(
                    torch.int32
                )
    return result


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


def pooled_topk(
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
    if (
        q.ndim != 3
        or key.shape != (q.shape[0], q.shape[-1])
        or gates.shape != key.shape
    ):
        raise ValueError("Expected Q [T,H,D] and key/gates [T,D]")
    if position_bias.ndim != 2 or position_bias.shape[1] != q.shape[-1]:
        raise ValueError("Expected per-pool-position bias [pool_size,D]")
    if head_weights.shape != q.shape[:2]:
        raise ValueError("Expected head weights [T,H]")
    if (
        boundaries.device.type != "cpu"
        or boundaries.ndim != 1
        or boundaries.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "Packed sequence boundaries must be explicit CPU integer metadata"
        )
    if (
        top_k < position_bias.shape[0]
        or top_k % position_bias.shape[0]
        or min(query_chunk, key_chunk) < 1
    ):
        raise ValueError(
            "top_k must be a positive multiple of pool size and tile sizes must be positive"
        )
    if not all(
        t.is_cuda and t.device == q.device
        for t in (q, key, gates, head_weights, position_bias)
    ):
        raise ValueError("Indexer payloads must share one CUDA device")
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
