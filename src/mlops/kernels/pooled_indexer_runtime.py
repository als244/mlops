"""Packed pooled selection whose boundary values remain on the device.

All allocation and launch sizes depend on tensor shapes, never boundary values.
Complete pools are compacted by a device prefix sum. Query chunks use the same
scorer and selection arithmetic as the explicit-host-metadata implementation.
"""

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

from . import pooled_indexer as shared
from .pooled_indexer import _interval, _pooled_scores_kernel, _select_row


@triton.jit
def _pool_keys(
    keys,
    gates,
    bias,
    boundaries,
    offsets,
    output,
    key_t,
    key_d,
    gate_t,
    gate_d,
    bias_p,
    bias_d,
    POOL: tl.constexpr,
    WIDTH: tl.constexpr,
    BOUNDARIES: tl.constexpr,
    POOL_BLOCK: tl.constexpr,
    WIDTH_BLOCK: tl.constexpr,
):
    pool = tl.program_id(0)
    total = tl.load(offsets + BOUNDARIES - 1)
    if pool < total:
        sequence = _interval(offsets, pool, BOUNDARIES)
        first = tl.load(boundaries + sequence)
        pool_first = tl.load(offsets + sequence)
        row = tl.arange(0, POOL_BLOCK)
        lane = tl.arange(0, WIDTH_BLOCK)
        token = first + (pool - pool_first) * POOL + row
        mask = (row[:, None] < POOL) & (lane[None, :] < WIDTH)
        value = tl.load(keys + token[:, None] * key_t + lane[None, :] * key_d, mask, 0)
        logits = tl.load(
            gates + token[:, None] * gate_t + lane[None, :] * gate_d, mask, 0
        ).to(tl.float32) + tl.load(
            bias + row[:, None] * bias_p + lane[None, :] * bias_d, mask, 0
        ).to(tl.float32)
        logits = tl.where(row[:, None] < POOL, logits, float("-inf"))
        probability = tl.exp(logits - tl.max(logits, 0)[None, :])
        probability /= tl.sum(probability, 0)[None, :]
        # Match the key-dtype softmax cast, product rounding and sum in _pooled_keys.
        product = (probability.to(value.dtype) * value).to(value.dtype)
        pooled = tl.sum(product.to(tl.float32), 0).to(value.dtype)
        tl.store(output + pool * WIDTH + lane, pooled, lane < WIDTH)


@triton.jit
def _causal_row(output, start, position, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    for first in range(tl.cdiv(WIDTH, BLOCK)):
        col = first * BLOCK + lane
        key = start + col
        tl.store(output + col, tl.where(key <= position, key, -1), col < WIDTH)


@triton.jit
def _causal_packed(output, boundaries, BOUNDARIES: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    sequence = _interval(boundaries, row, BOUNDARIES)
    start = tl.load(boundaries + sequence)
    _causal_row(output + row.to(tl.int64) * WIDTH, start, row, WIDTH, 256)


@triton.jit
def _select_packed(
    scores,
    output,
    boundaries,
    offsets,
    n_pools,
    first,
    out_stride,
    TOP_K: tl.constexpr,
    POOL: tl.constexpr,
    BOUNDARIES: tl.constexpr,
    POOL_BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    EMIT_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    position = first + row
    sequence = _interval(boundaries, position, BOUNDARIES)
    start = tl.load(boundaries + sequence)
    stop = tl.load(boundaries + sequence + 1)
    if stop - start <= TOP_K:
        _causal_row(
            output + row.to(tl.int64) * out_stride,
            start,
            position,
            TOP_K + POOL - 1,
            256,
        )
    else:
        pool_start = tl.load(offsets + sequence)
        _select_row(
            scores,
            output,
            n_pools,
            TOP_K // POOL,
            first - start,
            start,
            out_stride,
            POOL,
            POOL_BLOCK,
            ROW_BLOCK,
            EMIT_BLOCK,
            pool_start,
            row,
        )


def _wide_selection(scores, result, boundaries, offsets, first, pool, top_k):
    """Bounded torch.topk fallback for score rows too wide for register selection."""
    rows, device = scores.shape[0], scores.device
    positions = torch.arange(first, first + rows, device=device)
    sequence = torch.searchsorted(boundaries, positions, right=True) - 1
    start, stop = boundaries[sequence], boundaries[sequence + 1]
    pool_start = offsets[sequence]
    values, selected = scores.topk(top_k // pool, dim=-1)
    lane = torch.arange(pool, device=device)
    indices = (
        start[:, None, None] + (selected - pool_start[:, None])[..., None] * pool + lane
    ).masked_fill(~values.isfinite()[..., None], -1)
    result[:, :top_k] = indices.flatten(1).to(torch.int32)
    if pool > 1:
        visible = positions - start + 1
        tail = visible % pool
        lane = torch.arange(pool - 1, device=device)
        indices = (positions + 1 - tail)[:, None] + lane
        result[:, top_k:] = indices.masked_fill(lane >= tail[:, None], -1).to(
            torch.int32
        )
    columns = torch.arange(top_k + pool - 1, device=device)
    causal = start[:, None] + columns
    causal = causal.masked_fill(causal > positions[:, None], -1).to(torch.int32)
    result.copy_(torch.where((stop - start <= top_k)[:, None], causal, result))


def pooled_topk(q, key, gates, head_weights, position_bias, boundaries, top_k):
    tokens, heads, width = q.shape
    pool = position_bias.shape[0]
    result = torch.full(
        (tokens, top_k + pool - 1), -1, device=q.device, dtype=torch.int32
    )
    if not tokens:
        return result
    boundaries = boundaries.contiguous()
    if tokens <= top_k:
        _causal_packed[(tokens,)](
            result,
            boundaries,
            BOUNDARIES=boundaries.numel(),
            WIDTH=result.shape[1],
        )
        return result
    offsets = F.pad(((boundaries[1:] - boundaries[:-1]) // pool).cumsum(0), (1, 0))
    n_pools = tokens // pool
    half = (torch.bfloat16, torch.float16)
    dtype = q.dtype if q.dtype == key.dtype and q.dtype in half else torch.float32
    queries = q if q.dtype == dtype and q.stride(-1) == 1 else q.to(dtype).contiguous()
    width_block = triton.next_power_of_2(max(width, 16))
    if pool == 1:
        pooled = key.to(dtype).contiguous()
    else:
        # Rounding happens in the original key dtype, before the scorer's cast.
        pooled = key.new_zeros((n_pools, width))
        _pool_keys[(n_pools,)](
            key,
            gates,
            position_bias,
            boundaries,
            offsets,
            pooled,
            *key.stride(),
            *gates.stride(),
            *position_bias.stride(),
            POOL=pool,
            WIDTH=width,
            BOUNDARIES=boundaries.numel(),
            POOL_BLOCK=triton.next_power_of_2(pool),
            WIDTH_BLOCK=width_block,
        )
        pooled = pooled.to(dtype)
    block_k = shared._KEY_TILE_BYTES // (width_block * queries.element_size())
    block_k = max(16, min(shared._BLOCK_K, 1 << (block_k.bit_length() - 1)))
    chunk = max(
        shared._BLOCK_Q,
        shared.SCORE_CHUNK_BYTES // (4 * n_pools) // shared._BLOCK_Q * shared._BLOCK_Q,
    )
    # One reusable workspace; the last chunk is a leading-row view of the same buffer.
    workspace = torch.empty(
        (min(chunk, tokens), n_pools), device=q.device, dtype=torch.float32
    )
    for first in range(0, tokens, chunk):
        last = min(first + chunk, tokens)
        rows = last - first
        scores = workspace[:rows]
        _pooled_scores_kernel[
            (triton.cdiv(rows, shared._BLOCK_Q), triton.cdiv(n_pools, block_k))
        ](
            queries[first:last],
            pooled,
            head_weights[first:last],
            scores,
            rows,
            n_pools,
            first,
            queries.stride(0),
            queries.stride(1),
            head_weights.stride(0),
            head_weights.stride(1),
            width**-0.5,
            heads**-0.5,
            POOL=pool,
            HEADS=heads,
            WIDTH=width,
            WIDTH_BLOCK=width_block,
            IEEE=dtype is torch.float32,
            BLOCK_Q=shared._BLOCK_Q,
            BLOCK_K=block_k,
            boundaries_ptr=boundaries,
            pool_offsets_ptr=offsets,
            BOUNDARIES=boundaries.numel(),
            TOP_K=top_k,
            num_warps=shared._NUM_WARPS,
        )
        if n_pools <= shared._SELECT_POOLS_LIMIT:
            row_block = triton.next_power_of_2(n_pools)
            _select_packed[(rows,)](
                scores,
                result[first:last],
                boundaries,
                offsets,
                n_pools,
                first,
                result.stride(0),
                TOP_K=top_k,
                POOL=pool,
                BOUNDARIES=boundaries.numel(),
                POOL_BLOCK=triton.next_power_of_2(pool),
                ROW_BLOCK=row_block,
                EMIT_BLOCK=min(shared._EMIT_BLOCK, row_block),
                num_warps=max(4, row_block // 2048),
            )
        else:
            _wide_selection(
                scores, result[first:last], boundaries, offsets, first, pool, top_k
            )
    return result
