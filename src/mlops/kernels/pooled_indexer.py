"""Pooled causal index selection; boundaries are explicit CPU data.

``pooled_topk`` scores one chunk of queries against every pooled key of its
sequence in a single Triton launch, applying the scale, the ReLU and the head
weights in registers, then a second launch per query reads its FP32 scores and
writes its selection: the cutoff from a binary search over the float bit
patterns, ties to the lowest pool, the selected pools expanded to their tokens,
-1 where a pool is not visible, and the query's incomplete tail pool. Scores
accumulate in FP32 from the inputs' own dtype: 16-bit queries and keys use the
tensor cores, FP32 ones the IEEE path.
"""

from itertools import pairwise

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


# The FP32 scores of one query chunk, [rows, pools], stay within this size.
SCORE_CHUNK_BYTES = 256 << 20
# 32 queries by 128 pools with 8 warps measured fastest for 32 heads of width
# 128; a wider or FP32 key tile is kept within this many bytes of shared memory.
_BLOCK_Q = 32
_BLOCK_K = 128
_KEY_TILE_BYTES = 48 << 10
_NUM_WARPS = 8
# A query's scores are searched from registers up to this many pools, 64 keys
# per thread; a wider row does not fit an SM's registers and uses torch.topk.
_SELECT_POOLS_LIMIT = 16384
# The selection is then written in blocks of this many pools, read back from
# cache, so the search alone holds a whole row in registers.
_EMIT_BLOCK = 1024
# Keys order like the floats they come from: -inf's key, and one below it for
# the padding beyond a row's pools.
_NEGATIVE_INFINITY_KEY = -2139095041
_PADDING_KEY = -2147483648


if triton is not None:

    @triton.jit
    def _pooled_scores_kernel(
        q_ptr,
        pooled_ptr,
        weights_ptr,
        scores_ptr,
        rows,
        n_pools,
        first_position,
        stride_qt,
        stride_qh,
        stride_wt,
        stride_wh,
        query_scale,
        head_scale,
        POOL: tl.constexpr,
        HEADS: tl.constexpr,
        WIDTH: tl.constexpr,
        WIDTH_BLOCK: tl.constexpr,
        IEEE: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Score one [BLOCK_Q, BLOCK_K] tile of queries against pooled keys.

        scores[row, pool] = sum over heads of
            head_scale * weight[row, head]
            * relu(query_scale * q[row, head, :] . pooled[pool, :])
        for the complete pools the query at ``first_position + row`` can see,
        -inf elsewhere. The grid is (query blocks, pool blocks); a tile no row
        of it can see is written as -inf without loading its operands.
        """
        query_block = tl.program_id(0)
        pool_block = tl.program_id(1)
        row = query_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
        pool = pool_block * BLOCK_K + tl.arange(0, BLOCK_K)
        row_ok = row < rows
        pool_ok = pool < n_pools
        position = first_position + row
        visible = (
            (((pool + 1) * POOL - 1)[None, :] <= position[:, None])
            & row_ok[:, None]
            & pool_ok[None, :]
        )
        scores = tl.full((BLOCK_Q, BLOCK_K), float("-inf"), tl.float32)
        last_position = (
            first_position + tl.minimum(query_block * BLOCK_Q + BLOCK_Q, rows) - 1
        )
        if (pool_block * BLOCK_K + 1) * POOL - 1 <= last_position:
            lane = tl.arange(0, WIDTH_BLOCK)
            lane_ok = lane < WIDTH
            keys = tl.load(
                pooled_ptr + pool[:, None] * WIDTH + lane[None, :],
                mask=pool_ok[:, None] & lane_ok[None, :],
                other=0.0,
            )
            acc = tl.zeros((BLOCK_Q, BLOCK_K), tl.float32)
            for head in tl.static_range(HEADS):
                queries = tl.load(
                    q_ptr + row[:, None] * stride_qt + head * stride_qh + lane[None, :],
                    mask=row_ok[:, None] & lane_ok[None, :],
                    other=0.0,
                )
                weight = (
                    tl.load(
                        weights_ptr + row * stride_wt + head * stride_wh,
                        mask=row_ok,
                        other=0.0,
                    ).to(tl.float32)
                    * head_scale
                )
                if IEEE:
                    logits = tl.dot(queries, tl.trans(keys), input_precision="ieee")
                else:
                    logits = tl.dot(queries, tl.trans(keys))
                acc += tl.maximum(logits * query_scale, 0.0) * weight[:, None]
            scores = tl.where(visible, acc, float("-inf"))
        tl.store(
            scores_ptr + row[:, None] * n_pools + pool[None, :],
            scores,
            mask=row_ok[:, None] & pool_ok[None, :],
        )

    @triton.jit
    def _keys(scores_ptr, col, valid):
        """Signed keys ordered like the scores; padding ranks below -inf."""
        bits = tl.load(scores_ptr + col, mask=valid, other=float("-inf")).to(
            tl.int32, bitcast=True
        )
        return tl.where(valid, bits ^ ((bits >> 31) & 0x7FFFFFFF), -2147483648)

    @triton.jit
    def _select_pools_kernel(
        scores_ptr,
        out_ptr,
        n_pools,
        keep,
        first_position,
        start,
        out_stride,
        POOL: tl.constexpr,
        POOL_BLOCK: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
        EMIT_BLOCK: tl.constexpr,
    ):
        """Write one query's selection from its row of scores.

        The ``keep`` best-scoring pools, ties to the lowest pool, expanded to
        their tokens in pool order, -1 for a pool the query cannot see, then
        the query's incomplete tail pool. The cutoff is the largest key at
        least ``keep`` keys reach, found by a binary search over the 32-bit
        keys of the register-resident row with one count per step; the
        selection is then written block by block from the cached row, so the
        search's keys are the only whole-row tensor alive.
        """
        row = tl.program_id(0)
        row_scores = scores_ptr + row.to(tl.int64) * n_pools
        col = tl.arange(0, ROW_BLOCK)
        key = _keys(row_scores, col, col < n_pools)
        low = tl.full((), -2147483648, tl.int64)
        high = tl.full((), 2147483647, tl.int64)
        for _ in range(32):
            mid = low + ((high - low + 1) >> 1)
            reached = tl.sum((key >= mid.to(tl.int32)).to(tl.int32)) >= keep
            low = tl.where(reached, mid, low)
            high = tl.where(reached, high, mid - 1)
        threshold = low.to(tl.int32)
        needed = keep - tl.sum((key > threshold).to(tl.int32))
        out_row = out_ptr + row.to(tl.int64) * out_stride
        greater_before = tl.full((), 0, tl.int32)
        equal_before = tl.full((), 0, tl.int32)
        for block in tl.static_range(0, ROW_BLOCK, EMIT_BLOCK):
            col = block + tl.arange(0, EMIT_BLOCK)
            key = _keys(row_scores, col, col < n_pools)
            greater = (key > threshold).to(tl.int32)
            equal = (key == threshold).to(tl.int32)
            # one scan carries both running counts: greater below bit 16, equal above
            counts = tl.cumsum(greater + (equal << 16), 0)
            rank = equal_before + (counts >> 16) - equal
            selected = (greater == 1) | ((equal == 1) & (rank < needed))
            slot = (
                greater_before + (counts & 0xFFFF) - greater + tl.minimum(rank, needed)
            )
            # A pool the query cannot see scores -inf and is selected only when
            # fewer than keep pools are visible; it comes out as -1 padding.
            base = tl.where(key == -2139095041, -1, col * POOL + start)
            for lane in tl.static_range(POOL):
                tl.store(
                    out_row + slot * POOL + lane,
                    tl.where(base < 0, -1, base + lane),
                    mask=selected,
                )
            last = tl.max(counts, 0)
            greater_before += last & 0xFFFF
            equal_before += last >> 16
        if POOL > 1:
            visible = first_position + row + 1
            tail_count = visible % POOL
            tail_lane = tl.arange(0, POOL_BLOCK)
            tail = tl.where(
                tail_lane < tail_count, start + visible - tail_count + tail_lane, -1
            )
            tl.store(out_row + keep * POOL + tail_lane, tail, mask=tail_lane < POOL - 1)


def _validated_ends(boundaries, tokens):
    ends = boundaries.tolist()
    if (
        not ends
        or ends[0] != 0
        or ends[-1] != tokens
        or any(a >= b for a, b in pairwise(ends))
    ):
        raise ValueError("CPU boundaries must strictly cover all packed tokens")
    return ends


def _pooled_keys(key, gates, position_bias, start, n_pools):
    """Pool each complete group of keys under its gated softmax, in the key dtype."""
    pool, width = position_bias.shape
    count = n_pools * pool
    logits = (
        gates[start : start + count].reshape(n_pools, pool, width).float()
        + position_bias.float()
    )
    probabilities = logits.softmax(1).to(key.dtype)
    return (
        probabilities * key[start : start + count].reshape(n_pools, pool, width)
    ).sum(1)


def _select_every_causal_key(result, start, stop, rows_per_pass):
    """Below the key budget every causal key is selected; nothing is ranked."""
    keys = torch.arange(start, stop, device=result.device, dtype=torch.int32)
    for first in range(start, stop, rows_per_pass):
        last = min(first + rows_per_pass, stop)
        rows = torch.arange(first, last, device=result.device)
        result[first:last, : stop - start] = (
            keys[None, :]
            .expand(last - first, -1)
            .masked_fill(keys[None, :] > rows[:, None], -1)
        )


def _select_tail(result, start, first, last, pool, column):
    """Select each query's visible incomplete pool, -1 padded, at ``column``."""
    device = result.device
    visible_count = torch.arange(first - start + 1, last - start + 1, device=device)
    tail_count = visible_count % pool
    tail_offset = torch.arange(pool - 1, device=device)
    tail = start + (visible_count - tail_count)[:, None] + tail_offset
    tail = tail.masked_fill(tail_offset >= tail_count[:, None], -1)
    result[first:last, column : column + pool - 1] = tail.to(torch.int32)


def pooled_topk(q, key, gates, head_weights, position_bias, boundaries, top_k):
    """Select pooled causal keys; see the module docstring for the mechanism.

    Queries and pooled keys meet in one dtype: their own when they share a
    16-bit one, else FP32, which every input reaches exactly. The scores are
    the FP32 accumulation of exact products, so the selection matches FP32
    PyTorch arithmetic except at a cutoff between scores closer than rounding.
    A row's selection is written in pool order by the select kernel, or in
    score order by ``torch.topk`` for rows wider than ``_SELECT_POOLS_LIMIT``.
    """
    if triton is None:
        raise RuntimeError("the pooled indexer requires Triton")
    pool, width = position_bias.shape
    tokens, heads, _ = q.shape
    ends = _validated_ends(boundaries, tokens)
    result = torch.full(
        (tokens, top_k + pool - 1), -1, device=q.device, dtype=torch.int32
    )
    half = (torch.bfloat16, torch.float16)
    dtype = q.dtype if q.dtype == key.dtype and q.dtype in half else torch.float32
    queries = q if q.dtype == dtype and q.stride(-1) == 1 else q.to(dtype).contiguous()
    width_block = triton.next_power_of_2(max(width, 16))
    block_k = _KEY_TILE_BYTES // (width_block * queries.element_size())
    block_k = max(16, min(_BLOCK_K, 1 << (block_k.bit_length() - 1)))
    query_scale = width**-0.5
    head_scale = heads**-0.5
    lanes = torch.arange(pool, device=q.device, dtype=torch.int32)
    for start, stop in pairwise(ends):
        length = stop - start
        if length <= top_k:
            _select_every_causal_key(
                result, start, stop, max(1, SCORE_CHUNK_BYTES // (4 * length))
            )
            continue
        n_pools = length // pool
        keep = min(top_k // pool, n_pools)
        pooled = _pooled_keys(key, gates, position_bias, start, n_pools)
        pooled = pooled.to(dtype).contiguous()
        chunk = max(_BLOCK_Q, SCORE_CHUNK_BYTES // (4 * n_pools) // _BLOCK_Q * _BLOCK_Q)
        for first in range(start, stop, chunk):
            last = min(first + chunk, stop)
            rows = last - first
            scores = torch.empty((rows, n_pools), device=q.device, dtype=torch.float32)
            grid = (triton.cdiv(rows, _BLOCK_Q), triton.cdiv(n_pools, block_k))
            _pooled_scores_kernel[grid](
                queries[first:last],
                pooled,
                head_weights[first:last],
                scores,
                rows,
                n_pools,
                first - start,
                queries.stride(0),
                queries.stride(1),
                head_weights.stride(0),
                head_weights.stride(1),
                query_scale,
                head_scale,
                POOL=pool,
                HEADS=heads,
                WIDTH=width,
                WIDTH_BLOCK=width_block,
                IEEE=dtype is torch.float32,
                BLOCK_Q=_BLOCK_Q,
                BLOCK_K=block_k,
                num_warps=_NUM_WARPS,
            )
            if n_pools <= _SELECT_POOLS_LIMIT:
                row_block = triton.next_power_of_2(n_pools)
                _select_pools_kernel[(rows,)](
                    scores,
                    result[first:last],
                    n_pools,
                    keep,
                    first - start,
                    start,
                    result.stride(0),
                    POOL=pool,
                    POOL_BLOCK=triton.next_power_of_2(pool),
                    ROW_BLOCK=row_block,
                    EMIT_BLOCK=min(_EMIT_BLOCK, row_block),
                    num_warps=max(4, row_block // 2048),
                )
                continue
            values, pools = scores.topk(keep, dim=-1)
            indices = pools.to(torch.int32)[..., None] * pool + lanes + start
            indices = indices.masked_fill(~values.isfinite()[..., None], -1)
            result[first:last, : keep * pool] = indices.flatten(-2)
            if pool > 1:
                _select_tail(result, start, first, last, pool, keep * pool)
    return result
