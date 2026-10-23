"""SM90 pooled index selection using cuDNN CuTe scoring and radix top-k.

The cuDNN 1.30 SM90 kernel supports FP32 head weights, although its public
BF16 wrapper restricts them to BF16. Launch the typed kernel directly to retain
the operation's FP32 weights. No checkpoint or activation downcast is required.
Only compiled callables are cached; every tensor/workspace is call-local.
"""

from itertools import pairwise

import torch
import triton
import triton.language as tl

from . import pooled_indexer as common

# Scores + the radix kernel's two int32 work arrays + selected pool indices.
WORKSPACE_BYTES = 256 << 20
_COMPILED = {}


def _score(q, key, weights, offset, ratio):
    import cutlass
    from cudnn.deepseek_sparse_attention.indexer_forward.indexer_fwd_sm90 import (
        IndexerForwardSm90,
    )
    from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
    from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
    from cudnn.deepseek_sparse_attention.utils.tensor_conversion import (
        get_broadcast_dims,
        to_cute_tensor,
    )
    from cutlass import cute

    out = torch.full(
        (1, q.shape[1], key.shape[1]), -torch.inf, device=q.device, dtype=torch.float32
    )
    stream = resolve_stream(None)
    signature = (
        q.device.index,
        q.dtype,
        weights.dtype,
        q.shape[2],
        ratio,
        get_broadcast_dims(q),
        get_broadcast_dims(key),
        get_broadcast_dims(weights),
    )
    compiled = _COMPILED.get(signature)
    scale = (q.shape[2] * q.shape[3]) ** -0.5
    if compiled is None:
        w_dtype = (
            cutlass.Float32 if weights.dtype == torch.float32 else cutlass.BFloat16
        )
        kernel = IndexerForwardSm90(
            cutlass.BFloat16,
            w_dtype,
            head_dim=128,
            qhead_per_kvhead=q.shape[2],
            ratio=ratio,
            is_varlen=False,
            use_unchecked_qh64=False,
            use_unchecked_qh64_masked=False,
            compute_lse=False,
        )
        compiled = cute.compile(
            kernel,
            to_cute_tensor(q),
            to_cute_tensor(key),
            to_cute_tensor(weights),
            None,
            None,
            to_cute_tensor(out),
            None,
            cutlass.Int32(1),
            cutlass.Int32(q.shape[1]),
            cutlass.Int32(key.shape[1]),
            cutlass.Float32(scale),
            None,
            None,
            to_cute_tensor(offset, leading_dim=0),
            stream,
            options=compile_options(),
        )
        _COMPILED[signature] = compiled
    compiled(
        q,
        key,
        weights,
        None,
        None,
        out,
        None,
        cutlass.Int32(1),
        cutlass.Int32(q.shape[1]),
        cutlass.Int32(key.shape[1]),
        cutlass.Float32(scale),
        None,
        None,
        offset,
        stream,
    )
    current = torch.cuda.current_stream(q.device)
    for tensor in (q, key, weights, offset, out):
        tensor.record_stream(current)
    return out.squeeze(0)


@triton.jit
def _metadata(out, rows, offset_index, first, POOL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + i, (first + i + 1) // POOL, i < rows)
    if tl.program_id(0) == 0:
        tl.store(out + offset_index, first)


@triton.jit
def _expand(
    indices,
    out,
    keep: tl.constexpr,
    first,
    start,
    stride: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK: tl.constexpr,
    TAIL_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.arange(0, BLOCK)
    pool_id = tl.load(indices + row.to(tl.int64) * keep + slot, slot < keep, other=-1)
    visible = (first + row + 1) // POOL
    valid = (pool_id >= 0) & (pool_id < visible)
    for lane in tl.static_range(POOL):
        value = tl.where(valid, start + pool_id * POOL + lane, -1)
        tl.store(
            out + row.to(tl.int64) * stride + slot * POOL + lane,
            value,
            slot < keep,
        )
    if POOL > 1:
        lane = tl.arange(0, TAIL_BLOCK)
        count = (first + row + 1) % POOL
        value = tl.where(lane < count, start + visible * POOL + lane, -1)
        tl.store(
            out + row.to(tl.int64) * stride + keep * POOL + lane,
            value,
            lane < POOL - 1,
        )


def pooled_topk(q, key, gates, head_weights, position_bias, boundaries, top_k):
    from cudnn.deepseek_sparse_attention import indexer_top_k_wrapper

    pool, width = position_bias.shape
    ends = common._validated_ends(boundaries, q.shape[0])
    result = torch.full(
        (q.shape[0], top_k + pool - 1), -1, device=q.device, dtype=torch.int32
    )
    queries = q.contiguous()
    weights = head_weights.contiguous()
    for start, stop in pairwise(ends):
        length = stop - start
        if length <= top_k:
            common._select_every_causal_key(
                result, start, stop, max(1, WORKSPACE_BYTES // (4 * length))
            )
            continue
        pools = length // pool
        keep = min(top_k // pool, pools)
        keys = (
            common._pooled_keys(key, gates, position_bias, start, pools)
            .reshape(1, pools, 1, width)
            .contiguous()
        )
        # cuDNN radix uses two int32 arrays as large as its FP32 input.
        bytes_per_row = 12 * pools + 4 * keep + 4
        chunk = max(32, WORKSPACE_BYTES // bytes_per_row // 32 * 32)
        for first in range(0, length, chunk):
            last = min(first + chunk, length)
            rows = last - first
            # CuTe assumes 16-byte alignment, including this int32 scalar view.
            offset_index = triton.cdiv(rows, 4) * 4
            metadata = torch.empty(offset_index + 1, device=q.device, dtype=torch.int32)
            _metadata[(triton.cdiv(rows, 256),)](
                metadata, rows, offset_index, first, pool, 256
            )
            scores = _score(
                queries[start + first : start + last].unsqueeze(0),
                keys,
                weights[start + first : start + last].unsqueeze(0),
                metadata[offset_index:],
                pool,
            )
            indices = indexer_top_k_wrapper(
                scores, metadata[:rows], top_k=keep, next_n=1, return_val=False
            )["indices"]
            _expand[(rows,)](
                indices,
                result[start + first : start + last],
                keep,
                first,
                start,
                result.stride(0),
                pool,
                triton.next_power_of_2(keep),
                triton.next_power_of_2(pool),
            )
            del scores, indices, metadata
    return result
