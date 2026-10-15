"""E4M3 expert weights with one FP32 scale per 128x128 block, and the grouped
GEMM that reads them in either orientation.

A block-scaled weight ``[E, out, in]`` is stored once: E4M3 values and a
scale per 128x128 block (``value ~ data * scale``, the layout of DeepSeek-style
FP8 checkpoints). Because a block covers both axes, the same bytes serve the
forward GEMM, which reduces over ``in`` (``y = x @ W[e]ᵀ``, the weight
K-major), and the input gradient, which reduces over ``out`` (``dx = dy @
W[e]``, the weight MN-major); no transposed copy is kept. The GEMM walks K in
128-wide chunks, one block row or column at a time, and promotes each chunk's
FP32 partial by its block's scale before adding it into the accumulator.

The rows are expert-sorted, ``offsets [E + 1]`` bounding each expert's, and
may be gathered from a larger operand by ``gather``. The rows' operand is FP8
with one scale per row (applied once, after the K loop) or bf16. ``math``
picks the products: ``"fp8"`` (E4M3 x E4M3) or ``"bf16"`` (both tiles taken
to bf16, exactly, for a bf16 product). The output may be written with its
gate and up halves interleaved, ``out[:, 2j] = gate_j, out[:, 2j + 1] =
up_j``, the order the SwiGLU row kernels read, and the rows' operand may be
read in that order while the weight's K axis holds gate rows then up rows.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

BLOCK = 128
E4M3_MAX = 448.0


def _blocks(width):
    return -(-width // BLOCK)


def scale_shape(experts, out, inner):
    return experts, _blocks(out), _blocks(inner)


if triton is not None:

    @triton.jit
    def _quantize_blocks(X, Q, S, OUT, IN, BLOCKS_OUT, BLOCKS_IN):
        expert = tl.program_id(0).to(tl.int64)
        block_row, block_col = tl.program_id(1), tl.program_id(2)
        rows = block_row * 128 + tl.arange(0, 128)
        cols = block_col * 128 + tl.arange(0, 128)
        mask = (rows[:, None] < OUT) & (cols[None, :] < IN)
        offset = expert * OUT * IN + rows[:, None].to(tl.int64) * IN + cols[None, :]
        x = tl.load(X + offset, mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.max(tl.abs(x), 1), 0)
        scale = tl.maximum(tl.div_rn(amax, 448.0), 1e-12)
        quantized = tl.clamp(tl.div_rn(x, scale), -448.0, 448.0)
        tl.store(Q + offset, quantized.to(tl.float8e4nv), mask)
        tl.store(S + (expert * BLOCKS_OUT + block_row) * BLOCKS_IN + block_col, scale)


def quantize_blocks(weight):
    """``(data, scale)`` of ``weight [E, out, in]`` (bf16 or fp32): E4M3 values
    and the FP32 scale of each 128x128 block, ``amax / 448`` (at least
    1e-12); by a kernel on CUDA from compute capability 8.9, where Triton has
    E4M3, elsewhere by the same equations."""
    if weight.ndim != 3:
        raise ValueError("Expected an [experts, out, in] weight")
    experts, out, inner = weight.shape
    if (
        weight.is_cuda
        and triton is not None
        and torch.cuda.get_device_capability(weight.device) >= (8, 9)
    ):
        source = weight.contiguous()
        data = torch.empty(
            source.shape, dtype=torch.float8_e4m3fn, device=source.device
        )
        scale = torch.empty(
            scale_shape(experts, out, inner), dtype=torch.float32, device=source.device
        )
        if source.numel():
            _quantize_blocks[scale.shape](
                source, data, scale, out, inner, scale.shape[1], scale.shape[2]
            )
        return data, scale
    value = weight.float()
    padded = _padded(value)
    blocks = padded.view(experts, padded.shape[1] // BLOCK, BLOCK, -1, BLOCK)
    scale = (blocks.abs().amax(dim=(2, 4)) / E4M3_MAX).clamp_min(1e-12)
    data = (value / _expanded(scale, out, inner)).clamp(-E4M3_MAX, E4M3_MAX)
    return data.to(torch.float8_e4m3fn), scale


def dequantize_blocks(data, scale, *, dtype=torch.float32):
    """``data * scale`` per block, ``[E, out, in]`` at ``dtype``."""
    _experts, out, inner = data.shape
    return (data.float() * _expanded(scale.float(), out, inner)).to(dtype)


def _padded(value):
    experts, out, inner = value.shape
    rows, cols = _blocks(out) * BLOCK, _blocks(inner) * BLOCK
    if (rows, cols) == (out, inner):
        return value
    padded = value.new_zeros(experts, rows, cols)
    padded[:, :out, :inner] = value
    return padded


def _expanded(scale, out, inner):
    expanded = scale.repeat_interleave(BLOCK, 1).repeat_interleave(BLOCK, 2)
    return expanded[:, :out, :inner]


if triton is not None:

    @triton.jit
    def _grouped_block_gemm(
        A,
        A_SCALE,
        GATHER,
        W,
        W_SCALE,
        C,
        OUT,
        OFFSETS,
        TILES,
        EXPERTS,
        N,
        K,
        A_STRIDE,
        BLOCKS_N,
        BLOCKS_K,
        COL_TILES,
        A_FP8: tl.constexpr,
        FP8_MATH: tl.constexpr,
        TRANSPOSED: tl.constexpr,
        HAS_GATHER: tl.constexpr,
        HAS_C: tl.constexpr,
        INTERLEAVE_OUT: tl.constexpr,
        INTERLEAVED_A: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
    ):
        # One program per (row tile, column tile), in expert order: an
        # expert's row tiles run back to back for each column tile, so its
        # weight column and its rows stay in L2 while it is worked on.
        program = tl.program_id(0)
        if program >= tl.load(TILES + EXPERTS) * COL_TILES:
            return
        # The expert holding this program: the last ``e`` with
        # ``TILES[e] * COL_TILES <= program``.
        low = program * 0
        high = low + EXPERTS
        while low + 1 < high:
            middle = (low + high) // 2
            if tl.load(TILES + middle) * COL_TILES <= program:
                low = middle
            else:
                high = middle
        expert = low.to(tl.int64)
        start = tl.load(TILES + expert)
        tiles = tl.load(TILES + expert + 1) - start
        local = program - start * COL_TILES
        col_tile = local // tiles
        first = tl.load(OFFSETS + expert) + (local % tiles) * BM
        rows = first + tl.arange(0, BM)
        row_mask = rows < tl.load(OFFSETS + expert + 1)
        rows = rows.to(tl.int64)
        if HAS_GATHER:
            sources = tl.load(GATHER + rows, row_mask, other=0).to(tl.int64)
        else:
            sources = rows
        cols = col_tile * BN + tl.arange(0, BN)
        col_mask = cols < N
        # BN divides 128, so a column tile lies in one block column.
        block_n = (col_tile * BN) // 128
        inner = tl.arange(0, 128)
        half_k = K // 2
        accumulator = tl.zeros((BM, BN), dtype=tl.float32)
        for block_k in range(BLOCKS_K):
            k = block_k * 128 + inner
            k_mask = k < K
            if INTERLEAVED_A:
                # The weight's K axis holds gate rows, then up rows; the rows'
                # operand holds gate_j and up_j side by side.
                a_cols = tl.where(k < half_k, 2 * k, 2 * (k - half_k) + 1)
            else:
                a_cols = k
            a = tl.load(
                A + sources[:, None] * A_STRIDE + a_cols[None, :],
                row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            if TRANSPOSED:
                # dx = dy @ W[e]: W[e] is [K, N], read along its rows (MN-major).
                w = tl.load(
                    W + expert * K * N + k[:, None].to(tl.int64) * N + cols[None, :],
                    k_mask[:, None] & col_mask[None, :],
                    other=0.0,
                )
                scale = tl.load(
                    W_SCALE + (expert * BLOCKS_K + block_k) * BLOCKS_N + block_n
                )
            else:
                # y = x @ W[e]ᵀ: W[e] is [N, K], read along K (K-major).
                w = tl.load(
                    W + expert * N * K + cols[None, :].to(tl.int64) * K + k[:, None],
                    k_mask[:, None] & col_mask[None, :],
                    other=0.0,
                )
                scale = tl.load(
                    W_SCALE + (expert * BLOCKS_N + block_n) * BLOCKS_K + block_k
                )
            if FP8_MATH:
                partial = tl.dot(a, w, out_dtype=tl.float32, max_num_imprecise_acc=128)
            else:
                partial = tl.dot(
                    a.to(tl.bfloat16), w.to(tl.bfloat16), out_dtype=tl.float32
                )
            accumulator += partial * scale
        if A_FP8:
            # One scale per operand row, constant along K: applied once.
            accumulator *= tl.load(A_SCALE + sources, row_mask, other=0.0)[:, None]
        if INTERLEAVE_OUT:
            half_n = N // 2
            out_cols = tl.where(cols < half_n, 2 * cols, 2 * (cols - half_n) + 1)
        else:
            out_cols = cols
        mask = row_mask[:, None] & col_mask[None, :]
        offset = rows[:, None] * N + out_cols[None, :]
        if HAS_C:
            accumulator += tl.load(C + offset, mask, other=0.0).to(tl.float32)
        tl.store(OUT + offset, accumulator.to(OUT.dtype.element_ty), mask)


def tile_offsets(offsets, block_rows):
    """``[E + 1]`` cumulative row-tile counts of the experts' row ranges."""
    counts = offsets[1:] - offsets[:-1]
    tiles = torch.zeros_like(offsets)
    tiles[1:] = torch.cumsum(
        torch.div(counts + block_rows - 1, block_rows, rounding_mode="floor"), 0
    )
    return tiles


def grouped_block_gemm(
    rows,
    data,
    scale,
    offsets,
    *,
    row_scale=None,
    gather=None,
    transposed=False,
    addend=None,
    interleave_out=False,
    interleaved_rows=False,
    math="fp8",
    out_dtype=torch.bfloat16,
    block_rows=None,
    block_cols=None,
    num_warps=None,
    num_stages=None,
):
    """Per expert ``e``: ``rows[e] @ W[e]ᵀ`` (``W[e] = data[e] * scale[e]``,
    ``[N, K]``), or with ``transposed`` ``rows[e] @ W[e]`` (``W[e]`` then
    ``[K, N]``), plus ``addend`` when given, at ``out_dtype``.

    ``rows`` is E4M3 with ``row_scale [R]`` (one per row) or bf16; ``gather``
    picks each output row's operand row, else the rows are already
    expert-sorted. ``offsets [E + 1]`` bounds each expert's output rows.
    ``interleave_out`` writes the gate and up halves of N side by side;
    ``interleaved_rows`` reads ``rows`` with them side by side while the
    weight's K axis holds gate rows then up rows. ``math`` is ``"fp8"`` (both
    operands E4M3) or ``"bf16"``. ``block_rows``, ``block_cols`` (64 or 128),
    ``num_warps`` and ``num_stages`` override the launch configuration.
    """
    if triton is None:
        raise RuntimeError("the grouped block-scaled GEMM needs triton")
    experts = data.shape[0]
    n, k = (data.shape[2], data.shape[1]) if transposed else data.shape[1:]
    count = rows.shape[0] if gather is None else gather.shape[0]
    a_fp8 = rows.dtype == torch.float8_e4m3fn
    if a_fp8 != (row_scale is not None):
        raise ValueError("E4M3 rows need their row scales; bf16 rows take none")
    if math not in ("fp8", "bf16"):
        raise ValueError(f"math must be 'fp8' or 'bf16', not {math!r}")
    if math == "fp8" and not a_fp8:
        raise ValueError("fp8 math needs E4M3 rows")
    if rows.shape[1] != k or rows.stride(1) != 1:
        raise ValueError(f"Expected rows [*, {k}] with contiguous columns")
    if (interleave_out and n % 2) or (interleaved_rows and k % 2):
        raise ValueError("An interleaved gate/up axis has an even width")
    out = torch.empty(count, n, dtype=out_dtype, device=rows.device)
    if addend is not None and (addend.shape != out.shape or not addend.is_contiguous()):
        raise ValueError("The addend must be a contiguous [rows, N] tensor")
    if count == 0 or n == 0:
        return out
    bm = block_rows or (64 if count <= 64 * experts else 128)
    # Measured on an H100 at GLM's shapes (E=288, H=4096, I=2048, 8K tokens):
    # 64-wide column tiles, four stages for a projection, two for an input
    # gradient's MN-major reads.
    bn = block_cols or 64
    if 128 % bn:
        raise ValueError("block_cols must divide 128")
    tiles = tile_offsets(offsets, bm)
    col_tiles = triton.cdiv(n, bn)
    grid = ((triton.cdiv(count, bm) + experts) * col_tiles,)
    _grouped_block_gemm[grid](
        rows,
        row_scale if row_scale is not None else rows,
        gather if gather is not None else offsets,
        data,
        scale,
        addend if addend is not None else out,
        out,
        offsets,
        tiles,
        experts,
        n,
        k,
        rows.stride(0),
        scale.shape[2] if transposed else scale.shape[1],
        scale.shape[1] if transposed else scale.shape[2],
        col_tiles,
        A_FP8=a_fp8,
        FP8_MATH=math == "fp8",
        TRANSPOSED=transposed,
        HAS_GATHER=gather is not None,
        HAS_C=addend is not None,
        INTERLEAVE_OUT=interleave_out,
        INTERLEAVED_A=interleaved_rows,
        BM=bm,
        BN=bn,
        num_warps=num_warps or (8 if bm == 128 else 4),
        num_stages=num_stages or (2 if transposed else 4),
    )
    return out


__all__ = [
    "BLOCK",
    "dequantize_blocks",
    "grouped_block_gemm",
    "quantize_blocks",
    "scale_shape",
    "tile_offsets",
]
