"""QuACK's per-row E4M3 quantization equations as fused Triton kernels.

One program per row: ``scale = max(amax / 448, 1e-12)`` in FP32 (the
reciprocal multiply PyTorch's scalar division performs), RN division,
clipping to ±448 and the E4M3 rounding of the store. Rows may be gathered by
index as they are read, so a routed gather and its quantization are one pass
over memory.

Row counts, widths and pitches are runtime arguments: Triton specializes on
their 16-alignment only, so a new row count, gather or pitch never compiles
another kernel. The block size, the one compile-time constant besides the
gather flag, follows the row width.

Rows bound for the block-scaled transposing GEMM
(``expert_parallel/quack/experts/block_fp8_gemm.py``) are stored in that
GEMM's K order (``transposed_gemm=True``): its in-kernel weight transpose
delivers the weight's K rows permuted within every aligned group of 16, and the
row operand must carry the same permutation. A gate/up row interleaved as
``gate_j, up_j`` can be read as gate columns then up columns
(``gate_up=True``), the row order of a ``w13`` weight.
"""

import torch
import triton
import triton.language as tl

# Within every aligned group of 16 stored columns, stored position ``p`` holds
# column ``16 * group + TRANSPOSED_GEMM_ORDER[p]`` of the logical row (the K
# order the transposing GEMM's ldmatrix.trans / prmt / stmatrix sequence
# produces; the GEMM module's tests derive it from the kernel's copy layouts).
TRANSPOSED_GEMM_ORDER = (0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15)


@triton.jit
def quantize_row(value):
    """``(quantized, scale)`` of one row's FP32 values by the equations above;
    masked lanes must be zero so they leave the maximum alone."""
    amax = tl.max(tl.abs(value), axis=0)
    scale = tl.maximum(amax * (1.0 / 448.0), 1e-12)
    quantized = tl.minimum(tl.maximum(tl.div_rn(value, scale), -448.0), 448.0)
    return quantized, scale


@triton.jit
def _quantize_rows(
    X, Q, S, GATHER, N, XS, QS, HAS_GATHER: tl.constexpr, BLOCK: tl.constexpr
):
    # Large routed matrices can exceed 2**31 elements. Widen before multiplying
    # by either stride so reads and writes cannot wrap to negative addresses.
    row = tl.program_id(0).to(tl.int64)
    if HAS_GATHER:
        source = tl.load(GATHER + row).to(tl.int64)
    else:
        source = row
    col = tl.arange(0, BLOCK)
    mask = col < N
    value = tl.load(X + source * XS + col, mask, other=0).to(tl.float32)
    quantized, scale = quantize_row(value)
    tl.store(Q + row * QS + col, quantized, mask)
    tl.store(S + row, scale)


@triton.jit
def _quantize_rows_ordered(
    X,
    Q,
    S,
    GATHER,
    N,
    XS,
    QS,
    HAS_GATHER: tl.constexpr,
    TRANSPOSED_GEMM: tl.constexpr,
    GATE_UP: tl.constexpr,
    WHOLE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # `_quantize_rows` with each stored column taken from its source column:
    # the row's maximum, and so its scale, does not depend on the order.
    row = tl.program_id(0).to(tl.int64)
    if HAS_GATHER:
        source = tl.load(GATHER + row).to(tl.int64)
    else:
        source = row
    col = tl.arange(0, BLOCK)
    if WHOLE:
        # The row fills the block: read it contiguously, reorder in registers.
        value = tl.load(X + source * XS + col).to(tl.float32)
        if GATE_UP:
            # [gate_0, up_0, gate_1, ...] -> gate columns, then up columns.
            value = tl.reshape(tl.trans(tl.reshape(value, [BLOCK // 2, 2])), [BLOCK])
        if TRANSPOSED_GEMM:
            # Stored position 4a + 2b + c of each 16 holds column 8b + 2a + c.
            value = tl.reshape(value, [BLOCK // 16, 2, 4, 2])
            value = tl.reshape(tl.permute(value, [0, 2, 1, 3]), [BLOCK])
        quantized, scale = quantize_row(value)
        tl.store(Q + row * QS + col, quantized)
    else:
        mask = col < N
        logical = col
        if TRANSPOSED_GEMM:
            # TRANSPOSED_GEMM_ORDER[p] = 2 * (p // 4) + p % 2 + 8 * (p % 4 // 2)
            p = col % 16
            logical = col - p + 2 * (p // 4) + p % 2 + 8 * (p % 4 // 2)
        if GATE_UP:
            half = N // 2
            logical = tl.where(logical < half, 2 * logical, 2 * (logical - half) + 1)
        value = tl.load(X + source * XS + logical, mask, other=0).to(tl.float32)
        quantized, scale = quantize_row(value)
        tl.store(Q + row * QS + col, quantized, mask)
    tl.store(S + row, scale)


def warps(block):
    """Warps per program for a row block of ``block`` elements."""
    return 4 if block <= 2048 else 8


def source_columns(width, *, transposed_gemm=False, gate_up=False, device=None):
    """The source column of every stored column of a ``width``-wide row, as
    `quantize_rows_fp8` orders them (``int64 [width]``)."""
    _check_order(width, transposed_gemm, gate_up)
    logical = torch.arange(width, device=device)
    if transposed_gemm:
        order = torch.tensor(TRANSPOSED_GEMM_ORDER, device=device)
        logical = logical - logical % 16 + order[logical % 16]
    if gate_up:
        half = width // 2
        logical = torch.where(logical < half, 2 * logical, 2 * (logical - half) + 1)
    return logical


def _check_order(width, transposed_gemm, gate_up):
    if transposed_gemm and width % 16:
        raise ValueError("The transposed GEMM's K order needs a multiple of 16 columns")
    if gate_up and width % 2:
        raise ValueError("An interleaved gate/up row has an even width")


def quantize_rows_fp8(
    value, *, poison_padding=False, gather=None, transposed_gemm=False, gate_up=False
):
    """Return E4M3 values and FP32 amax/448 descales; logical shape is exact.

    Of ``value [rows, N]``'s rows, or of the rows ``gather`` names, in that
    order, when given. Only the physical row pitch is aligned to 16 bytes, as
    SM90 FP8 GEMM requires. The source must be contiguous along columns;
    callers own any transpose.

    ``transposed_gemm`` stores each row's columns in the K order of the
    block-scaled transposing GEMM (`TRANSPOSED_GEMM_ORDER` within every
    aligned group of 16; ``N`` a multiple of 16). ``gate_up`` reads a row
    whose gate and up halves are interleaved (``value[:, 2j]`` the gate,
    ``value[:, 2j + 1]`` the up column) as its gate columns then its up
    columns. `source_columns` gives the resulting column map.
    """
    if value.ndim != 2 or value.stride(1) != 1 or not value.is_cuda:
        raise ValueError("Expected a CUDA matrix contiguous along columns")
    rows, cols = value.shape
    if cols == 0:
        raise ValueError("Cannot quantize an empty reduction dimension")
    _check_order(cols, transposed_gemm, gate_up)
    if gather is not None:
        if (
            gather.ndim != 1
            or gather.device != value.device
            or gather.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Expected a vector of row indices on the source's device")
        rows = gather.shape[0]
    pitch = triton.cdiv(cols, 16) * 16
    data = torch.empty((rows, pitch), device=value.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(rows, device=value.device, dtype=torch.float32)
    if poison_padding:
        data.view(torch.uint8).fill_(0x7F)
    if rows:
        block = triton.next_power_of_2(cols)
        if transposed_gemm or gate_up:
            _quantize_rows_ordered[(rows,)](
                value,
                data,
                scales,
                value if gather is None else gather,
                cols,
                value.stride(0),
                pitch,
                gather is not None,
                transposed_gemm,
                gate_up,
                block == cols,
                block,
                num_warps=warps(block),
            )
        else:
            _quantize_rows[(rows,)](
                value,
                data,
                scales,
                value if gather is None else gather,
                cols,
                value.stride(0),
                pitch,
                gather is not None,
                block,
                num_warps=warps(block),
            )
    return data[:, :cols], scales


def quantize_rows_fp8_reference(
    value, *, gather=None, transposed_gemm=False, gate_up=False
):
    """`quantize_rows_fp8` by the same equations in PyTorch, on any device:
    contiguous ``(data [rows, N], scales [rows])``."""
    rows = value if gather is None else value[gather.long()]
    rows = rows.float()
    amax = rows.abs().amax(dim=1)
    scales = torch.clamp_min(amax * (1.0 / 448.0), 1e-12)
    quantized = torch.clamp(rows / scales[:, None], -448.0, 448.0)
    columns = source_columns(
        rows.shape[1],
        transposed_gemm=transposed_gemm,
        gate_up=gate_up,
        device=rows.device,
    )
    return quantized[:, columns].to(torch.float8_e4m3fn).contiguous(), scales
