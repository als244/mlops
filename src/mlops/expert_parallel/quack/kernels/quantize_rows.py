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
"""

import torch
import triton
import triton.language as tl


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


def warps(block):
    """Warps per program for a row block of ``block`` elements."""
    return 4 if block <= 2048 else 8


def quantize_rows_fp8(value, *, poison_padding=False, gather=None):
    """Return E4M3 values and FP32 amax/448 descales; logical shape is exact.

    Of ``value [rows, N]``'s rows, or of the rows ``gather`` names, in that
    order, when given. Only the physical row pitch is aligned to 16 bytes, as
    SM90 FP8 GEMM requires. The source must be contiguous along columns;
    callers own any transpose.
    """
    if value.ndim != 2 or value.stride(1) != 1 or not value.is_cuda:
        raise ValueError("Expected a CUDA matrix contiguous along columns")
    rows, cols = value.shape
    if cols == 0:
        raise ValueError("Cannot quantize an empty reduction dimension")
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
