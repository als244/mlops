"""Fused implementation of QuACK's per-row E4M3 quantization equations."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_rows(
    X, Q, S, N: tl.constexpr, XS: tl.constexpr, QS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * XS + col, col < N, other=0).to(tl.float32)
    amax = tl.max(tl.abs(value), axis=0)
    # PyTorch scalar division multiplies by the FP32 reciprocal here.
    scale = tl.maximum(amax * (1.0 / 448.0), 1e-12)
    quantized = tl.minimum(tl.maximum(tl.div_rn(value, scale), -448.0), 448.0)
    tl.store(Q + row * QS + col, quantized, col < N)
    tl.store(S + row, scale)


def quantize_rows_fp8(value, *, poison_padding=False):
    """Return E4M3 values and FP32 amax/448 descales; logical shape is exact.

    Only physical row pitch is aligned to 16 bytes, as SM90 FP8 GEMM requires.
    The source must be contiguous along columns; callers own any transpose.
    """
    if value.ndim != 2 or value.stride(1) != 1 or not value.is_cuda:
        raise ValueError("Expected a CUDA matrix contiguous along columns")
    rows, cols = value.shape
    if cols == 0:
        raise ValueError("Cannot quantize an empty reduction dimension")
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
            cols,
            value.stride(0),
            pitch,
            block,
            num_warps=4 if block <= 2048 else 8,
        )
    return data[:, :cols], scales
