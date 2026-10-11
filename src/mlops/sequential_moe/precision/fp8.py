"""DeepSeek-V3-style E4M3 GEMMs with block and token-group scaling.

Weights: 128x128 blocks. Activations: 1x128 along each GEMM reduction axis.
Each 128-element dot-product partial is scaled and added into FP32 registers.
The same kernel supports per-row operands for the full-training weight gradient.
"""

import torch
import triton
import triton.language as tl

BLOCK = 128


@triton.jit
def _weight_quant(X, Q, SCALE, M: tl.constexpr, N: tl.constexpr):
    r = tl.program_id(0) * 128 + tl.arange(0, 128)
    c = tl.program_id(1) * 128 + tl.arange(0, 128)
    mask = (r[:, None] < M) & (c[None, :] < N)
    x = tl.load(X + r[:, None] * N + c[None, :], mask, other=0.0).to(tl.float32)
    maximum = tl.max(tl.max(tl.abs(x), 1), 0)
    scale = tl.div_rn(tl.maximum(maximum, 1e-4), 448.0)
    quantized = tl.div_rn(x, scale).to(tl.float8e4nv)
    tl.store(Q + r[:, None] * N + c[None, :], quantized, mask)
    tl.store(SCALE + tl.program_id(0) * triton.cdiv(N, 128) + tl.program_id(1), scale)


def pack_weight(x):
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("FP8 weight quantization requires a contiguous matrix")
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (triton.cdiv(x.shape[0], 128), triton.cdiv(x.shape[1], 128)),
        device=x.device,
        dtype=torch.float32,
    )
    _weight_quant[scales.shape](x, q, scales, *x.shape, num_warps=8)
    return q, scales


def dequantize(q, scales):
    """Diagnostic/checkpoint conversion; never called by the execution backend."""
    return (
        q.float()
        * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[
            : q.shape[0], : q.shape[1]
        ]
    )


@triton.jit
def _row_quant(
    X,
    Q,
    SCALE,
    M: tl.constexpr,
    K: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    QS0: tl.constexpr,
    SS0: tl.constexpr,
):
    row = tl.program_id(0) * 8 + tl.arange(0, 8)
    col = tl.program_id(1) * 128 + tl.arange(0, 128)
    x = tl.load(
        X + row[:, None] * XS0 + col[None, :] * XS1,
        (row[:, None] < M) & (col[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    maximum = tl.max(tl.abs(x), 1)
    scale = tl.div_rn(tl.maximum(maximum, 1e-4), 448.0)
    quantized = tl.div_rn(x, scale[:, None]).to(tl.float8e4nv)
    tl.store(
        Q + row[:, None] * QS0 + col[None, :],
        quantized,
        (row[:, None] < M) & (col[None, :] < K),
    )
    tl.store(SCALE + row * SS0 + tl.program_id(1), scale, row < M)


class Quantizer:
    """Fixed maximum shape, including direct transpose-and-quantize for Wgrad."""

    def __init__(self, max_rows, max_columns, device):
        self.data = torch.empty(
            max_rows, max_columns, device=device, dtype=torch.float8_e4m3fn
        )
        self.scales = torch.empty(
            max_rows, triton.cdiv(max_columns, 128), device=device, dtype=torch.float32
        )

    def __call__(self, x):
        if x.shape[0] > self.data.shape[0] or x.shape[1] > self.data.shape[1]:
            raise ValueError("Input exceeds FP8 quantizer capacity")
        _row_quant[(triton.cdiv(x.shape[0], 8), triton.cdiv(x.shape[1], 128))](
            x,
            self.data,
            self.scales,
            *x.shape,
            *x.stride(),
            self.data.stride(0),
            self.scales.stride(0),
        )
        return self.data[: x.shape[0], : x.shape[1]], self.scales[
            : x.shape[0], : triton.cdiv(x.shape[1], 128)
        ]


@triton.jit
def _gemm(
    A,
    B,
    AS,
    BS,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    B0: tl.constexpr,
    AS0: tl.constexpr,
    BS0: tl.constexpr,
    B_GROUP: tl.constexpr,
    ADD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, 128)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for block in range(triton.cdiv(K, 128)):
        k = block * 128 + inner
        a = tl.load(
            A + row[:, None] * A0 + k[None, :],
            (row[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + col[None, :] * B0 + k[:, None],
            (col[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        sa = tl.load(AS + row * AS0 + block, row < M, other=0.0)
        sb = tl.load(BS + (col // B_GROUP) * BS0 + block, col < N, other=0.0)
        partial = tl.dot(a, b, out_dtype=tl.float32, max_num_imprecise_acc=128)
        accumulator += partial * sa[:, None] * sb[None, :]
    offset = row[:, None] * N + col[None, :]
    mask = (row[:, None] < M) & (col[None, :] < N)
    if ADD:
        accumulator += tl.load(OUT + offset, mask, other=0.0).to(tl.float32)
    tl.store(OUT + offset, accumulator, mask)


def gemm(a, b, out, *, weight_blocks=True, add=False):
    aq, asc = a
    bq, bsc = b
    return _gemm[(triton.cdiv(out.shape[0], 32), triton.cdiv(out.shape[1], 64))](
        aq,
        bq,
        asc,
        bsc,
        out,
        *out.shape,
        aq.shape[1],
        aq.stride(0),
        bq.stride(0),
        asc.stride(0),
        bsc.stride(0),
        128 if weight_blocks else 1,
        add,
        32,
        64,
        num_warps=4,
    )
