"""NVFP4 packing and cuBLAS GEMMs, without full-precision weight expansion.

Weights use 16x16 scales so their packed transpose represents identical values.
Activations use 1x16 scales. Both use a float32 global scale, E4M3 block scales,
and round-to-nearest-even E2M1 payloads. No stochastic recipe is implied.
"""

import torch
import triton
import triton.language as tl
from torch.nn.functional import ScalingType, SwizzleType


def scale_size(rows, cols):
    return triton.cdiv(rows, 128) * 128 * triton.cdiv(cols, 64) * 4


@triton.jit
def _amax(X, PARTIAL, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + i, i < N, other=0).to(tl.float32)
    tl.store(PARTIAL + tl.program_id(0), tl.max(tl.abs(value)))


@triton.jit
def _global_scale(PARTIAL, GLOBAL, N, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    value = tl.load(PARTIAL + i, i < N, other=0)
    # Positive floor only handles all-zero/small matrices. It does not move
    # a scale to the host or change scratch capacity.
    tl.store(GLOBAL, tl.maximum(tl.div_rn(tl.max(value), 6.0 * 448.0), 1.0e-12))


@triton.jit
def _encode(x):
    v = tl.abs(x)
    q = (v > 0.25).to(tl.int32) + (v >= 0.75).to(tl.int32)
    q += (v > 1.25).to(tl.int32) + (v >= 1.75).to(tl.int32)
    q += (v > 2.5).to(tl.int32) + (v >= 3.5).to(tl.int32)
    q += (v > 5.0).to(tl.int32)
    return (q | tl.where(x < 0, 8, 0)).to(tl.uint8)


@triton.jit
def _swizzled_offset(row, col, COL_BLOCKS: tl.constexpr):
    # cuBLAS scale layout: [row/128, col/4, row%32, (row%128)/32, col%4].
    return (
        ((row // 128 * COL_BLOCKS + col // 4) * 32 + row % 32) * 4 + row % 128 // 32
    ) * 4 + col % 4


@triton.jit
def _quantize(
    X,
    Q,
    SCALE,
    GLOBAL,
    M: tl.constexpr,
    K: tl.constexpr,
    WEIGHT: tl.constexpr,
    SWIZZLED: tl.constexpr = True,
):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    pair = tl.arange(0, 8)
    col = tl.program_id(1) * 16 + pair * 2
    a = tl.load(
        X + row[:, None] * K + col[None, :],
        (row[:, None] < M) & (col[None, :] < K),
        other=0,
    ).to(tl.float32)
    b = tl.load(
        X + row[:, None] * K + col[None, :] + 1,
        (row[:, None] < M) & (col[None, :] + 1 < K),
        other=0,
    ).to(tl.float32)
    global_scale = tl.load(GLOBAL)
    maximum = tl.max(tl.maximum(tl.abs(a), tl.abs(b)), 1)
    if WEIGHT:
        maximum = tl.full((16,), tl.max(maximum), tl.float32)
    scale = tl.div_rn(maximum, 6.0 * global_scale).to(tl.float8e4nv)
    decoded = scale.to(tl.float32) * global_scale
    denominator = tl.where(decoded > 0, decoded, 1.0)
    qa = _encode(tl.div_rn(a, denominator[:, None]))
    qb = _encode(tl.div_rn(b, denominator[:, None]))
    packed = qa | (qb << 4)
    tl.store(
        Q + row[:, None] * (K // 2) + col[None, :] // 2,
        packed,
        (row[:, None] < M) & (col[None, :] < K),
    )
    if WEIGHT:
        tl.store(
            SCALE + tl.program_id(0) * (K // 16) + tl.program_id(1),
            tl.sum(scale.to(tl.float32), 0) / 16,
        )
    else:
        offset = (
            _swizzled_offset(row, tl.program_id(1), triton.cdiv(K, 64))
            if SWIZZLED
            else row * (K // 16) + tl.program_id(1)
        )
        tl.store(SCALE + offset, scale, row < M)


@triton.jit
def _weight_scales(
    SRC,
    OUT,
    M: tl.constexpr,
    K: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cols = triton.cdiv(K, 64) * 4
    row, col = i // cols, i % cols
    if TRANSPOSE:
        source_offset = col * (M // 16) + row // 16
    else:
        source_offset = (row // 16) * (K // 16) + col
    value = tl.load(SRC + source_offset, (row < M) & (col < K // 16), other=0.0)
    dst = _swizzled_offset(row, col, triton.cdiv(K, 64))
    tl.store(OUT + dst, value, i < N)


def weight_scales(scales, rows, cols, transpose=False):
    out = torch.empty(scale_size(rows, cols), dtype=scales.dtype, device=scales.device)
    _weight_scales[(triton.cdiv(out.numel(), 512),)](
        scales, out, rows, cols, transpose, out.numel(), 512
    )
    return out


@triton.jit
def _transpose(
    SRC, OUT, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    new_row, old_row = i // (M // 2), (i % (M // 2)) * 2
    lo = tl.load(SRC + old_row * (K // 2) + new_row // 2, i < N, other=0)
    hi = tl.load(SRC + (old_row + 1) * (K // 2) + new_row // 2, i < N, other=0)
    shift = (new_row % 2) * 4
    packed = ((lo >> shift) & 15) | (((hi >> shift) & 15) << 4)
    tl.store(OUT + i, packed, i < N)


def transpose_packed(data):
    rows, cols = data.shape[0], data.shape[1] * 2
    out = torch.empty((cols, rows // 2), dtype=data.dtype, device=data.device)
    _transpose[(triton.cdiv(out.numel(), 512),)](
        data, out, rows, cols, out.numel(), 512
    )
    return out


class Quantizer:
    """Capacity-sized storage reused for every chunk and projection of this width."""

    def __init__(self, capacity, width, device, *, weight=False):
        self.capacity, self.width, self.weight = capacity, width, weight
        self.data = torch.empty(
            (capacity, width // 2), dtype=torch.uint8, device=device
        )
        shape = (
            (capacity // 16, width // 16) if weight else (scale_size(capacity, width),)
        )
        self.scales = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
        self.global_scale = torch.empty((), dtype=torch.float32, device=device)
        self.partial = torch.empty(
            triton.cdiv(capacity * width, 4096), dtype=torch.float32, device=device
        )

    def __call__(self, x):
        count = triton.cdiv(x.numel(), 4096)
        _amax[(count,)](x, self.partial, x.numel(), 4096)
        _global_scale[(1,)](
            self.partial, self.global_scale, count, triton.next_power_of_2(count)
        )
        _quantize[(triton.cdiv(x.shape[0], 16), self.width // 16)](
            x,
            self.data,
            self.scales,
            self.global_scale,
            x.shape[0],
            self.width,
            self.weight,
        )
        scales = (
            self.scales
            if self.weight
            else self.scales[: scale_size(x.shape[0], self.width)]
        )
        return self.data[: x.shape[0]], scales, self.global_scale


def pack_weight(weight, *, block_rows=16):
    """Pack one matrix; 1x16 is the standard checkpoint storage layout."""
    if not weight.is_contiguous() or any(d % 32 for d in weight.shape):
        raise ValueError(
            "NVFP4 prototype weights require contiguous dimensions divisible by 32"
        )
    if block_rows == 16:
        quantizer = Quantizer(*weight.shape, weight.device, weight=True)
        return quantizer(weight)
    if block_rows != 1:
        raise ValueError("NVFP4 block_rows must be 1 or 16")
    m, k = weight.shape
    data = torch.empty((m, k // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((m, k // 16), dtype=torch.float8_e4m3fn, device=weight.device)
    global_scale = torch.empty((), dtype=torch.float32, device=weight.device)
    count = triton.cdiv(weight.numel(), 4096)
    partial = torch.empty(count, dtype=torch.float32, device=weight.device)
    _amax[(count,)](weight, partial, weight.numel(), 4096)
    _global_scale[(1,)](partial, global_scale, count, triton.next_power_of_2(count))
    _quantize[(triton.cdiv(m, 16), k // 16)](
        weight, data, scales, global_scale, m, k, False, False
    )
    return data, scales, global_scale


def gemm(a, weight, out):
    aq, asc, aglobal = a
    bq, bsc, bglobal = weight
    recipe = [int(ScalingType.BlockWise1x16), int(ScalingType.TensorWise)]
    swizzle = [int(SwizzleType.SWIZZLE_32_4_4), int(SwizzleType.NO_SWIZZLE)]
    torch.ops.aten._scaled_mm_v2.out(
        aq.view(torch.float4_e2m1fn_x2),
        bq.view(torch.float4_e2m1fn_x2).T,
        [asc, aglobal],
        recipe,
        swizzle,
        [bsc, bglobal],
        recipe,
        swizzle,
        None,
        out.dtype,
        out=out,
    )


def unpack_weight(data, scales, global_scale):
    """Diagnostic materialization only; the layer never calls this."""
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6], device=data.device, dtype=torch.float32
    )
    nibbles = torch.stack((data & 15, data >> 4), -1).reshape(data.shape[0], -1).long()
    decoded = values[nibbles & 7] * torch.where(nibbles & 8 != 0, -1.0, 1.0)
    expanded = scales.float().repeat_interleave(16, 0).repeat_interleave(16, 1)
    return decoded * expanded * global_scale
