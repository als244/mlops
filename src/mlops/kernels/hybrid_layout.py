"""Layout-aware normalization and frozen-weight packed convolution gradients.

These kernels preserve the FLA adapter's arithmetic/rounding contract without
materializing contiguous inputs or convolution-backward intermediates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _l2_strided_fwd(
    X,
    Y,
    R,
    N,
    EPS: tl.constexpr,
    D: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    SD: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    remainder = row
    offset = tl.full((BT,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += (remainder % SHAPE[axis]) * STRIDES[axis]
        remainder //= SHAPE[axis]
    col = tl.arange(0, BD)
    mask = (row[:, None] < N) & (col[None, :] < D)
    x = tl.load(X + offset[:, None] + col[None, :] * SD, mask, other=0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, 1) + EPS)
    tl.store(Y + row[:, None] * D + col[None, :], x * rstd[:, None], mask)
    tl.store(R + row, rstd, row < N)


def strided_l2_forward(x, eps):
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rstd = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    rows = x.numel() // x.shape[-1]
    if rows:
        _l2_strided_fwd[(triton.cdiv(rows, 32),)](
            x,
            output,
            rstd,
            rows,
            float(eps),
            x.shape[-1],
            tuple(x.shape[:-1]),
            tuple(x.stride()[:-1]),
            x.stride(-1),
            BT=32,
            BD=triton.next_power_of_2(x.shape[-1]),
            num_warps=4,
        )
    return output, rstd


@triton.jit
def _causal_silu_input_bwd(
    X,
    WGT,
    DY,
    DX,
    CU,
    CHUNKS,
    T,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    SY0: tl.constexpr,
    SY1: tl.constexpr,
    SW0: tl.constexpr,
    SW1: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    PACKED: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BR: tl.constexpr,
):
    ci, ti = tl.program_id(0), tl.program_id(1)
    if PACKED:
        seq = tl.load(CHUNKS + 2 * ti)
        tile = tl.load(CHUNKS + 2 * ti + 1)
        begin = tl.load(CU + seq).to(tl.int64)
        end = tl.load(CU + seq + 1).to(tl.int64)
    else:
        tile = ti
        begin = tl.full((), 0, tl.int64)
        end = T
    col = ci * BC + tl.arange(0, BC)
    local = tile * BT + tl.arange(0, BR)
    row = begin + local
    active = (tl.arange(0, BR) < BT + W - 1) & (row < end)
    pre = tl.full((BR, BC), 0, tl.float32)
    # Reconstruct a tile and its forward halo, keeping preactivations on chip.
    for tap in tl.static_range(W):
        source = row + tap - W + 1
        x = tl.load(
            X + source[:, None] * SX0 + col[None, :] * SX1,
            (active & (source >= begin))[:, None] & (col[None, :] < D),
            other=0,
        ).to(tl.float32)
        weight = tl.load(WGT + col * SW0 + tap * SW1, col < D, other=0).to(tl.float32)
        pre += x * weight[None, :]
    # FLA's backward recomputes and stores preactivation in the input dtype.
    # Keep that rounding boundary even though this kernel avoids the store.
    pre = pre.to(X.dtype.element_ty).to(tl.float32)
    sig = tl.sigmoid(pre)
    dy = tl.load(
        DY + row[:, None] * SY0 + col[None, :] * SY1,
        active[:, None] & (col[None, :] < D),
        other=0,
    ).to(tl.float32)
    dpre = dy * sig * (1.0 + pre * (1.0 - sig))
    dx = tl.full((BT, BC), 0, tl.float32)
    out_local = tl.arange(0, BT)
    for tap in tl.static_range(W):
        index = (out_local[:, None] + tap) + tl.full((1, BC), 0, tl.int32)
        shifted = tl.gather(dpre, index, axis=0)
        weight = tl.load(WGT + col * SW0 + (W - 1 - tap) * SW1, col < D, other=0).to(
            tl.float32
        )
        dx += shifted * weight[None, :]
    out_row = begin + tile * BT + out_local
    tl.store(
        DX + out_row[:, None] * D + col[None, :],
        dx,
        (out_row[:, None] < end) & (col[None, :] < D),
    )


def causal_silu_input_backward(grad_output, x, weight, cumulative, chunk_indices=None):
    """Input gradient only; no global preactivation or partial weight gradients."""
    packed = cumulative.numel() != 0
    if packed and chunk_indices is None:
        from fla.ops.utils import prepare_chunk_indices

        chunk_indices = prepare_chunk_indices(cumulative, 64)
    tiles = len(chunk_indices) if packed else triton.cdiv(x.shape[0], 64)
    dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if tiles:
        _causal_silu_input_bwd[(triton.cdiv(x.shape[1], 32), tiles)](
            x,
            weight,
            grad_output,
            dx,
            cumulative,
            chunk_indices,
            x.shape[0],
            *x.stride(),
            *grad_output.stride(),
            weight.stride(0),
            weight.stride(-1),
            x.shape[1],
            weight.shape[-1],
            packed,
            BT=64,
            BC=32,
            BR=triton.next_power_of_2(64 + weight.shape[-1] - 1),
            num_warps=4,
        )
    return dx
