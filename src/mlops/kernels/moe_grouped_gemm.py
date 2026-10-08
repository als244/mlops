"""Standalone deterministic Triton grouped GEMM for expert segments."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


def _tiles(rows, columns, reduction, dtype, device):
    """Bound tiles by matrix dimensions and the device's shared-memory limit.

    Low-rank projections must not pay for a 256-column tile. The same policy
    also keeps FP32 and smaller shared-memory devices within their launch limit.
    No token counts are copied from the device.
    """
    bm = min(128, max(16, triton.next_power_of_2(max(1, rows))))
    bn = min(256, max(16, triton.next_power_of_2(max(1, columns))))
    bk = min(64, max(16, triton.next_power_of_2(max(1, reduction))))
    element_size = torch.finfo(dtype).bits // 8
    budget = (
        torch.cuda.get_device_properties(device).shared_memory_per_block_optin - 8192
    )
    stages = 3
    # Reserve space for pipeline operands and compiler scratch. Two stages
    # suffice when three would exceed the per-block resource budget.
    if stages * (bm + bn) * bk * element_size > budget:
        stages = 2
    while stages * (bm + bn) * bk * element_size > budget:
        if bn > bm:
            bn //= 2
        elif bm > 16:
            bm //= 2
        else:
            bk //= 2
    warps = 4 if bm * bn <= 8192 else 8
    return bm, bn, bk, warps, stages


if triton is not None:

    @triton.jit
    def _grouped_mm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        offsets_ptr,
        tile_prefix_ptr,
        n_dim,
        k_dim,
        experts,
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        FP32_PRECISION: tl.constexpr,
    ):
        program_m = tl.program_id(0)
        program_n = tl.program_id(1)
        total_tiles = tl.load(tile_prefix_ptr + experts)
        if program_m >= total_tiles:
            return
        low = 0
        high = experts
        while low < high:
            middle = (low + high + 1) // 2
            if tl.load(tile_prefix_ptr + middle) <= program_m:
                low = middle
            else:
                high = middle - 1
        expert = low
        segment_end = tl.load(offsets_ptr + expert + 1).to(tl.int64)
        row_start = (
            tl.load(offsets_ptr + expert).to(tl.int64)
            + (program_m - tl.load(tile_prefix_ptr + expert)).to(tl.int64) * BM
        )
        rows = row_start + tl.arange(0, BM).to(tl.int64)
        row_mask = rows < segment_end
        columns = (program_n * BN + tl.arange(0, BN)).to(tl.int64)
        column_mask = columns < n_dim
        accumulator = tl.zeros((BM, BN), dtype=tl.float32)
        b_base = b_ptr + expert.to(tl.int64) * stride_be
        for offset in range(0, k_dim, BK):
            reduction = (offset + tl.arange(0, BK)).to(tl.int64)
            reduction_mask = reduction < k_dim
            a = tl.load(
                a_ptr + rows[:, None] * stride_am + reduction[None, :] * stride_ak,
                mask=row_mask[:, None] & reduction_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_base + reduction[:, None] * stride_bk + columns[None, :] * stride_bn,
                mask=reduction_mask[:, None] & column_mask[None, :],
                other=0.0,
            )
            accumulator = tl.dot(a, b, accumulator, input_precision=FP32_PRECISION)
        tl.store(
            c_ptr + rows[:, None] * stride_cm + columns[None, :] * stride_cn,
            accumulator.to(c_ptr.dtype.element_ty),
            mask=row_mask[:, None] & column_mask[None, :],
        )

    @triton.jit
    def _grouped_wgrad_kernel(
        x_ptr,
        grad_ptr,
        output_ptr,
        offsets_ptr,
        k_dim,
        n_dim,
        stride_xm,
        stride_xk,
        stride_gm,
        stride_gn,
        stride_oe,
        stride_ok,
        stride_on,
        BI: tl.constexpr,
        BJ: tl.constexpr,
        BK: tl.constexpr,
        FP32_PRECISION: tl.constexpr,
    ):
        expert = tl.program_id(0).to(tl.int64)
        rows_start = tl.load(offsets_ptr + expert).to(tl.int64)
        rows_end = tl.load(offsets_ptr + expert + 1).to(tl.int64)
        inner = (tl.program_id(1) * BI + tl.arange(0, BI)).to(tl.int64)
        outer = (tl.program_id(2) * BJ + tl.arange(0, BJ)).to(tl.int64)
        inner_mask = inner < k_dim
        outer_mask = outer < n_dim
        accumulator = tl.zeros((BI, BJ), dtype=tl.float32)
        for row in range(rows_start, rows_end, BK):
            reduction = row + tl.arange(0, BK).to(tl.int64)
            reduction_mask = reduction < rows_end
            x = tl.load(
                x_ptr + reduction[None, :] * stride_xm + inner[:, None] * stride_xk,
                mask=reduction_mask[None, :] & inner_mask[:, None],
                other=0.0,
            )
            grad = tl.load(
                grad_ptr + reduction[:, None] * stride_gm + outer[None, :] * stride_gn,
                mask=reduction_mask[:, None] & outer_mask[None, :],
                other=0.0,
            )
            accumulator = tl.dot(x, grad, accumulator, input_precision=FP32_PRECISION)
        pointer = (
            output_ptr
            + expert * stride_oe
            + inner[:, None] * stride_ok
            + outer[None, :] * stride_on
        )
        tl.store(
            pointer,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=inner_mask[:, None] & outer_mask[None, :],
        )


def _tile_prefix(offsets, block_rows):
    counts = offsets[1:] - offsets[:-1]
    tiles = (counts + (block_rows - 1)) // block_rows
    prefix = torch.zeros_like(offsets)
    prefix[1:].copy_(torch.cumsum(tiles, 0).to(offsets.dtype))
    return prefix


def _native_torch(x, weight, offsets, transpose=False):
    output_width = weight.shape[1] if transpose else weight.shape[2]
    output = torch.zeros((x.shape[0], output_width), dtype=x.dtype, device=x.device)
    row_experts = torch.searchsorted(
        offsets[1:].contiguous(),
        torch.arange(x.shape[0], device=x.device, dtype=offsets.dtype),
        right=True,
    )
    for expert in range(weight.shape[0]):
        product = x @ (weight[expert].T if transpose else weight[expert])
        output.copy_(torch.where((row_experts == expert).unsqueeze(1), product, output))
    return output


def grouped_mm_forward(x, weight, offsets):
    if triton is None or not x.is_cuda:
        return _native_torch(x, weight, offsets)
    output = torch.empty((x.shape[0], weight.shape[2]), dtype=x.dtype, device=x.device)
    bm, bn, bk, warps, stages = _tiles(
        max(1, x.shape[0] // weight.shape[0]),
        weight.shape[2],
        weight.shape[1],
        x.dtype,
        x.device,
    )
    prefix = _tile_prefix(offsets, bm)
    grid_0 = triton.cdiv(x.shape[0], bm) + weight.shape[0]
    _grouped_mm_kernel[(grid_0, triton.cdiv(weight.shape[2], bn))](
        x,
        weight,
        output,
        offsets,
        prefix,
        weight.shape[2],
        weight.shape[1],
        weight.shape[0],
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=warps,
        num_stages=stages,
        FP32_PRECISION=(
            "ieee" if torch.get_float32_matmul_precision() == "highest" else "tf32"
        ),
    )
    return output


def grouped_mm_dgrad(grad, weight, offsets):
    if triton is None or not grad.is_cuda:
        return _native_torch(grad, weight, offsets, transpose=True)
    output = torch.empty(
        (grad.shape[0], weight.shape[1]), dtype=grad.dtype, device=grad.device
    )
    bm, bn, bk, warps, stages = _tiles(
        max(1, grad.shape[0] // weight.shape[0]),
        weight.shape[1],
        weight.shape[2],
        grad.dtype,
        grad.device,
    )
    prefix = _tile_prefix(offsets, bm)
    grid_0 = triton.cdiv(grad.shape[0], bm) + weight.shape[0]
    _grouped_mm_kernel[(grid_0, triton.cdiv(weight.shape[1], bn))](
        grad,
        weight,
        output,
        offsets,
        prefix,
        weight.shape[1],
        weight.shape[2],
        weight.shape[0],
        grad.stride(0),
        grad.stride(1),
        weight.stride(0),
        weight.stride(2),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=warps,
        num_stages=stages,
        FP32_PRECISION=(
            "ieee" if torch.get_float32_matmul_precision() == "highest" else "tf32"
        ),
    )
    return output


def grouped_mm_wgrad(x, grad, offsets, weight_shape, dtype=None):
    """Each expert's ``x.T @ grad`` over its rows, summed at fp32 and returned
    at ``dtype`` -- ``x``'s when ``None``."""
    output = torch.empty(
        weight_shape, dtype=x.dtype if dtype is None else dtype, device=x.device
    )
    if triton is not None and x.is_cuda:
        bi, bj, bk, warps, stages = _tiles(
            weight_shape[1],
            weight_shape[2],
            max(1, x.shape[0] // weight_shape[0]),
            x.dtype,
            x.device,
        )
        _grouped_wgrad_kernel[
            (
                weight_shape[0],
                triton.cdiv(weight_shape[1], bi),
                triton.cdiv(weight_shape[2], bj),
            )
        ](
            x,
            grad,
            output,
            offsets,
            weight_shape[1],
            weight_shape[2],
            x.stride(0),
            x.stride(1),
            grad.stride(0),
            grad.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BI=bi,
            BJ=bj,
            BK=bk,
            num_warps=warps,
            num_stages=stages,
            FP32_PRECISION=(
                "ieee" if torch.get_float32_matmul_precision() == "highest" else "tf32"
            ),
        )
        return output
    row_experts = torch.searchsorted(
        offsets[1:].contiguous(),
        torch.arange(x.shape[0], device=x.device, dtype=offsets.dtype),
        right=True,
    )
    for expert in range(weight_shape[0]):
        selected = (row_experts == expert).unsqueeze(1).to(x.dtype)
        output[expert].copy_((x * selected).T.to(output.dtype) @ grad.to(output.dtype))
    return output
