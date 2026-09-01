"""MoE stable sort, dispatch, combine, row scaling, and row dot kernels."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


_BM, _BD = 32, 128


if triton is not None:

    @triton.jit
    def _gather_sum_kernel(
        source_ptr,
        slots_ptr,
        weights_ptr,
        residual_ptr,
        output_ptr,
        tokens,
        width,
        rows,
        K: tl.constexpr,
        BM: tl.constexpr,
        BD: tl.constexpr,
        USE_WEIGHTS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
        column = tl.program_id(1).to(tl.int64) * BD + tl.arange(0, BD).to(tl.int64)
        token_mask = token < tokens
        column_mask = column < width
        mask = token_mask[:, None] & column_mask[None, :]
        if HAS_RESIDUAL:
            accumulator = tl.load(
                residual_ptr + token[:, None] * width + column[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        else:
            accumulator = tl.zeros((BM, BD), dtype=tl.float32)
        for index in range(K):
            slot = tl.load(slots_ptr + token * K + index, mask=token_mask, other=0).to(
                tl.int64
            )
            slot = tl.minimum(tl.maximum(slot, 0), rows - 1)
            value = tl.load(
                source_ptr + slot[:, None] * width + column[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            if USE_WEIGHTS:
                weight = tl.load(
                    weights_ptr + token * K + index,
                    mask=token_mask,
                    other=0.0,
                ).to(tl.float32)
                accumulator += value * weight[:, None]
            else:
                accumulator += value
        tl.store(
            output_ptr + token[:, None] * width + column[None, :],
            accumulator.to(output_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _scale_rows_kernel(x_ptr, scale_ptr, total, width, BLOCK: tl.constexpr):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(
            tl.int64
        )
        mask = offsets < total
        row = offsets // width
        value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + row, mask=mask, other=0.0)
        tl.store(
            x_ptr + offsets,
            (value * scale).to(x_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _row_dot_kernel(a_ptr, b_ptr, output_ptr, width, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
        for offset in range(0, width, BLOCK):
            columns = offset + tl.arange(0, BLOCK)
            mask = columns < width
            a = tl.load(a_ptr + row * width + columns, mask=mask, other=0.0).to(
                tl.float32
            )
            b = tl.load(b_ptr + row * width + columns, mask=mask, other=0.0).to(
                tl.float32
            )
            accumulator += a * b
        tl.store(output_ptr + row, tl.sum(accumulator))


def sort_assignments(route_ids, experts):
    flat = route_ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True).to(torch.int32)
    counts = torch.zeros(experts, dtype=torch.long, device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    offsets = torch.zeros(experts + 1, dtype=torch.int32, device=flat.device)
    offsets[1:].copy_(counts.cumsum(0).to(torch.int32))
    slots = torch.empty_like(flat, dtype=torch.int32)
    slots.scatter_(
        0,
        order.long(),
        torch.arange(flat.numel(), dtype=torch.int32, device=flat.device),
    )
    return order, offsets, slots.view_as(route_ids)


def dispatch(x, order, top_k):
    source = torch.div(order.long(), int(top_k), rounding_mode="floor")
    return torch.index_select(x, 0, source)


def _native_torch_gather(source, slots):
    tokens, top_k = slots.shape
    return (
        torch.index_select(source, 0, slots.reshape(-1).long())
        .view(tokens, top_k, source.shape[1])
        .float()
    )


def dispatch_backward(source, slots):
    output = torch.empty(
        (slots.shape[0], source.shape[1]),
        dtype=source.dtype,
        device=source.device,
    )
    if triton is not None and source.is_cuda:
        _gather_sum_kernel[
            (
                triton.cdiv(slots.shape[0], _BM),
                triton.cdiv(source.shape[1], _BD),
            )
        ](
            source,
            slots,
            source,
            source,
            output,
            slots.shape[0],
            source.shape[1],
            source.shape[0],
            K=slots.shape[1],
            BM=_BM,
            BD=_BD,
            USE_WEIGHTS=False,
            HAS_RESIDUAL=False,
        )
        return output
    output.copy_(_native_torch_gather(source, slots).sum(1))
    return output


def combine(source, slots, weights, residual):
    output = torch.empty_like(residual)
    if triton is not None and source.is_cuda:
        _gather_sum_kernel[
            (
                triton.cdiv(slots.shape[0], _BM),
                triton.cdiv(source.shape[1], _BD),
            )
        ](
            source,
            slots,
            weights,
            residual,
            output,
            slots.shape[0],
            source.shape[1],
            source.shape[0],
            K=slots.shape[1],
            BM=_BM,
            BD=_BD,
            USE_WEIGHTS=True,
            HAS_RESIDUAL=True,
        )
        return output
    result = (_native_torch_gather(source, slots) * weights.float().unsqueeze(-1)).sum(
        1
    )
    output.copy_((residual.float() + result).to(output.dtype))
    return output


def scale_rows_(x, scale):
    scale = scale.contiguous()
    if triton is not None and x.is_cuda and x.is_contiguous():
        _scale_rows_kernel[(triton.cdiv(x.numel(), 1024),)](
            x, scale, x.numel(), x.shape[1], BLOCK=1024
        )
    else:
        x.copy_((x.float() * scale.unsqueeze(1)).to(x.dtype))
    return x


def row_dot(a, b):
    output = torch.empty(a.shape[0], dtype=torch.float32, device=a.device)
    if triton is not None and a.is_cuda and a.is_contiguous() and b.is_contiguous():
        _row_dot_kernel[(a.shape[0],)](a, b, output, a.shape[1], BLOCK=1024)
        return output
    output.copy_((a.float() * b.float()).sum(-1))
    return output
