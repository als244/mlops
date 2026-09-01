"""Fused LayerNorm forward/backward adapted from the production kernel."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


_BLOCK = 1024
_ROWS_PER_PROGRAM = 64


if triton is not None:

    @triton.jit
    def _forward_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        mean_ptr,
        rstd_ptr,
        width,
        eps,
        HAS_BIAS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        total = tl.zeros((BLOCK,), tl.float32)
        for offset in range(0, width, BLOCK):
            mask = offset + columns < width
            value = tl.load(
                x_ptr + row * width + offset + columns,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            total += value
        mean = tl.sum(total) / width
        squares = tl.zeros((BLOCK,), tl.float32)
        for offset in range(0, width, BLOCK):
            mask = offset + columns < width
            value = tl.load(
                x_ptr + row * width + offset + columns,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            centered = tl.where(mask, value - mean, 0.0)
            squares += centered * centered
        rstd = 1.0 / tl.sqrt(tl.sum(squares) / width + eps)
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, rstd)

        for offset in range(0, width, BLOCK):
            mask = offset + columns < width
            value = tl.load(
                x_ptr + row * width + offset + columns,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            normalized = (
                ((value - mean) * rstd).to(output_ptr.dtype.element_ty).to(tl.float32)
            )
            weight = tl.load(weight_ptr + offset + columns, mask=mask, other=0.0).to(
                tl.float32
            )
            result = (
                (normalized * weight).to(output_ptr.dtype.element_ty).to(tl.float32)
            )
            if HAS_BIAS:
                bias = tl.load(bias_ptr + offset + columns, mask=mask, other=0.0).to(
                    tl.float32
                )
                result += bias
            tl.store(
                output_ptr + row * width + offset + columns,
                result.to(output_ptr.dtype.element_ty),
                mask=mask,
            )

    @triton.jit
    def _backward_kernel(
        grad_ptr,
        x_ptr,
        mean_ptr,
        rstd_ptr,
        weight_ptr,
        grad_x_ptr,
        partial_weight_ptr,
        partial_bias_ptr,
        rows,
        width,
        REFERENCE_CASTS: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        program = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        for row_offset in range(ROWS_PER_PROGRAM):
            row = program * ROWS_PER_PROGRAM + row_offset
            live = row < rows
            mean = tl.load(mean_ptr + row, mask=live, other=0.0)
            rstd = tl.load(rstd_ptr + row, mask=live, other=0.0)
            sum_grad = tl.zeros((BLOCK,), tl.float32)
            sum_product = tl.zeros((BLOCK,), tl.float32)
            for offset in range(0, width, BLOCK):
                mask = live & (offset + columns < width)
                value = tl.load(
                    x_ptr + row * width + offset + columns,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                grad = tl.load(
                    grad_ptr + row * width + offset + columns,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                weight = tl.load(
                    weight_ptr + offset + columns,
                    mask=offset + columns < width,
                    other=0.0,
                ).to(tl.float32)
                normalized = (value - mean) * rstd
                grad_normalized = grad * weight
                if REFERENCE_CASTS:
                    grad_normalized = grad_normalized.to(
                        grad_x_ptr.dtype.element_ty
                    ).to(tl.float32)
                sum_grad += tl.where(mask, grad_normalized, 0.0)
                sum_product += tl.where(mask, grad_normalized * normalized, 0.0)
            correction_1 = tl.sum(sum_grad) / width
            correction_2 = tl.sum(sum_product) / width

            for offset in range(0, width, BLOCK):
                mask = live & (offset + columns < width)
                value = tl.load(
                    x_ptr + row * width + offset + columns,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                grad = tl.load(
                    grad_ptr + row * width + offset + columns,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                weight = tl.load(
                    weight_ptr + offset + columns,
                    mask=offset + columns < width,
                    other=0.0,
                ).to(tl.float32)
                normalized = (value - mean) * rstd
                grad_normalized = grad * weight
                normalized_for_weight = normalized
                if REFERENCE_CASTS:
                    grad_normalized = grad_normalized.to(
                        grad_x_ptr.dtype.element_ty
                    ).to(tl.float32)
                    normalized_for_weight = normalized.to(
                        grad_x_ptr.dtype.element_ty
                    ).to(tl.float32)
                grad_x = rstd * (
                    grad_normalized - correction_1 - normalized * correction_2
                )
                tl.store(
                    grad_x_ptr + row * width + offset + columns,
                    grad_x.to(grad_x_ptr.dtype.element_ty),
                    mask=mask,
                )
                weight_partial = partial_weight_ptr + program * width + offset + columns
                bias_partial = partial_bias_ptr + program * width + offset + columns
                previous_weight = tl.load(
                    weight_partial,
                    mask=offset + columns < width,
                    other=0.0,
                )
                previous_bias = tl.load(
                    bias_partial,
                    mask=offset + columns < width,
                    other=0.0,
                )
                tl.store(
                    weight_partial,
                    previous_weight
                    + tl.where(
                        mask,
                        (
                            (grad * normalized_for_weight)
                            .to(grad_x_ptr.dtype.element_ty)
                            .to(tl.float32)
                            if REFERENCE_CASTS
                            else grad * normalized_for_weight
                        ),
                        0.0,
                    ),
                    mask=offset + columns < width,
                )
                tl.store(
                    bias_partial,
                    previous_bias + tl.where(mask, grad, 0.0),
                    mask=offset + columns < width,
                )


def layer_norm_forward(x, weight, bias, eps):
    width = x.shape[-1]
    rows = x.numel() // width
    output = torch.empty_like(x)
    mean = torch.empty(rows, dtype=torch.float32, device=x.device)
    rstd = torch.empty_like(mean)
    if (
        triton is not None
        and x.is_cuda
        and x.is_contiguous()
        and weight.is_contiguous()
        and (bias is None or bias.is_contiguous())
    ):
        _forward_kernel[(rows,)](
            x,
            weight,
            x if bias is None else bias,
            output,
            mean,
            rstd,
            width,
            float(eps),
            HAS_BIAS=bias is not None,
            BLOCK=_BLOCK,
        )
        return output, mean, rstd
    x_float = x.reshape(rows, width).float()
    mean_value = x_float.mean(-1)
    rstd_value = torch.rsqrt(
        (x_float - mean_value.unsqueeze(-1)).square().mean(-1) + float(eps)
    )
    normalized = ((x_float - mean_value.unsqueeze(-1)) * rstd_value.unsqueeze(-1)).to(
        x.dtype
    )
    result = normalized * weight
    if bias is not None:
        result = result + bias
    output.copy_(result.to(x.dtype).reshape_as(x))
    mean.copy_(mean_value)
    rstd.copy_(rstd_value)
    return output, mean, rstd


def layer_norm_backward(grad_output, x, weight, mean, rstd):
    width = x.shape[-1]
    rows = x.numel() // width
    grad_output = grad_output.contiguous()
    grad_x = torch.empty_like(x)
    if (
        triton is not None
        and x.is_cuda
        and x.is_contiguous()
        and weight.is_contiguous()
    ):
        programs = triton.cdiv(rows, _ROWS_PER_PROGRAM)
        partials = torch.zeros(
            (2, programs, width), dtype=torch.float32, device=x.device
        )
        _backward_kernel[(programs,)](
            grad_output,
            x,
            mean,
            rstd,
            weight,
            grad_x,
            partials[0],
            partials[1],
            rows,
            width,
            REFERENCE_CASTS=True,
            ROWS_PER_PROGRAM=_ROWS_PER_PROGRAM,
            BLOCK=_BLOCK,
        )
        return (
            grad_x,
            partials[0].sum(0).to(weight.dtype),
            partials[1].sum(0).to(weight.dtype),
        )
    x_float = x.reshape(rows, width).float()
    normalized = (x_float - mean.unsqueeze(-1)) * rstd.unsqueeze(-1)
    normalized_storage = normalized.to(x.dtype)
    grad_normalized = (grad_output.reshape(rows, width) * weight).to(x.dtype).float()
    grad_x.copy_(
        (
            rstd.unsqueeze(-1)
            * (
                grad_normalized
                - grad_normalized.mean(-1, keepdim=True)
                - normalized * (grad_normalized * normalized).mean(-1, keepdim=True)
            )
        )
        .to(x.dtype)
        .reshape_as(x)
    )
    grad_weight = (
        (grad_output.reshape(rows, width) * normalized_storage).sum(0).to(weight.dtype)
    )
    grad_bias = grad_output.reshape(rows, width).sum(0).to(weight.dtype)
    return grad_x, grad_weight, grad_bias


__all__ = [
    "layer_norm_backward",
    "layer_norm_forward",
]
