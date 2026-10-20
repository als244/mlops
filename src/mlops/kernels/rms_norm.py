"""Fused RMSNorm forward/backward, adapted from the production kernels."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised in environments without Triton
    triton = None
    tl = None


_BLOCK = 1024
_ROWS_PER_PROGRAM = 64


if triton is not None:

    @triton.jit
    def _forward_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        rstd_ptr,
        width,
        eps,
        ZERO_CENTERED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        squares = tl.zeros((BLOCK,), tl.float32)
        for offset in range(0, width, BLOCK):
            mask = offset + columns < width
            x = tl.load(
                x_ptr + row * width + offset + columns,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            squares += x * x
        rstd = 1.0 / tl.sqrt(tl.sum(squares) / width + eps)
        tl.store(rstd_ptr + row, rstd)

        for offset in range(0, width, BLOCK):
            mask = offset + columns < width
            x = tl.load(
                x_ptr + row * width + offset + columns,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(weight_ptr + offset + columns, mask=mask, other=0.0).to(
                tl.float32
            )
            if ZERO_CENTERED:
                # A zero-centered weight is an offset from one: the fp32
                # normalized value takes the gain 1 + weight before the one
                # cast to the activation dtype.
                output = x * rstd * (1.0 + weight)
            else:
                # Match the production/reference convention: normalization is
                # rounded to the activation dtype before applying the gain.
                x_hat = (x * rstd).to(output_ptr.dtype.element_ty).to(tl.float32)
                output = x_hat * weight
            tl.store(
                output_ptr + row * width + offset + columns,
                output.to(output_ptr.dtype.element_ty),
                mask=mask,
            )

    @triton.jit
    def _backward_kernel(
        grad_ptr,
        x_ptr,
        rstd_ptr,
        weight_ptr,
        grad_x_ptr,
        partial_weight_ptr,
        rows,
        width,
        REFERENCE_CASTS: tl.constexpr,
        ZERO_CENTERED: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        program = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        row_start = program * ROWS_PER_PROGRAM
        for row_offset in range(ROWS_PER_PROGRAM):
            row = row_start + row_offset
            live = row < rows
            rstd = tl.load(rstd_ptr + row, mask=live, other=0.0)
            dot = tl.zeros((BLOCK,), tl.float32)
            for offset in range(0, width, BLOCK):
                mask = live & (offset + columns < width)
                x = tl.load(
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
                if ZERO_CENTERED:
                    weight = 1.0 + weight
                grad_normalized = grad * weight
                if REFERENCE_CASTS:
                    # Raw PyTorch multiplies two bf16 tensors here before the
                    # cast's straight-through gradient returns to fp32.
                    grad_normalized = grad_normalized.to(
                        grad_x_ptr.dtype.element_ty
                    ).to(tl.float32)
                dot += grad_normalized * (x * rstd)
            correction = tl.sum(dot) / width

            for offset in range(0, width, BLOCK):
                mask = live & (offset + columns < width)
                x = tl.load(
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
                if ZERO_CENTERED:
                    weight = 1.0 + weight
                x_hat = x * rstd
                grad_normalized = grad * weight
                x_hat_for_weight = x_hat
                if REFERENCE_CASTS:
                    grad_normalized = grad_normalized.to(
                        grad_x_ptr.dtype.element_ty
                    ).to(tl.float32)
                    x_hat_for_weight = x_hat.to(grad_x_ptr.dtype.element_ty).to(
                        tl.float32
                    )
                grad_x = rstd * (grad_normalized - x_hat * correction)
                tl.store(
                    grad_x_ptr + row * width + offset + columns,
                    grad_x.to(grad_x_ptr.dtype.element_ty),
                    mask=mask,
                )
                partial_ptr = partial_weight_ptr + program * width + offset + columns
                previous = tl.load(
                    partial_ptr,
                    mask=offset + columns < width,
                    other=0.0,
                )
                contribution = grad * x_hat_for_weight
                if REFERENCE_CASTS:
                    contribution = contribution.to(grad_x_ptr.dtype.element_ty).to(
                        tl.float32
                    )
                contribution = tl.where(mask, contribution, 0.0)
                tl.store(
                    partial_ptr,
                    previous + contribution,
                    mask=offset + columns < width,
                )


def _can_use_triton(*tensors: torch.Tensor) -> bool:
    return bool(
        triton is not None
        and tensors
        and all(t.is_cuda and t.is_contiguous() for t in tensors)
    )


def rms_norm_can_use_triton(*tensors: torch.Tensor) -> bool:
    """Return whether the raw Triton kernels accept ``tensors``."""
    return _can_use_triton(*tensors)


def rms_norm_triton_available() -> bool:
    """Return whether Triton imported; this check never touches tensor storage."""
    return triton is not None


def rms_norm_forward_triton(
    x: torch.Tensor, weight: torch.Tensor, eps: float, zero_centered: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run only the Triton forward; unsupported inputs never fall back. A
    zero-centered weight is an offset from one, applied as ``1 + weight`` to
    the fp32 normalized value before the one cast."""
    if not _can_use_triton(x, weight):
        raise RuntimeError(
            "builtin Triton RMSNorm requires contiguous CUDA x and weight tensors"
        )
    width = x.shape[-1]
    rows = x.numel() // width
    output = torch.empty_like(x)
    rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
    _forward_kernel[(rows,)](
        x,
        weight,
        output,
        rstd,
        width,
        float(eps),
        ZERO_CENTERED=bool(zero_centered),
        BLOCK=_BLOCK,
    )
    return output, rstd


def rms_norm_backward_triton(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    grad_weight_dtype: torch.dtype | None = None,
    zero_centered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run only the Triton VJP; unsupported inputs never fall back. The
    zero-centered VJP is fp32 throughout, as its forward is: the gain is
    ``1 + weight`` and nothing is rounded before the stores."""
    grad_output = grad_output.contiguous()
    if not _can_use_triton(grad_output, x, weight, rstd):
        raise RuntimeError(
            "builtin Triton RMSNorm VJP requires contiguous CUDA tensors"
        )
    width = x.shape[-1]
    rows = x.numel() // width
    grad_x = torch.empty_like(x)
    programs = triton.cdiv(rows, _ROWS_PER_PROGRAM)
    partials = torch.zeros((programs, width), dtype=torch.float32, device=x.device)
    _backward_kernel[(programs,)](
        grad_output,
        x,
        rstd,
        weight,
        grad_x,
        partials,
        rows,
        width,
        REFERENCE_CASTS=not zero_centered,
        ZERO_CENTERED=bool(zero_centered),
        ROWS_PER_PROGRAM=_ROWS_PER_PROGRAM,
        BLOCK=_BLOCK,
    )
    output_dtype = weight.dtype if grad_weight_dtype is None else grad_weight_dtype
    return grad_x, partials.sum(0).to(output_dtype)


def rms_norm_forward(
    x: torch.Tensor, weight: torch.Tensor, eps: float, zero_centered: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized output and the only freshly saved value, ``rstd``."""
    width = x.shape[-1]
    rows = x.numel() // width
    output = torch.empty_like(x)
    rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
    if _can_use_triton(x, weight):
        return rms_norm_forward_triton(x, weight, eps, zero_centered)

    x_float = x.reshape(rows, width).float()
    rstd_value = torch.rsqrt(x_float.square().mean(-1) + float(eps))
    if zero_centered:
        normalized = x_float * rstd_value.unsqueeze(-1) * (1 + weight.float())
        output.copy_(normalized.to(x.dtype).reshape_as(x))
    else:
        x_hat = (x_float * rstd_value.unsqueeze(-1)).to(x.dtype)
        output.copy_((x_hat * weight).to(x.dtype).reshape_as(x))
    rstd.copy_(rstd_value)
    return output, rstd


def rms_norm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    grad_weight_dtype: torch.dtype | None = None,
    zero_centered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = x.shape[-1]
    rows = x.numel() // width
    grad_output = grad_output.contiguous()
    grad_x = torch.empty_like(x)
    if _can_use_triton(grad_output, x, weight, rstd):
        return rms_norm_backward_triton(
            grad_output, x, weight, rstd, grad_weight_dtype, zero_centered
        )

    if zero_centered:
        x_hat = x.reshape(rows, width).float() * rstd.unsqueeze(-1)
        grad_rows = grad_output.reshape(rows, width).float()
        grad_hat = grad_rows * (1 + weight.float())
        correction = (grad_hat * x_hat).mean(-1, keepdim=True)
        grad_x.copy_(
            (rstd.unsqueeze(-1) * (grad_hat - x_hat * correction))
            .to(x.dtype)
            .reshape_as(x)
        )
        products = (grad_rows * x_hat).sum(0)
        return grad_x, products.to(
            weight.dtype if grad_weight_dtype is None else grad_weight_dtype
        )

    x_float = x.reshape(rows, width).float()
    x_hat = x_float * rstd.unsqueeze(-1)
    x_hat_storage = x_hat.to(x.dtype)
    grad_hat = (grad_output.reshape(rows, width) * weight).to(x.dtype).float()
    correction = (grad_hat * x_hat).mean(-1, keepdim=True)
    grad_x.copy_(
        (rstd.unsqueeze(-1) * (grad_hat - x_hat * correction)).to(x.dtype).reshape_as(x)
    )
    products = grad_output.reshape(rows, width) * x_hat_storage
    grad_weight = (
        products.sum(0).to(weight.dtype)
        if grad_weight_dtype is None
        else products.sum(0, dtype=grad_weight_dtype)
    )
    return grad_x, grad_weight


__all__ = [
    "rms_norm_backward",
    "rms_norm_backward_triton",
    "rms_norm_can_use_triton",
    "rms_norm_forward",
    "rms_norm_forward_triton",
]
