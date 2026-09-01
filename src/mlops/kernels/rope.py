"""Fused Llama rotate-half RoPE forward/backward."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


_BLOCK = 512


def build_rope_tables(
    device: torch.device,
    rows: int,
    head_dim: int,
    base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build reusable reference-equivalent fp32 cosine/sine tables.

    The returned tensors are ordinary caller-owned inputs to RoPE. They are
    not invocation-local kernel workspace and this module never caches them.
    """
    inverse = 1.0 / (
        float(base)
        ** (
            torch.arange(
                0,
                int(head_dim),
                2,
                device=device,
                dtype=torch.float32,
            )
            / int(head_dim)
        )
    )
    position = torch.arange(int(rows), device=device, dtype=torch.float32)
    frequencies = torch.outer(position, inverse)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos(), embedding.sin()


if triton is not None:

    @triton.jit
    def _rope_kernel(
        x_ptr,
        output_ptr,
        positions_ptr,
        pairs,
        width,
        head_dim,
        base,
        SIN_SIGN: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < pairs
        half = head_dim // 2
        pairs_per_row = width // 2
        row = offsets // pairs_per_row
        remainder = offsets % pairs_per_row
        head = remainder // half
        index = remainder % half

        position = tl.load(positions_ptr + row, mask=mask, other=0).to(tl.float32)
        exponent = -2.0 * index.to(tl.float32) / head_dim
        frequency = tl.exp2(exponent * tl.log2(base))
        angle = position * frequency
        cosine = tl.cos(angle)
        sine = tl.sin(angle) * SIN_SIGN

        low_index = row * width + head * head_dim + index
        high_index = low_index + half
        low = tl.load(x_ptr + low_index, mask=mask, other=0.0).to(tl.float32)
        high = tl.load(x_ptr + high_index, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            output_ptr + low_index,
            (low * cosine - high * sine).to(output_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            output_ptr + high_index,
            (high * cosine + low * sine).to(output_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _rope_table_kernel(
        x_ptr,
        output_ptr,
        positions_ptr,
        cosine_ptr,
        sine_ptr,
        pairs,
        width,
        head_dim,
        SIN_SIGN: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < pairs
        half = head_dim // 2
        pairs_per_row = width // 2
        row = offsets // pairs_per_row
        remainder = offsets % pairs_per_row
        head = remainder // half
        index = remainder % half
        position = tl.load(positions_ptr + row, mask=mask, other=0).to(tl.int64)
        table_offset = position * head_dim + index
        cosine = tl.load(cosine_ptr + table_offset, mask=mask, other=1.0).to(tl.float32)
        sine = (
            tl.load(sine_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
            * SIN_SIGN
        )

        low_index = row * width + head * head_dim + index
        high_index = low_index + half
        low = tl.load(x_ptr + low_index, mask=mask, other=0.0).to(tl.float32)
        high = tl.load(x_ptr + high_index, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            output_ptr + low_index,
            (low * cosine - high * sine).to(output_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            output_ptr + high_index,
            (high * cosine + low * sine).to(output_ptr.dtype.element_ty),
            mask=mask,
        )


def _native_torch(
    x: torch.Tensor,
    positions: torch.Tensor,
    cosine_table: torch.Tensor,
    sine_table: torch.Tensor,
    sign: float,
) -> torch.Tensor:
    head_dim = x.shape[-1]
    cosine = cosine_table[positions.long()]
    sine = sine_table[positions.long()] * sign
    flat = x.reshape(positions.numel(), -1, head_dim).float()
    half = head_dim // 2
    rotated = torch.cat((-flat[..., half:], flat[..., :half]), dim=-1)
    return (
        (flat * cosine[:, None, :] + rotated * sine[:, None, :])
        .to(x.dtype)
        .reshape_as(x)
    )


def reference_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cosine_table: torch.Tensor,
    sine_table: torch.Tensor,
) -> torch.Tensor:
    """Raw-model RoPE expression using ordinary PyTorch autograd.

    Unlike the custom boundary, this function leaves the elementwise
    expression visible to autograd. It is a diagnostic variant for separating
    Triton forward/FMA effects from custom inverse-rotation backward effects.
    """
    return _native_torch(x, positions, cosine_table, sine_table, 1.0)


def reference_rope_backward(
    grad_output: torch.Tensor,
    positions: torch.Tensor,
    cosine_table: torch.Tensor,
    sine_table: torch.Tensor,
) -> torch.Tensor:
    """Apply the ordinary PyTorch inverse rotation for an explicit VJP."""
    return _native_torch(
        grad_output, positions, cosine_table, sine_table, -1.0
    )


def _validate_tables(
    x: torch.Tensor,
    positions: torch.Tensor,
    cosine_table: torch.Tensor,
    sine_table: torch.Tensor,
) -> None:
    head_dim = x.shape[-1]
    if cosine_table.shape != sine_table.shape:
        raise ValueError("cosine and sine tables must have identical shapes")
    if cosine_table.ndim != 2 or cosine_table.shape[1] != head_dim:
        raise ValueError(
            "RoPE tables must have shape (positions, head_dim); "
            f"got {tuple(cosine_table.shape)} for head_dim={head_dim}"
        )
    if cosine_table.shape[0] == 0:
        raise ValueError("RoPE tables must contain at least one position")
    if cosine_table.device != x.device or sine_table.device != x.device:
        raise ValueError("RoPE tables and x must share a device")


def _apply(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    cosine_table: torch.Tensor,
    sine_table: torch.Tensor,
    sign: float,
    variant: str,
) -> torch.Tensor:
    if x.shape[-1] % 2:
        raise ValueError(f"RoPE head dimension must be even; got {x.shape[-1]}")
    positions = positions.contiguous()
    rows = positions.numel()
    if x.numel() % rows:
        raise ValueError("positions do not describe the leading token dimensions")
    head_dim = x.shape[-1]
    _validate_tables(x, positions, cosine_table, sine_table)
    width = x.numel() // rows
    output = torch.empty_like(x)
    variant = str(variant)
    if (
        variant == "triton_analytic"
        and triton is not None
        and x.is_cuda
        and x.is_contiguous()
    ):
        pairs = rows * (width // 2)
        _rope_kernel[(triton.cdiv(pairs, _BLOCK),)](
            x,
            output,
            positions,
            pairs,
            width,
            head_dim,
            float(base),
            SIN_SIGN=float(sign),
            BLOCK=_BLOCK,
        )
        return output
    if triton is not None and x.is_cuda and x.is_contiguous():
        pairs = rows * (width // 2)
        _rope_table_kernel[(triton.cdiv(pairs, _BLOCK),)](
            x,
            output,
            positions,
            cosine_table,
            sine_table,
            pairs,
            width,
            head_dim,
            SIN_SIGN=float(sign),
            BLOCK=_BLOCK,
        )
        return output
    cosine = cosine_table[positions.long()]
    sine = sine_table[positions.long()] * float(sign)
    flat = x.reshape(rows, -1, head_dim).float()
    half = head_dim // 2
    rotated = torch.cat((-flat[..., half:], flat[..., :half]), dim=-1)
    return (
        (flat * cosine[:, None, :] + rotated * sine[:, None, :])
        .to(x.dtype)
        .reshape_as(x)
    )


def rope_forward(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    cosine_table: torch.Tensor,
    sine_table: torch.Tensor,
    *,
    variant: str,
) -> torch.Tensor:
    return _apply(
        x, positions, base, cosine_table, sine_table, 1.0, variant
    )


def rope_backward(
    grad_output: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    cosine_table: torch.Tensor,
    sine_table: torch.Tensor,
    *,
    variant: str,
) -> torch.Tensor:
    return _apply(
        grad_output.contiguous(),
        positions,
        base,
        cosine_table,
        sine_table,
        -1.0,
        variant,
    )


def rope_implementation(x: torch.Tensor, variant: str) -> str:
    """Describe the backend selected for a contiguous RoPE input."""
    if variant == "triton_analytic" and triton is not None and x.is_cuda:
        return "triton.rope_analytic"
    if triton is not None and x.is_cuda:
        return "triton.rope_table"
    return "native_torch.rope_table_fallback"


__all__ = [
    "build_rope_tables",
    "reference_rope",
    "reference_rope_backward",
    "rope_backward",
    "rope_forward",
    "rope_implementation",
]
