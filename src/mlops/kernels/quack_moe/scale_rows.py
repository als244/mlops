"""Rows scaled by one FP32 factor each and rounded once.

``out[i] = x[i] * factors[i]``: each row of ``x`` is read in FP32, multiplied by
its own factor and rounded to ``out``'s dtype once, in one pass and with no FP32
copy of ``x``. Routed expert rows are scaled by their route scores this way, and
E4M3 rows are dequantized by their descales. Rows may be pitched; the elements
within a row are contiguous.
"""

import torch
import triton
import triton.language as tl

from .quantize_rows import warps


@triton.jit
def _scale_rows(X, S, OUT, WIDTH, X_PITCH, OUT_PITCH, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    # Masked lanes are never stored; an E4M3 load takes no integer fill.
    value = tl.load(X + row * X_PITCH + col, mask).to(tl.float32)
    factor = tl.load(S + row).to(tl.float32)
    scaled = (value * factor).to(OUT.dtype.element_ty)
    tl.store(OUT + row * OUT_PITCH + col, scaled, mask)


def _rows(value, name):
    if value.ndim != 2 or not value.is_cuda:
        raise ValueError(f"Expected {name} as CUDA [rows, width]")
    if value.shape[1] > 1 and value.stride(1) != 1:
        raise ValueError(f"Expected {name}'s rows to be contiguous")


def scale_rows(x, factors, *, out=None, dtype=None):
    """``x [rows, width]`` with row ``i`` multiplied by ``factors[i]`` in FP32,
    rounded once to ``dtype`` (``x``'s dtype by default), into ``out`` when
    given."""
    _rows(x, "the rows")
    rows, width = x.shape
    if factors.shape != (rows,) or not factors.is_cuda:
        raise ValueError("Expected one factor per row")
    if out is None:
        out = torch.empty(
            rows, width, dtype=x.dtype if dtype is None else dtype, device=x.device
        )
    else:
        _rows(out, "the destination")
        if out.shape != x.shape or (dtype is not None and out.dtype != dtype):
            raise ValueError("The destination must match the rows and the dtype")
    if rows and width:
        block = triton.next_power_of_2(width)
        _scale_rows[(rows,)](
            x,
            factors.contiguous(),
            out,
            width,
            x.stride(0),
            out.stride(0),
            block,
            num_warps=warps(block),
        )
    return out
