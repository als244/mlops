"""Matrix products whose sums are kept at a dtype of the caller's choosing."""

from __future__ import annotations

from functools import cache

import torch


@cache
def _multiply_writes(operands: torch.dtype, result: torch.dtype) -> bool:
    """Whether a matrix multiply of ``operands`` sums and writes at ``result``
    itself: the operator's own check, asked of meta tensors."""
    probe = torch.empty((1, 1), dtype=operands, device="meta")
    try:
        torch.ops.aten.mm.dtype(probe, probe, result)
    except RuntimeError:
        return False
    return True


def _in_multiply(left: torch.Tensor, right: torch.Tensor, dtype: torch.dtype) -> bool:
    return (
        left.is_cuda
        and left.dtype == right.dtype
        and _multiply_writes(left.dtype, dtype)
    )


def product_at(
    left: torch.Tensor, right: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """``left @ right`` with its products summed and returned at ``dtype``.

    The multiply does it itself where it can -- fp32 from bf16 or fp16 on
    CUDA -- and otherwise multiplies the operands taken to ``dtype``, which
    for a wider dtype is exact.
    """
    if _in_multiply(left, right, dtype):
        return torch.ops.aten.mm.dtype(left, right, dtype)
    return left.to(dtype) @ right.to(dtype)


def add_product_(
    accumulator: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Add ``left @ right`` into ``accumulator`` in place, at its dtype.

    Where the multiply can write that dtype it adds as it writes -- cuBLAS's
    ``C = A @ B + C`` -- so the product is never stored on its own; otherwise
    the operands are taken to the accumulator's dtype and the product added.
    """
    if _in_multiply(left, right, accumulator.dtype):
        torch.ops.aten.addmm.dtype_out(
            accumulator, left, right, accumulator.dtype, out=accumulator
        )
    else:
        accumulator.add_(left.to(accumulator.dtype) @ right.to(accumulator.dtype))
    return accumulator


__all__ = ["add_product_", "product_at"]
