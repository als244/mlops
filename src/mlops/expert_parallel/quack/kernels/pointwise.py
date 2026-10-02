"""Mask only unused dispatch capacity; expert epilogues handle the other elementwise math."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _tail(X, END, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    # END remains device-resident. Start at the unused suffix and use a
    # bounded grid instead of launching over every active buffer element.
    start = tl.maximum(tl.load(END).to(tl.int64), 0) * WIDTH
    base = start + tl.program_id(0).to(tl.int64) * BLOCK
    while base < N:
        i = base + tl.arange(0, BLOCK)
        tl.store(X + i, 0.0, i < N)
        base += tl.num_programs(0) * BLOCK


class _Pointwise:
    def mask_tail(self, x, end):
        _tail[(min(1024, triton.cdiv(x.numel(), 1024)),)](
            x, end, x.numel(), x.shape[-1] if x.ndim == 2 else 1, 1024
        )
