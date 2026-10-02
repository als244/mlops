"""Tail masking and the Transformer Engine path's pointwise operations."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

if triton is not None:

    @triton.jit
    def _tail(X, END, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        # END remains device-resident. Start at the unused suffix and use a
        # bounded grid instead of launching over every active buffer element.
        start = tl.maximum(tl.load(END).to(tl.int64), 0) * WIDTH
        base = start + tl.program_id(0).to(tl.int64) * BLOCK
        while base < N:
            i = base + tl.arange(0, BLOCK)
            tl.store(X + i, 0, i < N)
            base += tl.num_programs(0) * BLOCK

    @triton.jit
    def _silu_mul(G, U, O, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        g = tl.load(G + i, i < N, 0).to(tl.float32)
        u = tl.load(U + i, i < N, 0).to(tl.float32)
        tl.store(O + i, g / (1 + tl.exp(-g)) * u, i < N)

    @triton.jit
    def _silu_mul_bwd(G, U, DH, DG, DU, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        g = tl.load(G + i, i < N, 0).to(tl.float32)
        u = tl.load(U + i, i < N, 0).to(tl.float32)
        dh = tl.load(DH + i, i < N, 0).to(tl.float32)
        s = 1 / (1 + tl.exp(-g))
        tl.store(DG + i, dh * u * s * (1 + g * (1 - s)), i < N)
        tl.store(DU + i, dh * g * s, i < N)

    @triton.jit
    def _scale_rows(X, P, Y, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + i, i < N, 0).to(tl.float32)
        p = tl.load(P + i // WIDTH, i < N, 0)
        tl.store(Y + i, x * p, i < N)

    @triton.jit
    def _prob_grad(DY, Y, DP, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        j = tl.arange(0, BLOCK)
        dy = tl.load(DY + row * WIDTH + j, j < WIDTH, 0).to(tl.float32)
        y = tl.load(Y + row * WIDTH + j, j < WIDTH, 0).to(tl.float32)
        tl.store(DP + row, tl.sum(dy * y, 0))

    @triton.jit
    def _prob_grad_scale(DY, Y, P, DP, DE, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        j = tl.arange(0, BLOCK)
        dy = tl.load(DY + row * WIDTH + j, j < WIDTH, 0).to(tl.float32)
        y = tl.load(Y + row * WIDTH + j, j < WIDTH, 0).to(tl.float32)
        p = tl.load(P + row)
        tl.store(DP + row, tl.sum(dy * y, 0))
        tl.store(DE + row * WIDTH + j, dy * p, j < WIDTH)


class _Pointwise:
    def mask_tail(self, x, end):
        _tail[(min(1024, triton.cdiv(x.numel(), 1024)),)](
            x, end, x.numel(), x.shape[-1] if x.ndim == 2 else 1, 1024
        )

    def swiglu(self, g, u):
        o = torch.empty_like(g)
        _silu_mul[(triton.cdiv(g.numel(), 1024),)](g, u, o, g.numel(), 1024)
        return o

    def swiglu_backward(self, g, u, dh):
        dg, du = torch.empty_like(g), torch.empty_like(u)
        _silu_mul_bwd[(triton.cdiv(g.numel(), 256),)](g, u, dh, dg, du, g.numel(), 256)
        return dg, du

    def scale(self, x, p, *, inplace=False, out=None):
        if inplace and out is not None:
            raise ValueError("Choose inplace or an explicit output, not both")
        o = out if out is not None else x if inplace else torch.empty_like(x)
        _scale_rows[(triton.cdiv(x.numel(), 1024),)](
            x, p, o, x.numel(), x.shape[1], 1024
        )
        return o

    def probability_grad(self, dy, y):
        out = torch.empty(dy.shape[0], device=dy.device, dtype=torch.float32)
        block = triton.next_power_of_2(dy.shape[1])
        _prob_grad[(dy.shape[0],)](
            dy, y, out, dy.shape[1], block, num_warps=4 if block < 2048 else 8
        )
        return out

    def probability_grad_and_scale(self, dy, y, p, *, inplace=False):
        dp = torch.empty(dy.shape[0], device=dy.device, dtype=torch.float32)
        de = dy if inplace else torch.empty_like(dy)
        block = triton.next_power_of_2(dy.shape[1])
        _prob_grad_scale[(dy.shape[0],)](
            dy, y, p, dp, de, dy.shape[1], block, num_warps=4 if block < 2048 else 8
        )
        return dp, de
