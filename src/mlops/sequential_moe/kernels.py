"""Bounded gather, activation, and scatter operations; no expanded dispatch."""

import triton
import triton.language as tl


@triton.jit
def publish(source, host, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(host + i, tl.load(source + i, i < N, other=0), i < N)


@triton.jit
def _gather(
    X,
    ORDER,
    OUT,
    START,
    VALID,
    D: tl.constexpr,
    K: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // D, i % D
    assignment = tl.load(ORDER + START + row, row < VALID, other=0).to(tl.int64)
    value = tl.load(
        X + (assignment // K) * D + col, (i < TOTAL) & (row < VALID), other=0
    )
    tl.store(OUT + i, value, i < TOTAL)


def gather(x, order, start, valid, out, top_k):
    _gather[(triton.cdiv(out.numel(), 512),)](
        x, order, out, start, valid, x.shape[1], top_k, out.numel(), 512
    )


@triton.jit
def _scatter(
    VALUE,
    ORDER,
    PROBS,
    OUT,
    START,
    VALID,
    D: tl.constexpr,
    K: tl.constexpr,
    WEIGHTED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // D, i % D
    mask = row < VALID
    slot = tl.load(ORDER + START + row, mask, other=0).to(tl.int64)
    value = tl.load(VALUE + i, mask, other=0).to(tl.float32)
    if WEIGHTED:
        value *= tl.load(PROBS + slot, mask, other=0)
    dst = OUT + (slot // K) * D + col
    prior = tl.load(dst, mask, other=0).to(tl.float32)
    tl.store(dst, prior + value, mask)


def scatter(value, order, probs, start, valid, out, top_k, weighted):
    # A top-k assignment contains an expert at most once per token, so rows
    # within one expert are unique. Experts/chunks execute on the same stream.
    _scatter[(triton.cdiv(valid * out.shape[1], 512),)](
        value, order, probs, out, start, valid, out.shape[1], top_k, weighted, 512
    )


@triton.jit
def _swiglu(
    GU, H, N: tl.constexpr, WIDTH: tl.constexpr, CLIP: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // WIDTH, i % WIDTH
    g = tl.load(GU + row * 2 * WIDTH + col, i < N, other=0).to(tl.float32)
    u = tl.load(GU + row * 2 * WIDTH + WIDTH + col, i < N, other=0).to(tl.float32)
    if CLIP > 0:
        limit = tl.full((), CLIP, tl.float32).to(GU.dtype.element_ty).to(tl.float32)
        g = tl.minimum(g, limit)
        u = tl.minimum(tl.maximum(u, -limit), limit)
        # Match the material activation-dtype SiLU result in the HF expression.
        a = (g * tl.sigmoid(g)).to(GU.dtype.element_ty).to(tl.float32)
    else:
        a = g * tl.sigmoid(g)
    tl.store(H + i, a * u, i < N)


def swiglu(gu, h, clip=0.0):
    _swiglu[(triton.cdiv(h.numel(), 512),)](gu, h, h.numel(), h.shape[1], clip, 512)


@triton.jit
def _swiglu_backward(
    GU,
    DH,
    DGU,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    CLIP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // WIDTH, i % WIDTH
    base = row * 2 * WIDTH + col
    raw_g = tl.load(GU + base, i < N, other=0).to(tl.float32)
    raw_u = tl.load(GU + base + WIDTH, i < N, other=0).to(tl.float32)
    dh = tl.load(DH + i, i < N, other=0).to(tl.float32)
    if CLIP > 0:
        limit = tl.full((), CLIP, tl.float32).to(GU.dtype.element_ty).to(tl.float32)
        g = tl.minimum(raw_g, limit)
        u = tl.minimum(tl.maximum(raw_u, -limit), limit)
        sig = tl.sigmoid(g)
        a = (g * sig).to(GU.dtype.element_ty).to(tl.float32)
        da = (dh * u).to(GU.dtype.element_ty).to(tl.float32)
        dg = tl.where(raw_g <= limit, da * sig * (1 + g * (1 - sig)), 0)
        du = tl.where((raw_u >= -limit) & (raw_u <= limit), dh * a, 0)
    else:
        sig = tl.sigmoid(raw_g)
        dg = dh * raw_u * sig * (1 + raw_g * (1 - sig))
        du = dh * raw_g * sig
    tl.store(DGU + base, dg, i < N)
    tl.store(DGU + base + WIDTH, du, i < N)


def swiglu_backward(gu, dh, dgu, clip=0.0):
    _swiglu_backward[(triton.cdiv(dh.numel(), 512),)](
        gu, dh, dgu, dh.numel(), dh.shape[1], clip, 512
    )


@triton.jit
def _weighted_dy(
    DY,
    Y,
    ORDER,
    PROBS,
    DP,
    OUT,
    START,
    VALID,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    slot = tl.load(ORDER + START + row, row < VALID, other=0).to(tl.int64)
    dy = tl.load(DY + row * D + col, (row < VALID) & (col < D), other=0).to(tl.float32)
    y = tl.load(Y + row * D + col, (row < VALID) & (col < D), other=0).to(tl.float32)
    p = tl.load(PROBS + slot, row < VALID, other=0)
    contribution = tl.sum(dy * y)
    if ACCUMULATE:
        contribution += tl.load(DP + slot, row < VALID, other=0)
    tl.store(DP + slot, contribution, row < VALID)
    tl.store(OUT + row * D + col, dy * p, col < D)


def weighted_dy(dy, y, order, probs, dp, start, valid, out, *, accumulate=False):
    _weighted_dy[(out.shape[0],)](
        dy,
        y,
        order,
        probs,
        dp,
        out,
        start,
        valid,
        out.shape[1],
        triton.next_power_of_2(out.shape[1]),
        accumulate,
    )
