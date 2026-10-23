"""Four-stream residual mixing without a tensor-core GEMM's padded tiles."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mix_four(
    A,
    X,
    Y,
    N: tl.constexpr,
    D: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    A2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = idx // D
    col = idx % D
    valid = idx < N * D
    base = token * 4 * D + col
    x0 = tl.load(X + base, valid, 0).to(tl.float32)
    x1 = tl.load(X + base + D, valid, 0).to(tl.float32)
    x2 = tl.load(X + base + 2 * D, valid, 0).to(tl.float32)
    x3 = tl.load(X + base + 3 * D, valid, 0).to(tl.float32)
    for row in tl.static_range(4):
        a = token * A0 + row * A1
        a0 = tl.load(A + a, valid, 0).to(tl.float32)
        a1 = tl.load(A + a + A2, valid, 0).to(tl.float32)
        a2 = tl.load(A + a + 2 * A2, valid, 0).to(tl.float32)
        a3 = tl.load(A + a + 3 * A2, valid, 0).to(tl.float32)
        value = ((a0 * x0 + a1 * x1) + a2 * x2) + a3 * x3
        tl.store(Y + base + row * D, value, valid)


def mix_four(matrix, streams):
    output = torch.empty_like(streams, memory_format=torch.contiguous_format)
    tokens, _, width = streams.shape
    if tokens and width:
        _mix_four[(triton.cdiv(tokens * width, 1024),)](
            matrix,
            streams,
            output,
            tokens,
            width,
            *matrix.stride(),
            BLOCK=1024,
            num_warps=4,
        )
    return output


@triton.jit
def _post_four(B, X, P, M, Y, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, col = idx // D, idx % D
    valid = idx < N * D
    base = token * 4 * D + col
    b = tl.load(B + idx, valid, 0).to(tl.float32)
    x0 = tl.load(X + base, valid, 0).to(tl.float32)
    x1 = tl.load(X + base + D, valid, 0).to(tl.float32)
    x2 = tl.load(X + base + 2 * D, valid, 0).to(tl.float32)
    x3 = tl.load(X + base + 3 * D, valid, 0).to(tl.float32)
    for j in tl.static_range(4):
        m0 = tl.load(M + token * 16 + j, valid, 0)
        m1 = tl.load(M + token * 16 + 4 + j, valid, 0)
        m2 = tl.load(M + token * 16 + 8 + j, valid, 0)
        m3 = tl.load(M + token * 16 + 12 + j, valid, 0)
        p = tl.load(P + token * 4 + j, valid, 0)
        residual = ((m0 * x0 + m1 * x1) + m2 * x2) + m3 * x3
        tl.store(Y + base + j * D, residual + p * b, valid)


@triton.jit
def _post_four_input_grads(
    G, P, M, DB, DX, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, col = idx // D, idx % D
    valid = idx < N * D
    base = token * 4 * D + col
    g0 = tl.load(G + base, valid, 0).to(tl.float32)
    g1 = tl.load(G + base + D, valid, 0).to(tl.float32)
    g2 = tl.load(G + base + 2 * D, valid, 0).to(tl.float32)
    g3 = tl.load(G + base + 3 * D, valid, 0).to(tl.float32)
    p0 = tl.load(P + token * 4, valid, 0)
    p1 = tl.load(P + token * 4 + 1, valid, 0)
    p2 = tl.load(P + token * 4 + 2, valid, 0)
    p3 = tl.load(P + token * 4 + 3, valid, 0)
    tl.store(DB + idx, ((p0 * g0 + p1 * g1) + p2 * g2) + p3 * g3, valid)
    for i in tl.static_range(4):
        m0 = tl.load(M + token * 16 + i * 4, valid, 0)
        m1 = tl.load(M + token * 16 + i * 4 + 1, valid, 0)
        m2 = tl.load(M + token * 16 + i * 4 + 2, valid, 0)
        m3 = tl.load(M + token * 16 + i * 4 + 3, valid, 0)
        tl.store(DX + base + i * D, ((m0 * g0 + m1 * g1) + m2 * g2) + m3 * g3, valid)


@triton.jit
def _post_four_coefficient_grads(G, B, X, DP, DM, D: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < D
    base = token * 4 * D + col
    b = tl.load(B + token * D + col, valid, 0).to(tl.float32)
    for j in tl.static_range(4):
        g = tl.load(G + base + j * D, valid, 0).to(tl.float32)
        tl.store(DP + token * 4 + j, tl.sum(b * g, 0))
        for i in tl.static_range(4):
            x = tl.load(X + base + i * D, valid, 0).to(tl.float32)
            tl.store(DM + token * 16 + i * 4 + j, tl.sum(x * g, 0))


def post_four(branch, streams, post, mixing):
    output = torch.empty_like(streams)
    n, _, d = streams.shape
    if n and d:
        _post_four[(triton.cdiv(n * d, 1024),)](
            branch, streams, post, mixing, output, n, d, BLOCK=1024, num_warps=4
        )
    return output


def post_four_input_grads(grad, post, mixing):
    n, _, d = grad.shape
    db = torch.empty((n, d), device=grad.device, dtype=grad.dtype)
    dx = torch.empty_like(grad)
    if n and d:
        _post_four_input_grads[(triton.cdiv(n * d, 1024),)](
            grad, post, mixing, db, dx, n, d, BLOCK=1024, num_warps=4
        )
    return db, dx


def post_four_coefficient_grads(grad, branch, streams):
    n, _, d = streams.shape
    dp = torch.empty((n, 4), device=streams.device, dtype=torch.float32)
    dm = torch.empty((n, 4, 4), device=streams.device, dtype=torch.float32)
    if n:
        _post_four_coefficient_grads[(n,)](
            grad,
            branch,
            streams,
            dp,
            dm,
            d,
            BLOCK=triton.next_power_of_2(max(d, 1)),
            num_warps=8 if d >= 2048 else 4,
        )
    return dp, dm
