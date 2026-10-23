"""Private epilogue for the four-stream mHC preprocessing VJP."""

import torch
import triton
import triton.language as tl


@triton.jit
def _project_finish(
    DU,
    W,
    X,
    SQUARE_GRAD,
    DIRECT,
    OUT,
    TOKENS: tl.constexpr,
    WIDTH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, 32)
    du = tl.load(
        DU + rows[:, None] * 24 + kk[None, :],
        (rows[:, None] < TOKENS) & (kk[None, :] < 24),
        0,
    )
    weight = tl.load(
        W + kk[:, None] * WIDTH + cols[None, :],
        (kk[:, None] < 24) & (cols[None, :] < WIDTH),
        0,
    ).to(tl.float32)
    projected = tl.dot(du, weight, input_precision="tf32x3")
    offset = rows[:, None] * WIDTH + cols[None, :]
    mask = (rows[:, None] < TOKENS) & (cols[None, :] < WIDTH)
    x = tl.load(X + offset, mask, 0).to(tl.float32)
    square_grad = tl.load(SQUARE_GRAD + rows, rows < TOKENS, 0)
    direct = tl.load(DIRECT + offset, mask, 0).to(tl.float32)
    normalized = (
        (projected + 2.0 * x * square_grad[:, None]).to(tl.bfloat16).to(tl.float32)
    )
    tl.store(OUT + offset, normalized + direct, mask)


def project_input_gradient(du, projection, streams, square_grad, direct):
    result = torch.empty_like(streams)
    tokens, _, hidden = streams.shape
    width = 4 * hidden
    _project_finish[(triton.cdiv(tokens, 32), triton.cdiv(width, 128))](
        du,
        projection.contiguous(),
        streams,
        square_grad,
        direct,
        result,
        tokens,
        width,
        32,
        128,
        num_warps=4,
    )
    return result
