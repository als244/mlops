"""Clipped SwiGLU with one Triton forward kernel and one backward kernel."""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(
    X, Y, N: tl.constexpr, H: tl.constexpr, LIMIT: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    offset = (i // H) * (2 * H) + i % H
    g = tl.load(X + offset, i < N, 0).to(tl.float32)
    u = tl.load(X + offset + H, i < N, 0).to(tl.float32)
    limit = tl.full((), LIMIT, tl.float32).to(X.dtype.element_ty).to(tl.float32)
    g = tl.minimum(g, limit)
    u = tl.minimum(tl.maximum(u, -limit), limit)
    a = (g * tl.sigmoid(g)).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + i, a * u, i < N)


@triton.jit
def _backward(
    X, DY, DX, N: tl.constexpr, H: tl.constexpr, LIMIT: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    offset = (i // H) * (2 * H) + i % H
    raw_g = tl.load(X + offset, i < N, 0).to(tl.float32)
    raw_u = tl.load(X + offset + H, i < N, 0).to(tl.float32)
    limit = tl.full((), LIMIT, tl.float32).to(X.dtype.element_ty).to(tl.float32)
    g = tl.minimum(raw_g, limit)
    u = tl.minimum(tl.maximum(raw_u, -limit), limit)
    dy = tl.load(DY + i, i < N, 0).to(tl.float32)
    sigmoid = tl.sigmoid(g)
    a = (g * sigmoid).to(X.dtype.element_ty).to(tl.float32)
    dact = (dy * u).to(X.dtype.element_ty).to(tl.float32)
    dg = dact * sigmoid * (1 + g * (1 - sigmoid))
    du = dy * a
    tl.store(DX + offset, tl.where(raw_g <= limit, dg, 0), i < N)
    tl.store(
        DX + offset + H, tl.where((raw_u >= -limit) & (raw_u <= limit), du, 0), i < N
    )


@torch.library.custom_op("mlops_glm::clipped_swiglu", mutates_args=())
def _op(x: torch.Tensor, limit: float) -> torch.Tensor:
    y = x.new_empty((*x.shape[:-1], x.shape[-1] // 2))
    _forward[(triton.cdiv(y.numel(), 256),)](x, y, y.numel(), y.shape[-1], limit, 256)
    return y


@_op.register_fake
def _fake(x, limit):
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


@torch.library.custom_op("mlops_glm::clipped_swiglu_backward", mutates_args=())
def _bwd(x: torch.Tensor, dy: torch.Tensor, limit: float) -> torch.Tensor:
    dx = torch.empty_like(x)
    _backward[(triton.cdiv(dy.numel(), 256),)](
        x, dy.contiguous(), dx, dy.numel(), dy.shape[-1], limit, 256
    )
    return dx


@_bwd.register_fake
def _fake_bwd(x, dy, limit):
    return torch.empty_like(x)


def _setup(ctx, inputs, output):
    x, ctx.limit = inputs
    ctx.save_for_backward(x)


def _autograd(ctx, dy):
    return _bwd(ctx.saved_tensors[0], dy, ctx.limit), None


_op.register_autograd(_autograd, setup_context=_setup)


def clipped_swiglu(x, limit=10.0):
    if not x.is_cuda or x.ndim < 1 or x.shape[-1] % 2:
        raise ValueError("Expected a CUDA tensor with packed gate/up on its final axis")
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16) or limit <= 0:
        raise ValueError("Expected floating input and a positive clipping limit")
    return _op(x.contiguous(), float(limit))
