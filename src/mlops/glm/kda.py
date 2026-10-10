"""FLA KDA kernels behind explicit, exportable tensor-only operation boundaries.

Sequence and chunk metadata are prepared by the caller. No CUDA-to-host read,
Python autograd context, or sequence-length cache is needed inside these ops.
Q/K normalization and the GLM decay/beta gates remain ordinary outside ops.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _define_unused_triangle(A, cumulative, chunks, H: tl.constexpr):
    """FLA leaves the unused upper Aqk blocks unwritten; define op outputs."""
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    sequence = tl.load(chunks + 2 * chunk)
    local_chunk = tl.load(chunks + 2 * chunk + 1)
    begin = tl.load(cumulative + sequence)
    end = tl.load(cumulative + sequence + 1)
    row = tl.arange(0, 64)
    column = tl.arange(0, 64)
    token = begin + local_chunk * 64 + row
    offset = (token[:, None] * H + head) * 64 + column[None, :]
    mask = (token[:, None] < end) & (column[None, :] > row[:, None])
    tl.store(A + offset, 0, mask)


@torch.library.custom_op("mlops_glm::kda_forward", mutates_args=())
def _forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cumulative: torch.Tensor,
    chunks: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from .kernels.kda_execution import forward

    o, gates, aqk, akk = forward(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
        cumulative,
        chunks,
        scale,
    )
    _define_unused_triangle[(chunks.shape[0], q.shape[1])](
        aqk, cumulative, chunks, q.shape[1]
    )
    return o.squeeze(0), gates.squeeze(0), aqk.squeeze(0), akk.squeeze(0)


@_forward.register_fake
def _fake(q, k, v, g, beta, cumulative, chunks, scale):
    matrices = (*beta.shape, 64)
    return (
        torch.empty_like(v),
        torch.empty_like(g, dtype=torch.float32),
        q.new_empty(matrices),
        q.new_empty(matrices),
    )


@torch.library.custom_op("mlops_glm::kda_backward", mutates_args=())
def _backward(
    dy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    gates: torch.Tensor,
    aqk: torch.Tensor,
    akk: torch.Tensor,
    cumulative: torch.Tensor,
    chunks: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from .kernels.kda_execution import backward

    dq, dk, dv, dg, db = backward(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        beta.unsqueeze(0),
        gates.unsqueeze(0),
        aqk.unsqueeze(0),
        akk.unsqueeze(0),
        dy.contiguous().unsqueeze(0),
        cumulative,
        chunks,
        scale,
    )
    return (
        dq.squeeze(0).to(q),
        dk.squeeze(0).to(k),
        dv.squeeze(0).to(v),
        dg.squeeze(0).to(g),
        db.squeeze(0).to(beta),
    )


@_backward.register_fake
def _fake_bwd(dy, q, k, v, g, beta, gates, aqk, akk, cumulative, chunks, scale):
    return tuple(torch.empty_like(t) for t in (q, k, v, g, beta))


def _setup(ctx, inputs, output):
    q, k, v, g, beta, cumulative, chunks, ctx.scale = inputs
    _, gates, aqk, akk = output
    ctx.save_for_backward(q, k, v, g, beta, gates, aqk, akk, cumulative, chunks)
    ctx.mark_non_differentiable(gates, aqk, akk)


def _autograd(ctx, dy, *_):
    return (*_backward(dy, *ctx.saved_tensors, ctx.scale), None, None, None)


_forward.register_autograd(_autograd, setup_context=_setup)


def kimi_delta_attention(q, k, v, g, beta, cumulative, chunks):
    if (
        q.ndim != 3
        or v.ndim != 3
        or q.shape != k.shape
        or q.shape != g.shape
        or q.shape[:2] != v.shape[:2]
    ):
        raise ValueError(
            "Expected packed [tokens, heads, width] tensors with matching Q/K/g"
        )
    if beta.shape != q.shape[:2] or q.shape[-1] not in (32, 64, 128, 256):
        raise ValueError("Invalid beta shape or unsupported KDA key width")
    if not all(
        t.is_cuda and t.device == q.device
        for t in (q, k, v, g, beta, cumulative, chunks)
    ):
        raise ValueError("KDA payloads and metadata must share one CUDA device")
    if (
        q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise ValueError("KDA Q/K/V must use the same BF16 or FP16 dtype")
    if g.dtype != torch.float32:
        raise ValueError("KDA log decay must be FP32")
    if beta.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("KDA beta must use a floating-point activation dtype")
    if (
        cumulative.ndim != 1
        or cumulative.numel() < 2
        or chunks.ndim != 2
        or chunks.shape[1] != 2
        or chunks.shape[0] == 0
        or cumulative.dtype not in (torch.int32, torch.int64)
        or chunks.dtype not in (torch.int32, torch.int64)
        or not cumulative.is_contiguous()
        or not chunks.is_contiguous()
    ):
        raise ValueError(
            "KDA metadata requires contiguous integer boundaries [sequences+1] "
            "and chunk pairs [chunks,2]"
        )
    return _forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        cumulative,
        chunks,
        q.shape[-1] ** -0.5,
    )[0]
