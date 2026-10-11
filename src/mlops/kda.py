"""Packed Kimi Delta Attention; projections and gate parameters are caller-owned."""

import torch

from .dispatch import resolve_implementation


def _validate(q, k, v, g, beta, cumulative, chunks):
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


def kimi_delta_attention(q, k, v, g, beta, cumulative, chunks):
    """Return packed KDA values with explicit CUDA sequence/chunk metadata."""
    _validate(q, k, v, g, beta, cumulative, chunks)
    implementation = resolve_implementation(
        "kimi_delta_attention",
        q,
        k,
        v,
        g,
        beta,
        cumulative,
        chunks,
        surface="semantic",
    )
    return implementation.apply(q, k, v, g, beta, cumulative, chunks)


def kda_decay(raw, dt_bias, a_log, *, lower_bound=-5.0):
    """Map [T,H,D] projected gates to FP32 log-decays in [lower_bound, 0]."""
    if raw.ndim != 3 or dt_bias.numel() != raw.shape[1] * raw.shape[2]:
        raise ValueError(
            "Expected [tokens, heads, width] gates and one bias per channel"
        )
    if a_log.shape != (raw.shape[1],) or lower_bound >= 0:
        raise ValueError(
            "Expected one decay parameter per head and a negative lower bound"
        )
    biased = raw.float() + dt_bias.float().reshape(1, raw.shape[1], raw.shape[2])
    return lower_bound * (a_log.float().exp()[None, :, None] * biased).sigmoid()


__all__ = ["kda_decay", "kimi_delta_attention"]
