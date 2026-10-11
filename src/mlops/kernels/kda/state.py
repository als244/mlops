"""FLA state kernels with call-local packed-sequence metadata.

Launch setup adapted from FLA 0.5.2 common/chunk_delta_h.py (MIT).
Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li.
See FLA_LICENSE. The numerical kernels are imported, not copied.

The upstream convenience functions cache tensor arguments by identity. Tensor
identity does not imply immutable contents when an execution engine reuses
storage. Compute the small offset array for each call and retain no tensors.
Scope: packed KDA, chunk size 64, key-first layout, zero initial state.
"""

import torch
import triton
import triton.language as tl
from torch.nn import functional as F


@triton.jit
def _offsets_kernel(
    cumulative, offsets, N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.arange(0, BLOCK)
    end = tl.load(cumulative + row * STRIDE, row < N, other=0).to(tl.int64)
    start = tl.load(cumulative + (row - 1) * STRIDE, (row > 0) & (row < N), other=0).to(
        tl.int64
    )
    counts = tl.where((row > 0) & (row < N), (end - start + 63) // 64, 0)
    tl.store(offsets + row, tl.cumsum(counts), row < N)


def _offsets(cumulative):
    # One launch for ordinary packing tables. Bound the scan's per-block
    # resources for unusually large sequence counts; neither path reads the
    # table on the host or retains it after the call.
    count = cumulative.numel()
    if count > 1024:
        counts = (cumulative[1:] - cumulative[:-1] + 63) // 64
        return F.pad(counts, (1, 0)).cumsum(0)
    offsets = torch.empty(count, device=cumulative.device, dtype=torch.int64)
    _offsets_kernel[(1,)](
        cumulative, offsets, count, cumulative.stride(0), triton.next_power_of_2(count)
    )
    return offsets


def forward(k, w, u, gates, cumulative, chunks):
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )

    batch, tokens, heads, width = k.shape
    value_heads, value_width = u.shape[2:]
    offsets = _offsets(cumulative)
    h = k.new_empty(batch, len(chunks), value_heads, width, value_width)
    v_new = torch.empty_like(u)

    def grid(meta):
        return triton.cdiv(value_width, meta["BV"]), (len(cumulative) - 1) * value_heads

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=None,
        gk=gates,
        h=h,
        h0=None,
        ht=None,
        cu_seqlens=cumulative,
        chunk_offsets=offsets,
        T=tokens,
        H=heads,
        HV=value_heads,
        K=width,
        V=value_width,
        BT=64,
        STATE_V_FIRST=False,
    )
    return h, v_new


def backward(q, k, w, gates, dy, dv, cumulative, chunks, scale):
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
    )

    batch, tokens, heads, width = q.shape
    value_heads, value_width = dy.shape[2:]
    offsets = _offsets(cumulative)
    dh = q.new_empty(batch, len(chunks), value_heads, width, value_width)
    dv_new = torch.empty_like(dv)

    def grid(meta):
        return triton.cdiv(value_width, meta["BV"]), (len(cumulative) - 1) * value_heads

    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=None,
        gk=gates,
        dht=None,
        dh0=None,
        do=dy,
        dh=dh,
        dv=dv,
        dv2=dv_new,
        cu_seqlens=cumulative,
        chunk_offsets=offsets,
        scale=scale,
        T=tokens,
        H=heads,
        HV=value_heads,
        K=width,
        V=value_width,
        BT=64,
        STATE_V_FIRST=False,
    )
    return dh, dv_new
