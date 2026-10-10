"""FLA composition with explicit IEEE precision for intra-chunk arithmetic.

Based on FLA 0.5.2 chunk_bwd.py and chunk_intra.py (MIT license).
Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li.
See FLA_LICENSE and docs/KERNELS.md.

Numerical kernels come from FLA; the isolated intra-chunk copy explicitly
uses IEEE FP32 dots. No process-wide precision settings or installed files
are changed. Scope: packed KDA,
equal query/value head counts, zero initial state, no final-state gradient.
"""

import torch
import triton


def _intra(q, k, gates, beta, daqk, dakk, dq, dk, dbeta, dg, cumulative, chunks):
    from .kda_intra import (
        IS_GATHER_SUPPORTED,
        chunk_kda_bwd_kernel_intra,
    )

    batch, tokens, heads, width = k.shape
    chunk_size, subchunk_size = 64, 16
    tile_width = min(32, triton.next_power_of_2(width))
    subchunks = chunk_size // subchunk_size
    width_tiles = triton.cdiv(width, tile_width)
    dq_out = torch.empty_like(dq)
    dk_out = torch.empty_like(dk)
    dbeta_parts = beta.new_empty(width_tiles, *beta.shape, dtype=torch.float32)
    dg_out = torch.empty_like(dg, dtype=torch.float32)
    chunk_kda_bwd_kernel_intra[(width_tiles * subchunks, len(chunks), batch * heads)](
        q=q,
        k=k,
        g=gates,
        beta=beta,
        dAqk=daqk,
        dAkk=dakk,
        dq=dq,
        dq2=dq_out,
        dk=dk,
        dk2=dk_out,
        dg=dg,
        dg2=dg_out,
        db=dbeta_parts,
        cu_seqlens=cumulative,
        chunk_indices=chunks,
        B=batch,
        T=tokens,
        H=heads,
        HV=heads,
        K=width,
        BT=chunk_size,
        BC=subchunk_size,
        BK=tile_width,
        NC=subchunks,
        SAFE_GATE=True,
        USE_GATHER=IS_GATHER_SUPPORTED,
    )
    return dq_out, dk_out, dbeta_parts.sum(0).add_(dbeta), dg_out


def backward(q, k, v, beta, gates, aqk, akk, dy, cumulative, chunks, scale):
    """Return dq, dk, dv, dGate, dBeta, preserving upstream storage dtypes."""
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu,
        chunk_gated_delta_rule_fwd_h,
    )
    from fla.ops.kda.chunk_bwd import (
        chunk_kda_bwd_dAv,
        chunk_kda_bwd_wy_dqkg_fused,
    )
    from fla.ops.kda.wy_fast import recompute_w_u_fwd
    from fla.ops.utils import chunk_local_cumsum

    metadata = {"cu_seqlens": cumulative, "chunk_indices": chunks}
    w, u, qg, kg = recompute_w_u_fwd(
        q=q, k=k, v=v, beta=beta, A=akk, gk=gates, **metadata
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=gates,
        initial_state=None,
        output_final_state=False,
        chunk_size=64,
        state_v_first=False,
        **metadata,
    )
    daqk, dv = chunk_kda_bwd_dAv(
        q=q, k=k, v=v_new, do=dy, A=aqk, scale=scale, chunk_size=64, **metadata
    )
    dh, _, dv = chunk_gated_delta_rule_bwd_dhu(
        q=qg,
        k=kg,
        w=w,
        gk=gates,
        h0=None,
        dht=None,
        do=dy,
        dv=dv,
        scale=scale,
        chunk_size=64,
        state_v_first=False,
        **metadata,
    )
    dq, dk, dv, dbeta, dg, dakk = chunk_kda_bwd_wy_dqkg_fused(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=gates,
        beta=beta,
        A=akk,
        h=h,
        do=dy,
        dh=dh,
        dv=dv,
        scale=scale,
        chunk_size=64,
        state_v_first=False,
        **metadata,
    )
    dq, dk, dbeta, dg = _intra(
        q, k, gates, beta, daqk, dakk, dq, dk, dbeta, dg, cumulative, chunks
    )
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, **metadata)
    return dq, dk, dv, dg, dbeta


def forward(q, k, v, g, beta, cumulative, chunks, scale):
    """Packed KDA using FP32 gates and explicit IEEE intra-chunk arithmetic."""
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2

    from .kda_intra import chunk_kda_fwd_intra

    metadata = {"cu_seqlens": cumulative, "chunk_indices": chunks}
    gates = chunk_local_cumsum(g=g, scale=RCP_LN2, chunk_size=64, **metadata)
    w, u, _qg, kg, aqk, akk = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=gates,
        beta=beta,
        scale=scale,
        chunk_size=64,
        safe_gate=True,
        disable_recompute=False,
        **metadata,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=gates,
        initial_state=None,
        output_final_state=False,
        chunk_size=64,
        state_v_first=False,
        **metadata,
    )
    output = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=gates,
        A=aqk,
        h=h,
        scale=scale,
        chunk_size=64,
        state_v_first=False,
        **metadata,
    )
    return output, gates, aqk, akk
