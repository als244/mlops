"""FLA gated-delta compositions with private scratch released at last use.

Adapted from fla-core 0.5.2, fla/ops/gated_delta_rule/chunk.py,
chunk_gated_delta_rule_fwd and chunk_gated_delta_rule_bwd (source SHA256s:
forward: dae73a3e734dfe069747b45286279dad19fa77e372c4eb4d3ee0456549ba19ad;
backward: 5b342b42f3caa97af4ac43f6a04398152114252abc5cc375e47268fc590a5179).
Only the existing MLOps contract is exposed: packed sequences, 64-token
recurrence tiles, zero initial state and no final-state gradient. Kernels,
their order and arithmetic are unchanged, including FLA's backend dispatch.

Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def forward(q, k, v, beta, a, a_log, dt_bias, cumulative, chunks, scale):
    """Return output and paired residuals without retaining dead WY factors."""
    from fla.ops.gated_delta_rule import chunk as fla

    gate = fla.gdn_gate_chunk_cumsum(
        g=a,
        A_log=a_log,
        chunk_size=64,
        scale=fla.RCP_LN2,
        dt_bias=dt_bias,
        cu_seqlens=cumulative,
        chunk_indices=chunks,
    )
    w, u, matrix = fla.chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=gate,
        beta=beta,
        cu_seqlens=cumulative,
        chunk_indices=chunks,
        chunk_size=64,
    )
    h, v_new, _final = fla.chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=gate,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cumulative,
        chunk_indices=chunks,
        state_v_first=False,
        chunk_size=64,
    )
    # The output kernel uses h and v_new; neither WY factor has another reader.
    del w, u
    output = fla.chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=gate,
        scale=scale,
        cu_seqlens=cumulative,
        chunk_indices=chunks,
        state_v_first=False,
        chunk_size=64,
    )
    return output, gate, matrix


def backward(
    q, k, v, g, beta, matrix, grad, a, a_log, dt_bias, cumulative, chunks, scale
):
    """Consume contiguous batched operands and immutable forward residuals."""
    from fla.ops.gated_delta_rule import chunk as fla

    w, u = fla.recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=matrix,
        g=g,
        cu_seqlens=cumulative,
        chunk_indices=chunks,
    )
    h, v_new, _final = fla.chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cumulative,
        chunk_indices=chunks,
        state_v_first=False,
        chunk_size=64,
    )
    # u is private replay scratch; the state reconstruction is its last reader.
    del u
    dv = fla.chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=grad,
        scale=scale,
        cu_seqlens=cumulative,
        chunk_indices=chunks,
        chunk_size=64,
    )
    dh, _initial, dv = fla.chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=None,
        dht=None,
        do=grad,
        dv=dv,
        scale=scale,
        cu_seqlens=cumulative,
        chunk_indices=chunks,
        state_v_first=False,
        chunk_size=64,
    )
    dq, dk, dw, dg = fla.chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=grad,
        dh=dh,
        scale=scale,
        cu_seqlens=cumulative,
        chunk_indices=chunks,
        state_v_first=False,
        chunk_size=64,
    )
    # WY backward needs dw and dv, but no longer needs these large histories.
    del w, h, v_new, dh
    dk2, dv, db, dg2 = fla.prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=matrix,
        dw=dw,
        du=dv,
        cu_seqlens=cumulative,
        chunk_indices=chunks,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    dg = fla.chunk_local_cumsum(
        dg,
        chunk_size=64,
        reverse=True,
        cu_seqlens=cumulative,
        chunk_indices=chunks,
    )
    da, da_log, ddt_bias = fla.gdn_gate_bwd(
        g=a,
        A_log=a_log,
        dt_bias=dt_bias,
        dyg=dg,
    )
    return dq, dk, dv, db, da, da_log, ddt_bias
