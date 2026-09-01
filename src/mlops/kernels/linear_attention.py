"""Gated delta-rule recurrence used by Qwen3.5 linear-attention layers."""

from __future__ import annotations

import torch


def fla_linear_attention_forward(
    q,
    k,
    v,
    beta,
    a,
    a_log,
    dt_bias,
    cumulative,
    chunk_indices,
    scale,
):
    """Run the same low-level FLA chunk kernel as the production path."""
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

    sequence_metadata = None if cumulative.numel() == 0 else cumulative
    chunk_metadata = None if chunk_indices.numel() == 0 else chunk_indices
    gate, output, matrix, _final, _initial, _input_gate = chunk_gated_delta_rule_fwd(
        q.contiguous().unsqueeze(0),
        k.contiguous().unsqueeze(0),
        v.contiguous().unsqueeze(0),
        a.contiguous().unsqueeze(0),
        beta.contiguous().unsqueeze(0),
        scale=float(scale),
        initial_state=None,
        output_final_state=False,
        cu_seqlens=sequence_metadata,
        chunk_indices=chunk_metadata,
        use_gate_in_kernel=True,
        A_log=a_log.float(),
        dt_bias=dt_bias.float(),
    )
    return output.squeeze(0).to(v.dtype), gate, matrix


def fla_linear_attention_backward(
    grad_output,
    q,
    k,
    v,
    beta,
    a,
    a_log,
    dt_bias,
    gate,
    matrix,
    cumulative,
    chunk_indices,
    scale,
):
    """Run FLA's paired backward with its forward intermediates."""
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd

    dq, dk, dv, dbeta, da, _initial, da_log, ddt_bias = chunk_gated_delta_rule_bwd(
        q=q.contiguous().unsqueeze(0),
        k=k.contiguous().unsqueeze(0),
        v=v.contiguous().unsqueeze(0),
        g=gate,
        beta=beta.contiguous().unsqueeze(0),
        A=matrix,
        scale=float(scale),
        initial_state=None,
        do=grad_output.contiguous().unsqueeze(0),
        dht=None,
        cu_seqlens=cumulative,
        chunk_indices=chunk_indices,
        use_gate_in_kernel=True,
        g_input=a.contiguous().unsqueeze(0),
        A_log=a_log.float(),
        dt_bias=dt_bias.float(),
    )
    return (
        dq.squeeze(0),
        dk.squeeze(0),
        dv.squeeze(0),
        dbeta.squeeze(0),
        da.squeeze(0),
        da_log,
        ddt_bias,
    )


def _segment_delta(q, k, v, beta, a, a_log, dt_bias, scale):
    key_heads, key_dim = q.shape[1], q.shape[2]
    value_heads, value_dim = v.shape[1], v.shape[2]
    repeat = value_heads // key_heads
    q_float = q.float().repeat_interleave(repeat, dim=1)
    k_float = k.float().repeat_interleave(repeat, dim=1)
    v_float = v.float()
    beta_float = beta.float()
    decay = -a_log.float().exp() * torch.nn.functional.softplus(
        a.float() + dt_bias.float()
    )
    state = torch.zeros(
        value_heads,
        key_dim,
        value_dim,
        dtype=torch.float32,
        device=q.device,
    )
    outputs = []
    for index in range(q.shape[0]):
        state = state * decay[index].exp()[:, None, None]
        error = v_float[index] - torch.einsum("hk,hkv->hv", k_float[index], state)
        update = beta_float[index][:, None] * error
        state = state + k_float[index][:, :, None] * update[:, None, :]
        outputs.append(torch.einsum("hk,hkv->hv", q_float[index] * float(scale), state))
    return torch.stack(outputs).to(v.dtype)


def linear_attention_forward(q, k, v, beta, a, a_log, dt_bias, lengths, scale):
    outputs = []
    start = 0
    for length in lengths:
        stop = start + int(length)
        outputs.append(
            _segment_delta(
                q[start:stop],
                k[start:stop],
                v[start:stop],
                beta[start:stop],
                a[start:stop],
                a_log,
                dt_bias,
                scale,
            )
        )
        start = stop
    return torch.cat(outputs, dim=0)
