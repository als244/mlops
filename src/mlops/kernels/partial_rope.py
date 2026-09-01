"""Partial-channel RoPE built on the standalone fused RoPE kernel."""

from __future__ import annotations

from .rope import rope_backward, rope_forward


def partial_rope_forward(
    x, positions, base, rotary_dim, cosine_table, sine_table
):
    output = x.clone()
    rotated = rope_forward(
        x[..., :rotary_dim].contiguous(),
        positions,
        float(base),
        cosine_table,
        sine_table,
        variant="triton_table",
    )
    output[..., :rotary_dim].copy_(rotated)
    return output


def partial_rope_backward(
    grad_output, positions, base, rotary_dim, cosine_table, sine_table
):
    grad_input = grad_output.clone()
    rotated = rope_backward(
        grad_output[..., :rotary_dim].contiguous(),
        positions,
        float(base),
        cosine_table,
        sine_table,
        variant="triton_table",
    )
    grad_input[..., :rotary_dim].copy_(rotated)
    return grad_input
