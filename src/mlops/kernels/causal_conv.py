"""Depthwise causal conv1d + SiLU with packed-sequence resets."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def fla_causal_conv_forward(x, weight, cumulative, chunk_indices=None):
    from fla.modules.conv.triton.ops import causal_conv1d_fwd

    sequence_metadata = None if cumulative.numel() == 0 else cumulative
    output, _final = causal_conv1d_fwd(
        x.contiguous().unsqueeze(0),
        weight.squeeze(1).contiguous(),
        bias=None,
        residual=None,
        activation="silu",
        cu_seqlens=sequence_metadata,
        chunk_indices=(
            None
            if sequence_metadata is None or chunk_indices is None
            else chunk_indices
        ),
    )
    return output.squeeze(0)


def fla_causal_conv_backward(
    grad_output, x, weight, cumulative, chunk_indices=None
):
    from fla.modules.conv.triton.ops import causal_conv1d_bwd

    grad_x, grad_weight, _bias, _residual, _initial = causal_conv1d_bwd(
        x.contiguous().unsqueeze(0),
        grad_output.contiguous().unsqueeze(0),
        dht=None,
        weight=weight.squeeze(1).contiguous(),
        bias=None,
        residual=None,
        activation="silu",
        cu_seqlens=None if cumulative.numel() == 0 else cumulative,
        chunk_indices=(
            None
            if cumulative.numel() == 0 or chunk_indices is None
            else chunk_indices
        ),
    )
    return grad_x.squeeze(0), grad_weight.unsqueeze(1)


def causal_conv_forward(x, weight, lengths):
    """``x`` is flattened token-major; ``weight`` is ``(channels, 1, W)``."""
    outputs = []
    start = 0
    width = weight.shape[-1]
    for length in lengths:
        stop = start + int(length)
        segment = x[start:stop].T.unsqueeze(0).float()
        padded = functional.pad(segment, (width - 1, 0))
        convolved = functional.conv1d(padded, weight.float(), groups=x.shape[-1])
        outputs.append(
            functional.silu(convolved.transpose(1, 2).squeeze(0)).to(x.dtype)
        )
        start = stop
    return torch.cat(outputs, dim=0)
