"""Model-facing packed causal-convolution semantic façade."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .dispatch import resolve_implementation


def causal_conv_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    lengths: Sequence[int],
    cumulative: torch.Tensor,
    chunk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply sequence-reset depthwise causal convolution and SiLU.

    ``cumulative`` and ``chunk_indices`` are caller-owned packed-round
    metadata returned by :func:`mlops.prepare_packed_sequence_metadata`.  A
    single sequence uses empty INT64 tensors.  Supplying ``chunk_indices``
    avoids reconstructing them from CUDA data inside provider kernels.
    """
    lengths = tuple(int(length) for length in lengths)
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[0] != sum(lengths):
        raise ValueError("causal-conv sequence lengths do not match tokens")
    implementation = resolve_implementation(
        "causal_conv_silu",
        x,
        weight,
        lengths,
        cumulative,
        chunk_indices,
        surface="semantic",
    )
    return implementation.apply(x, weight, lengths, cumulative, chunk_indices)


__all__ = ["causal_conv_silu"]
