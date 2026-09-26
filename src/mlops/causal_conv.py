"""Model-facing packed causal-convolution semantic façade."""

from __future__ import annotations

import torch

from .dispatch import resolve_implementation


def causal_conv_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    cumulative: torch.Tensor,
    chunk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply sequence-reset depthwise causal convolution and SiLU.

    ``cumulative`` and ``chunk_indices`` are caller-owned packed-round
    metadata returned by :func:`mlops.prepare_packed_sequence_metadata`.  A
    single sequence uses empty INT64 tensors.  Supplying ``chunk_indices``
    avoids reconstructing them from CUDA data inside provider kernels.  The
    metadata is data: a captured graph takes it as an input, so one graph
    serves every packing of the same tokens.
    """
    implementation = resolve_implementation(
        "causal_conv_silu",
        x,
        weight,
        cumulative,
        chunk_indices,
        surface="semantic",
    )
    return implementation.apply(x, weight, cumulative, chunk_indices)


__all__ = ["causal_conv_silu"]
