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


def causal_conv_silu_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight: torch.Tensor,
    cumulative: torch.Tensor,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convolve separate token-major Q/K/V projections with packed resets.

    Inputs are [tokens, channels] matrices (channel counts may differ). Weight
    rows are Q, then K, then V. Return three tensors with their input shapes.
    Frozen-weight providers can preserve separate gradients across backward;
    trainable weights and unsupported widths use the ordinary composition.
    Q/K normalization and projection/gate parameters remain caller-owned.
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("Q/K/V must be token-major matrices")
    if any(
        t.shape[0] != q.shape[0] or t.device != q.device or t.dtype != q.dtype
        for t in (k, v)
    ):
        raise ValueError("Q/K/V must share token count, device and dtype")
    channels = q.shape[1] + k.shape[1] + v.shape[1]
    if (
        any(t.shape[1] <= 0 for t in (q, k, v))
        or weight.ndim != 3
        or weight.shape[:2] != (channels, 1)
        or weight.shape[-1] <= 0
    ):
        raise ValueError(
            "Expected [Q+K+V channels, 1, positive width] convolution weights"
        )
    if (
        weight.device != q.device
        or cumulative.device != q.device
        or (chunk_indices is not None and chunk_indices.device != q.device)
    ):
        raise ValueError(
            "Convolution tensors and sequence metadata must share a device"
        )
    implementation = resolve_implementation(
        "causal_conv_silu_qkv",
        q,
        k,
        v,
        weight,
        cumulative,
        chunk_indices,
        surface="semantic",
    )
    return implementation.apply(q, k, v, weight, cumulative, chunk_indices)


__all__ = ["causal_conv_silu", "causal_conv_silu_qkv"]
