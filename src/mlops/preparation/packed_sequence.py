"""Graph-visible materialization of host-derived packed metadata."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _materialize(
    lengths: Sequence[int],
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = tuple(int(length) for length in lengths)
    if len(normalized) == 1:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty((0, 2), dtype=torch.int64, device=device),
        )

    boundaries = [0]
    for length in normalized:
        boundaries.append(boundaries[-1] + length)
    pairs = [
        (sequence, chunk)
        for sequence, length in enumerate(normalized)
        for chunk in range((length + chunk_size - 1) // chunk_size)
    ]
    cumulative_host = torch.tensor(boundaries, dtype=torch.int64)
    chunks_host = torch.tensor(pairs, dtype=torch.int64).reshape(-1, 2)
    if device.type == "cuda":
        cumulative_host = cumulative_host.pin_memory()
        chunks_host = chunks_host.pin_memory()
    return (
        cumulative_host.to(device, non_blocking=device.type == "cuda"),
        chunks_host.to(device, non_blocking=device.type == "cuda"),
    )


@torch.library.custom_op(
    "mlops::prepare_packed_sequence_metadata",
    mutates_args=(),
)
def _prepare_op(
    like: torch.Tensor,
    lengths: list[int],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize call-derived metadata on ``like.device`` without state."""
    return _materialize(lengths, like.device, int(chunk_size))


@_prepare_op.register_fake
def _prepare_fake(like, lengths, chunk_size):
    chunks = 0 if len(lengths) == 1 else sum(
        (int(length) + int(chunk_size) - 1) // int(chunk_size)
        for length in lengths
    )
    cumulative = 0 if len(lengths) == 1 else len(lengths) + 1
    return (
        torch.empty(cumulative, dtype=torch.int64, device=like.device),
        torch.empty((chunks, 2), dtype=torch.int64, device=like.device),
    )


def prepare(
    like: torch.Tensor,
    lengths: Sequence[int],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the private compiler-visible preparation target."""
    return _prepare_op(like, list(lengths), int(chunk_size))


__all__ = ["prepare"]
