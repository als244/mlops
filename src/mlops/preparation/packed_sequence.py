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


def prepare_from_tensor(
    like: torch.Tensor,
    lengths: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive fixed-shape metadata on the device from lengths that are data.

    Plain tensor arithmetic, so a captured graph recomputes it from its
    lengths input on every call. The offsets gain one empty sequence at the
    end, and every chunk row past the real chunks belongs to it: a kernel
    finds that sequence has no tokens and reads and writes nothing.
    """
    if lengths.ndim != 1 or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("a lengths tensor must be one-dimensional integers")
    if lengths.device != like.device:
        raise ValueError("a lengths tensor must be on the device of like")
    lengths = lengths.to(torch.int64)
    ends = torch.cumsum(lengths, 0)
    cumulative = torch.cat((ends.new_zeros(1), ends, ends[-1:]))
    chunks = (lengths + chunk_size - 1) // chunk_size
    chunk_ends = torch.cumsum(chunks, 0)
    starts = torch.cat((chunk_ends - chunks, chunk_ends[-1:]))
    bound = -(-like.numel() // chunk_size) + lengths.shape[0]
    rows = torch.arange(bound, dtype=torch.int64, device=like.device)
    sequence = torch.searchsorted(chunk_ends, rows, right=True)
    return cumulative, torch.stack((sequence, rows - starts[sequence]), dim=1)


__all__ = ["prepare", "prepare_from_tensor"]
