"""Public caller-owned packed-sequence metadata preparation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .preparation.packed_sequence import prepare, prepare_from_tensor


def prepare_packed_sequence_metadata(
    lengths: Sequence[int] | torch.Tensor,
    like: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return caller-owned cumulative lengths and chunk indices.

    Both tensors use INT64 and live on ``like.device``. A single sequence uses
    empty tensors because FLA's non-varlen path expects ``None``; provider
    adapters convert the empty sentinels without reading device data. CUDA
    copies originate in pinned host memory and are non-blocking.

    ``lengths`` may instead be a one-dimensional integer tensor on
    ``like.device``, zero-padded to a fixed number of sequences. The lengths
    are then data: the metadata is derived from them by graph-visible tensor
    arithmetic, in shapes fixed by ``like`` and the tensor alone, so one
    captured graph serves every packing of the same tokens. ``like`` must then
    be the packed tokens, since their count ``T`` bounds the chunks: there is
    one chunk row for each chunk ``N`` sequences of ``T`` tokens could need,
    ``ceil(T / chunk_size) + N``. Rows past the real chunks belong to one
    more, empty sequence at the end, where kernels read and write nothing.

    Call this once per packed round and pass the returned tensors to every
    applicable operation. The graph-visible preparation target retains no
    inputs or outputs and has no autograd formula because its integer outputs
    are non-differentiable.
    """
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not isinstance(like, torch.Tensor):
        raise TypeError("like must be a tensor whose device owns the metadata")
    if isinstance(lengths, torch.Tensor):
        return prepare_from_tensor(like, lengths, chunk_size)
    normalized = tuple(int(length) for length in lengths)
    if not normalized or any(length <= 0 for length in normalized):
        raise ValueError("lengths must contain positive sequence lengths")
    return prepare(like, normalized, chunk_size)


__all__ = ["prepare_packed_sequence_metadata"]
