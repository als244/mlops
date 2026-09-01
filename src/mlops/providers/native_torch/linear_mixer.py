"""Per-sequence native-PyTorch projection graph."""

from __future__ import annotations

import torch

from ...dispatch.registry import Implementation, register_implementation
from ..builtin.linear_mixer import _supports


def apply(forward_segment, hidden, lengths):
    normalized = tuple(int(length) for length in lengths)
    if len(normalized) <= 1:
        return forward_segment(hidden, normalized)
    outputs = []
    start = 0
    for length in normalized:
        stop = start + length
        outputs.append(forward_segment(hidden[:, start:stop], (length,)))
        start = stop
    return torch.cat(outputs, dim=1)


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="linear_mixer",
        implementation_id="native_torch.linear_mixer.per_sequence",
        provider="native_torch",
        priority=0,
        deterministic=True,
        supports=_supports,
        apply=apply,
    )
)

__all__ = ["IMPLEMENTATION", "apply"]
