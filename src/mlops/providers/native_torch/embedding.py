"""Ordinary native-PyTorch embedding graph."""

from __future__ import annotations

import torch.nn.functional as functional

from ...dispatch.registry import Implementation, register_implementation
from ..builtin.embedding import _supports, backward, forward


def apply(tokens, weight):
    return functional.embedding(tokens, weight)


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="embedding",
        implementation_id="native_torch.embedding",
        provider="native_torch",
        priority=0,
        deterministic=True,
        supports=_supports,
        apply=apply,
        forward=forward,
        backward=backward,
    )
)

__all__ = ["IMPLEMENTATION", "apply", "backward", "forward"]
