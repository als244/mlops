"""Ordinary native-PyTorch tanh-GELU graph."""

from __future__ import annotations

import torch.nn.functional as functional

from ...dispatch.registry import Implementation, register_implementation
from ..builtin.gelu import _supports, backward, forward


def apply(x):
    return functional.gelu(x, approximate="tanh")


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="gelu",
        implementation_id="native_torch.gelu",
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
