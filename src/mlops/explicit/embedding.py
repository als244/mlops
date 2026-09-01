"""Explicit embedding lookup and table-gradient entry points."""

from __future__ import annotations

import torch

from ..dispatch import resolve_implementation


def forward(tokens: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return rows of ``weight[V, D]`` selected by integer ``tokens[*]``."""
    implementation = resolve_implementation(
        "embedding", tokens, weight, surface="explicit"
    )
    with torch.no_grad():
        return implementation.forward(tokens, weight)


def backward(
    tokens: torch.Tensor,
    grad_output: torch.Tensor,
    num_embeddings: int,
) -> torch.Tensor:
    """Return the dense ``[V, D]`` table gradient; token indices have no VJP."""
    # The table shape is enough to construct a support probe without retaining
    # the original forward weight.
    probe = grad_output.new_empty((int(num_embeddings), grad_output.shape[-1]))
    implementation = resolve_implementation(
        "embedding", tokens, probe, surface="explicit"
    )
    with torch.no_grad():
        return implementation.backward(tokens, grad_output, int(num_embeddings))


__all__ = ["backward", "forward"]
