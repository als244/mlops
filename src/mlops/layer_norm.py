"""Model-facing LayerNorm semantic façade."""

from __future__ import annotations

from .dispatch import resolve_implementation


def layer_norm(x, weight, bias=None, eps=1e-5):
    """LayerNorm retaining only fresh fp32 mean and rstd rows."""
    implementation = resolve_implementation(
        "layer_norm", x, weight, bias, float(eps), surface="semantic"
    )
    return implementation.apply(x, weight, bias, float(eps))


__all__ = ["layer_norm"]
