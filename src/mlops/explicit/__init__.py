"""Autograd-independent forward and backward entry points for atomic operations.

Import an operation module and call its argument-driven ``forward`` and
``backward`` functions.  These calls never record an autograd graph or write
``Tensor.grad``; callers own saved tensors, gradient accumulation, and lifetime
management.
"""

from . import (
    causal_conv_silu,
    cross_entropy,
    dsa_attention,
    embedding,
    flash_attention,
    gated_rms_norm,
    gelu,
    head_loss,
    l2_norm,
    layer_norm,
    linear_attention,
    moe,
    packed_swiglu,
    partial_rope,
    rms_norm,
    rope,
    swiglu,
)

__all__ = [
    "causal_conv_silu",
    "cross_entropy",
    "dsa_attention",
    "embedding",
    "flash_attention",
    "gated_rms_norm",
    "gelu",
    "head_loss",
    "l2_norm",
    "layer_norm",
    "linear_attention",
    "moe",
    "packed_swiglu",
    "partial_rope",
    "rms_norm",
    "rope",
    "swiglu",
]
