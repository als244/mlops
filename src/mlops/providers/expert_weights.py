"""FP8 expert weight containers as the bf16 implementations meet them.

`QuackFP8Weight` (``expert_parallel/quack/parameters/fp8.py``) holds an
expert weight's E4M3 rows and transpose with their descales; `BlockFP8Weight`
(``mlops/block_fp8.py``) its E4M3 values once with a scale per 128x128 block.
Implementations
that compute in the weights' dtype take such weights only when asked for
``compute_precision="bf16"``, dequantizing them per call; by default FP8
weights mean FP8 math, which only the SonicMoE implementations serve.
"""

from __future__ import annotations


def _containers():
    from ..block_fp8 import BlockFP8Weight
    from ..expert_parallel.quack.parameters.fp8 import QuackFP8Weight

    return QuackFP8Weight, BlockFP8Weight


def fp8_expert_weights(*weights) -> bool:
    """Whether any expert weight is an FP8 container (`QuackFP8Weight` or
    `BlockFP8Weight`)."""
    return any(isinstance(weight, _containers()) for weight in weights)


def bf16_compute_support(compute_precision, w13_experts, w2_experts):
    """Why an implementation that computes in the weights' dtype declines a
    request, or ``None``: it serves ``compute_precision`` ``None`` for plain
    weights and ``"bf16"`` for any, dequantizing FP8 containers per call; FP8
    containers default to FP8 math, which it has not."""
    if compute_precision not in (None, "bf16"):
        return (
            f"this implementation computes in the weights' dtype; "
            f"compute_precision={compute_precision!r} is not available here"
        )
    if compute_precision is None and fp8_expert_weights(w13_experts, w2_experts):
        return (
            "FP8 expert weights compute in FP8 by default; ask "
            "compute_precision='bf16' to dequantize them here"
        )
    return None


def dequantize_expert_weights(w13_experts, w2_experts, *, dtype):
    """The expert weights as plain tensors at ``dtype``: FP8 containers
    dequantized (straight through for their gradients), others unchanged."""
    return tuple(
        weight.dequantize(dtype=dtype) if isinstance(weight, _containers()) else weight
        for weight in (w13_experts, w2_experts)
    )
