"""One-pass chunked output heads for installed, summed token objectives."""

from __future__ import annotations

from .dispatch import resolve_implementation


def head_objective(
    hidden,
    head_weight,
    targets,
    *,
    objective,
    inputs=(),
    controls=(),
    lora_a=None,
    lora_b=None,
    scale=1.0,
    chunk_size=None,
):
    """Return ``(loss, detached_target_logprobs)`` with bounded logits.

    ``objective="cross_entropy"`` uses row inputs ``(mask, weights)`` and no
    controls. ``objective="clipped_policy"`` uses ``(mask, weights, old_logprobs,
    advantages)`` and the scalar control ``clip_ratio``; optional reference
    logprobs and a ``reference_beta`` control add the sampled k3 penalty.
    These common objectives use fused CUDA kernels without a Python formula
    call inside the chunk loop. All inputs and controls remain runtime tensors.

    Advanced custom objectives can instead supply an installed module-level
    function taking ``(logprobs, row_inputs, controls)`` and returning
    ``(scalar_sum, dloss_dlogprobs)``. The function
    must be separable across rows: splitting rows must split the same sum.
    Each row input has the target shape; controls are scalar tensors. Neither
    is differentiated. The supplied derivative is consumed inside each chunk
    to accumulate hidden/weight gradients before releasing its logits.
    Backward only scales saved gradients. The ordinary PyTorch provider ignores
    the supplied derivative and differentiates the loss, providing an oracle.

    For a custom function, the owner of a compiled artifact must pin its source
    and declare its allocation behavior, just as for its other custom operators.
    Cross-row statistics belong outside this operation.
    """
    implementation = resolve_implementation(
        "head_objective",
        hidden,
        head_weight,
        targets,
        surface="semantic",
        objective=objective,
        inputs=inputs,
        controls=controls,
        lora_a=lora_a,
        lora_b=lora_b,
        scale=scale,
        chunk_size=chunk_size,
    )
    return implementation.apply(
        hidden,
        head_weight,
        targets,
        objective=objective,
        inputs=inputs,
        controls=controls,
        lora_a=lora_a,
        lora_b=lora_b,
        scale=scale,
        chunk_size=chunk_size,
    )


__all__ = ["head_objective"]
