"""Joint gate/up and down LoRA with independent factors for every expert."""

import torch
from torch import nn

from ..providers.expert_weights import bf16_compute_support, dequantize_expert_weights
from ..swiglu import swiglu
from .grouped import grouped_linear
from .routing import (
    combine_rows,
    dispatch_rows,
    layout,
    routing,
    routing_logits,
    statistics,
)


def initialize_factors(module, gate_up, down, config):
    """Register only small factors, preserving the existing base parameters.

    The base weights are ``[E, out, in]``; each factor pair follows nn.Linear
    LoRA: ``A [E, rank, in]`` and ``B [E, out, rank]``.
    """
    options = {"device": gate_up.device, "dtype": config.factor_dtype}
    shapes = {
        "gate_up_a": (gate_up.shape[0], config.rank, gate_up.shape[2]),
        "gate_up_b": (gate_up.shape[0], gate_up.shape[1], config.rank),
        "down_a": (down.shape[0], config.rank, down.shape[2]),
        "down_b": (down.shape[0], down.shape[1], config.rank),
    }
    for name, shape in shapes.items():
        module.register_parameter(
            "lora_" + name, nn.Parameter(torch.empty(shape, **options))
        )
    reset_factors(module)


def reset_factors(module):
    for prefix in ("gate_up", "down"):
        a, b = getattr(module, f"lora_{prefix}_a"), getattr(module, f"lora_{prefix}_b")
        nn.init.normal_(a, std=a.shape[2] ** -0.5)
        nn.init.zeros_(b)


def _project(x, weight, a, b, offsets, scale):
    low = grouped_linear(x, a.to(x.dtype), offsets)
    return grouped_linear(x, weight, offsets) + scale * grouped_linear(
        low, b.to(x.dtype), offsets
    )


def expert_lora(
    hidden,
    residual,
    router_weight,
    gate_up,
    down,
    a13,
    b13,
    a2,
    b2,
    *,
    top_k,
    routing_mode,
    scale,
    router_bias=None,
    n_group=1,
    topk_group=1,
    routed_scaling=1.0,
    lengths=None,
    swiglu_limit=None,
    implementation="auto",
    compute_precision=None,
):
    """Fused kernels with compiler-visible low-rank projections and lifetimes.

    Weight shapes are [E,2H,D] and [E,D,H] with factors [E,R,D], [E,2H,R],
    [E,R,H], [E,D,R]; router_weight is [E,D]. No dense B@A update or frozen
    base weight gradient is constructed.

    Routing and its statistics are `mlops.moe`'s: ``routing_mode`` a softmax
    mode, or ``"sigmoid_noaux_tc"`` with the correction ``router_bias``
    (selection only), ``n_group``/``topk_group`` and ``routed_scaling``, its
    balance statistic taken per sequence of ``lengths``. ``swiglu_limit``
    clamps every expert's SwiGLU as `mlops.moe`'s does.

    ``implementation`` is ``"auto"`` (SonicMoE's kernels where they apply --
    compute capability 9.x/10.x, bfloat16, the ``quack`` and ``sonicmoe``
    extras -- else the builtin grouped GEMMs), ``"builtin"``, or
    ``"sonicmoe"``, which fails with the reason where it does not apply.
    ``compute_precision`` is `mlops.moe`'s: ``None`` for the base weights'
    own precision (FP8 for ``QuackFP8Weight`` or ``BlockFP8Weight``
    containers, which a frozen base loaded from an FP8 checkpoint is),
    ``"bf16"``, ``"fp8_current"`` or ``"fp8_block"``; the builtin composition
    dequantizes containers and computes in bf16.
    """
    if implementation not in ("auto", "builtin", "sonicmoe"):
        raise ValueError(f"unknown expert LoRA implementation {implementation!r}")
    if swiglu_limit is not None:
        swiglu_limit = float(swiglu_limit)
        if not swiglu_limit > 0:
            raise ValueError("swiglu_limit must be positive")
    options = {
        "router_bias": router_bias,
        "n_group": n_group,
        "topk_group": topk_group,
        "routed_scaling": routed_scaling,
        "lengths": lengths,
        "swiglu_limit": swiglu_limit,
    }
    if implementation != "builtin":
        from ..providers.sonicmoe import lora as sonicmoe_lora

        support = sonicmoe_lora.supports(
            hidden, gate_up, down, compute_precision=compute_precision
        )
        if implementation == "sonicmoe" and not support:
            raise ValueError(
                f"the SonicMoE expert LoRA path does not apply here: {support.reason}"
            )
        if support:
            return sonicmoe_lora.expert_lora(
                hidden,
                residual,
                router_weight,
                gate_up,
                down,
                a13,
                b13,
                a2,
                b2,
                top_k=top_k,
                routing_mode=routing_mode,
                scale=scale,
                compute_precision=compute_precision,
                **options,
            )

    refusal = bf16_compute_support(compute_precision, gate_up, down)
    if refusal is not None:
        raise ValueError(f"the builtin expert LoRA composition: {refusal}")
    gate_up, down = dequantize_expert_weights(gate_up, down, dtype=hidden.dtype)
    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = routing_logits(flat, router_weight)
    weights, ids = routing(
        logits,
        top_k,
        routing_mode,
        bias=router_bias,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling=routed_scaling,
    )
    order, offsets, slots = layout(ids, gate_up.shape[0])
    auxiliary, counts, probability_sum = statistics(
        logits, ids, offsets, routing_mode=routing_mode, lengths=lengths
    )
    x = dispatch_rows(flat, order, slots)
    h13 = _project(x, gate_up, a13, b13, offsets, scale)
    gate, up = h13.chunk(2, -1)
    activated = swiglu(gate, up, limit=swiglu_limit)
    y = _project(activated, down, a2, b2, offsets, scale)
    output = combine_rows(
        y, order, slots, weights, residual.reshape_as(flat).contiguous()
    )
    return output.reshape_as(hidden), auxiliary, counts, probability_sum
