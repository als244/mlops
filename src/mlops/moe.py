"""Model-facing routed-MoE semantic façade and shared-expert composition."""

from __future__ import annotations

import torch
from torch.nn import functional

from .dispatch import resolve_implementation
from .swiglu import packed_swiglu


def _production_shared_base(
    h2,
    residual,
    shared_w13,
    shared_w2,
    shared_gate_weight,
    shared_mode,
):
    """Compose the optional shared expert before routed expert dispatch.

    The shared weights are nn.Linear weights: ``shared_w13 [2Fs, D]`` with gate
    rows then up rows, ``shared_w2 [D, Fs]`` and ``shared_gate_weight [1, D]``.
    """
    if shared_w13 is None:
        return residual
    flat = h2.reshape(-1, h2.shape[-1])
    shared = functional.linear(
        packed_swiglu(functional.linear(flat, shared_w13)), shared_w2
    )
    if shared_mode == "sigmoid":
        gate = torch.sigmoid(functional.linear(flat, shared_gate_weight).float())
        shared = (gate * shared.float()).to(residual.dtype)
    return residual + shared.reshape_as(residual)


def moe(
    h2,
    residual,
    router_weight,
    w13_experts,
    w2_experts,
    *,
    top_k,
    routing_mode,
    router_bias=None,
    n_group=1,
    topk_group=1,
    routed_scaling=1.0,
    lengths=None,
    shared_w13=None,
    shared_w2=None,
    shared_gate_weight=None,
    shared_mode="add",
    compute_precision=None,
):
    """Return routed output, auxiliary loss, counts, and probability sum.

    ``compute_precision`` names the precision of the experts' GEMMs, apart
    from how the weights are stored: ``"bf16"`` (FP8 weights are dequantized
    per call), ``"fp8_current"`` (E4M3 with current per-row scaling of the
    activations, served by the SonicMoE implementations on SM90; bf16 weights
    are quantized per call) or ``None``, the weights' own -- FP8 for
    ``QuackFP8Weight`` containers (``expert_parallel/quack/parameters/fp8.py``,
    the E4M3 rows and transposes with their descales, built once), bf16
    otherwise. Dispatch declines implementations that cannot honor it.
    """
    flat_rows = h2.numel() // h2.shape[-1]
    if compute_precision not in (None, "bf16", "fp8_current"):
        raise ValueError(f"unsupported compute precision {compute_precision!r}")
    lengths = (flat_rows,) if lengths is None else tuple(int(x) for x in lengths)
    if (shared_w13 is None) != (shared_w2 is None):
        raise ValueError("shared_w13 and shared_w2 must be supplied together")
    if shared_mode not in {"add", "sigmoid"}:
        raise ValueError(f"unsupported shared expert mode {shared_mode!r}")
    if shared_mode == "sigmoid" and shared_gate_weight is None:
        raise ValueError("sigmoid shared expert requires shared_gate_weight")
    implementation = resolve_implementation(
        "moe",
        h2,
        residual,
        router_weight,
        w13_experts,
        w2_experts,
        surface="semantic",
        top_k=int(top_k),
        routing_mode=routing_mode,
        router_bias=router_bias,
        n_group=int(n_group),
        topk_group=int(topk_group),
        routed_scaling=float(routed_scaling),
        lengths=lengths,
        shared_w13=shared_w13,
        shared_w2=shared_w2,
        shared_gate_weight=shared_gate_weight,
        shared_mode=shared_mode,
        compute_precision=compute_precision,
    )
    return implementation.apply(
        h2,
        residual,
        router_weight,
        w13_experts,
        w2_experts,
        top_k=int(top_k),
        routing_mode=routing_mode,
        router_bias=router_bias,
        n_group=int(n_group),
        topk_group=int(topk_group),
        routed_scaling=float(routed_scaling),
        lengths=lengths,
        shared_w13=shared_w13,
        shared_w2=shared_w2,
        shared_gate_weight=shared_gate_weight,
        shared_mode=shared_mode,
        compute_precision=compute_precision,
    )


__all__ = ["moe"]
