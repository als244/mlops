"""ScatterMoE 0.3.0 routed-expert adapter and functional custom-op pair."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import torch

from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.moe_router import current_route_weight_precision
from ..moe_common import _backward_with_engine, _forward_with_engine


try:
    _VERSION = version("scattermoe")
    from . import engine as _engine  # noqa: F401

    _IMPORT_ERROR = None
except (ImportError, PackageNotFoundError) as error:
    _VERSION = None
    _IMPORT_ERROR = error


def supports(
    h2,
    residual,
    router_weight,
    w13_experts,
    w2_experts,
    *,
    surface,
    top_k,
    routing_mode,
    router_bias=None,
    n_group=1,
    topk_group=1,
    routed_scaling=1.0,
    lengths=None,
    **_kwargs,
):
    del surface, routing_mode, router_bias, n_group, topk_group, routed_scaling
    if _IMPORT_ERROR is not None:
        return SupportResult.no(f"ScatterMoE is unavailable: {_IMPORT_ERROR}")
    if _VERSION != "0.3.0":
        return SupportResult.no(f"requires ScatterMoE 0.3.0, found {_VERSION}")
    tensors = (h2, residual, router_weight, w13_experts, w2_experts)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return SupportResult.no("MoE core arguments must be tensors")
    if not all(value.is_cuda for value in tensors):
        return SupportResult.no("ScatterMoE requires CUDA tensors")
    if h2.dtype not in {torch.float16, torch.bfloat16}:
        return SupportResult.no("ScatterMoE activations must be float16 or bfloat16")
    if w13_experts.dtype != h2.dtype or w2_experts.dtype != h2.dtype:
        return SupportResult.no("expert weights must match the activation dtype")
    if not all(value.is_contiguous() for value in (h2, residual, w13_experts, w2_experts)):
        return SupportResult.no("hidden, residual, and expert weights must be contiguous")
    if h2.shape != residual.shape:
        return SupportResult.no("h2 and residual must have equal shape")
    experts = router_weight.shape[1]
    if w13_experts.shape[:2] != (experts, h2.shape[-1]):
        return SupportResult.no("w13 must use [experts, model_width, 2 * expert_width]")
    if w2_experts.shape[0] != experts or w2_experts.shape[2] != h2.shape[-1]:
        return SupportResult.no("w2 must use [experts, expert_width, model_width]")
    if w13_experts.shape[2] != 2 * w2_experts.shape[1]:
        return SupportResult.no("w13 packed width must be twice w2 input width")
    if not 0 < int(top_k) <= experts:
        return SupportResult.no("top_k must be within the expert count")
    rows = h2.numel() // h2.shape[-1]
    normalized = (rows,) if lengths is None else tuple(int(value) for value in lengths)
    if sum(normalized) != rows:
        return SupportResult.no("lengths must cover all hidden rows")
    return SupportResult.yes()


def forward(
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
    weight_precision="float32",
    swiglu_implementation="builtin.packed_swiglu.triton",
):
    swiglu_variant = (
        "triton_legacy"
        if swiglu_implementation.endswith("triton_legacy")
        else "triton"
    )
    return _forward_with_engine(
        h2,
        residual,
        router_weight,
        w13_experts,
        w2_experts,
        top_k=top_k,
        routing_mode=routing_mode,
        router_bias=router_bias,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling=routed_scaling,
        lengths=lengths,
        weight_precision=weight_precision,
        swiglu_variant=swiglu_variant,
        expert_engine="scattermoe",
    )


def backward(*args, swiglu_implementation="builtin.packed_swiglu.triton", **kwargs):
    swiglu_variant = (
        "triton_legacy"
        if swiglu_implementation.endswith("triton_legacy")
        else "triton"
    )
    return _backward_with_engine(
        *args,
        swiglu_variant=swiglu_variant,
        expert_engine="scattermoe",
        **kwargs,
    )


@torch.library.custom_op(
    "mlops::moe_scattermoe_fwd", mutates_args=()
)
def _forward_op(
    h2: torch.Tensor,
    residual: torch.Tensor,
    router_weight: torch.Tensor,
    w13_experts: torch.Tensor,
    w2_experts: torch.Tensor,
    router_bias: torch.Tensor,
    top_k: int,
    routing_mode: str,
    n_group: int,
    topk_group: int,
    routed_scaling: float,
    lengths: list[int],
    weight_precision: str,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor,
]:
    return forward(
        h2,
        residual,
        router_weight,
        w13_experts,
        w2_experts,
        top_k=top_k,
        routing_mode=routing_mode,
        router_bias=None if router_bias.numel() == 0 else router_bias,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling=routed_scaling,
        lengths=lengths,
        weight_precision=weight_precision,
    )


@_forward_op.register_fake
def _forward_fake(
    h2, residual, router_weight, w13_experts, w2_experts, router_bias,
    top_k, routing_mode, n_group, topk_group, routed_scaling, lengths,
    weight_precision,
):
    del residual, w2_experts, router_bias, routing_mode, n_group, topk_group, routed_scaling, lengths
    rows = h2.numel() // h2.shape[-1]
    experts = router_weight.shape[1]
    assignments = rows * int(top_k)
    return (
        torch.empty_like(h2),
        torch.empty((), dtype=torch.float32, device=h2.device),
        torch.empty(experts, dtype=torch.int64, device=h2.device),
        torch.empty(experts, dtype=torch.float32, device=h2.device),
        torch.empty((rows, experts), dtype=router_weight.dtype, device=h2.device),
        torch.empty(
            (rows, int(top_k)),
            dtype=torch.float32 if weight_precision == "float32" else router_weight.dtype,
            device=h2.device,
        ),
        torch.empty((rows, int(top_k)), dtype=torch.int32, device=h2.device),
        torch.empty(assignments, dtype=torch.int32, device=h2.device),
        torch.empty(experts + 1, dtype=torch.int32, device=h2.device),
        torch.empty((rows, int(top_k)), dtype=torch.int32, device=h2.device),
        torch.empty((assignments, w13_experts.shape[2]), dtype=h2.dtype, device=h2.device),
    )


@torch.library.custom_op(
    "mlops::moe_scattermoe_bwd", mutates_args=()
)
def _backward_op(
    grad_output: torch.Tensor,
    grad_aux: torch.Tensor,
    grad_probability_sum: torch.Tensor,
    h2: torch.Tensor,
    router_weight: torch.Tensor,
    w13_experts: torch.Tensor,
    w2_experts: torch.Tensor,
    logits: torch.Tensor,
    route_weights: torch.Tensor,
    route_ids: torch.Tensor,
    order: torch.Tensor,
    offsets: torch.Tensor,
    slots: torch.Tensor,
    h13: torch.Tensor,
    top_k: int,
    routing_mode: str,
    lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return backward(
        grad_output,
        grad_aux,
        grad_probability_sum,
        h2,
        router_weight,
        w13_experts,
        w2_experts,
        logits,
        route_weights,
        route_ids,
        order,
        offsets,
        slots,
        h13,
        top_k=top_k,
        routing_mode=routing_mode,
        lengths=lengths,
        swiglu_implementation="builtin.packed_swiglu.triton",
    )


@_backward_op.register_fake
def _backward_fake(
    grad_output, grad_aux, grad_probability_sum, h2, router_weight,
    w13_experts, w2_experts, logits, route_weights, route_ids, order,
    offsets, slots, h13, top_k, routing_mode, lengths,
):
    del grad_aux, grad_probability_sum, logits, route_weights, route_ids, order, offsets, slots, h13, top_k, routing_mode, lengths
    return (
        torch.empty_like(h2),
        torch.empty_like(grad_output),
        torch.empty_like(router_weight),
        torch.empty_like(w13_experts),
        torch.empty_like(w2_experts),
    )


def _setup_context(ctx, inputs, output):
    (
        h2, _residual, router_weight, w13_experts, w2_experts, _router_bias,
        top_k, routing_mode, _n_group, _topk_group, _routed_scaling, lengths,
        _weight_precision,
    ) = inputs
    (
        _result, _aux, counts, _probability_sum, logits, route_weights,
        route_ids, order, offsets, slots, h13,
    ) = output
    ctx.save_for_backward(
        h2, router_weight, w13_experts, w2_experts, logits, route_weights,
        route_ids, order, offsets, slots, h13,
    )
    ctx.top_k = int(top_k)
    ctx.routing_mode = routing_mode
    ctx.lengths = list(lengths)
    ctx.mark_non_differentiable(
        counts, logits, route_weights, route_ids, order, offsets, slots, h13
    )


def _autograd_backward(
    ctx, grad_output, grad_aux, _grad_counts, grad_probability_sum,
    _grad_logits, _grad_route_weights, _grad_route_ids, _grad_order,
    _grad_offsets, _grad_slots, _grad_h13,
):
    saved = ctx.saved_tensors
    h2, router_weight = saved[:2]
    if grad_aux is None:
        grad_aux = torch.zeros((), dtype=torch.float32, device=h2.device)
    if grad_probability_sum is None:
        grad_probability_sum = torch.zeros(
            router_weight.shape[1], dtype=torch.float32, device=h2.device
        )
    gradients = _backward_op(
        grad_output,
        grad_aux,
        grad_probability_sum,
        *saved,
        ctx.top_k,
        ctx.routing_mode,
        ctx.lengths,
    )
    return (*gradients, None, None, None, None, None, None, None, None)


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def apply(
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
):
    from ...moe import _production_shared_base

    rows = h2.numel() // h2.shape[-1]
    normalized_lengths = (rows,) if lengths is None else tuple(int(x) for x in lengths)
    bias = torch.empty(0, dtype=h2.dtype, device=h2.device) if router_bias is None else router_bias
    base = _production_shared_base(
        h2, residual, shared_w13, shared_w2, shared_gate_weight, shared_mode
    )
    outputs = _forward_op(
        h2,
        base,
        router_weight,
        w13_experts,
        w2_experts,
        bias,
        int(top_k),
        routing_mode,
        int(n_group),
        int(topk_group),
        float(routed_scaling),
        list(normalized_lengths),
        current_route_weight_precision(),
    )
    return outputs[:4]


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="moe",
        implementation_id="scattermoe.moe",
        provider="scattermoe",
        priority=50,
        deterministic=True,
        supports=supports,
        apply=apply,
        forward=forward,
        backward=backward,
    )
)

__all__ = ["IMPLEMENTATION", "apply", "backward", "forward", "supports"]
