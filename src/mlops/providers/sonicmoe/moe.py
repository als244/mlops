"""Routed MoE through SonicMoE's fused kernels, as one opaque operator pair.

Routing is the package's own (``kernels.moe_router``), so every routing mode,
bias and group rule, the auxiliary loss and the probability sums are exactly
the builtin's. SonicMoE provides the routing metadata, the gathered up
projection with SwiGLU, the down projection with the score-weighted combine,
and the fused backward kernels, all on Quack's CuTe DSL GEMMs. The package's
nn.Linear weight layout -- ``w13 [E, 2I, H]`` with gate rows then up rows,
``w2 [E, H, I]`` -- is SonicMoE's native storage, so no weight is copied.

The experts' GEMMs run in FP8 on request (``compute_precision="fp8_current"``
or ``"fp8_block"``) or by default for weights that arrive as FP8 containers:
two-copy ``QuackFP8Weight``s (``fp8.py``) or one-copy block-scaled
``BlockFP8Weight``s (``block_fp8.py``). The operators take containers as the
expert-parallel operators do: a dispatch rule turns each into its E4M3 data
and the trailing ``weight_components`` list; with ``"bf16"`` asked of
container weights, they are dequantized per call instead.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec

import torch
from torch.nn import functional

from ...dispatch import logical_costs as logical
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.moe_router import current_route_weight_precision, route
from ...kernels.router import router_logits
from ..builtin.moe import supports as builtin_supports
from ..moe_common import (
    add_router_input_gradient_,
    requested,
    router_weight_gradient,
    routing_gradient,
    routing_statistics,
    skipped_gradient,
)
from . import bf16, fp8

# A bf16 row of this many elements is 16 bytes, the unit Quack's loads move.
ROW_ALIGNMENT = 8
CAPABILITIES = (9, 10)


_AVAILABLE: bool | None = None


@torch.compiler.assume_constant_result
def available() -> bool:
    """Whether the ``quack`` and ``sonicmoe`` extras are installed.

    Found without importing them: importing ``sonicmoe`` replaces Quack's
    tuning policy, which the shared runtime must see untouched first. The
    answer is a constant of the process, held in a module global rather
    than an ``lru_cache`` so Dynamo, which meets this call in ``supports``
    while tracing a model, takes the value instead of tracing the probe.
    """
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = all(find_spec(name) is not None for name in ("quack", "sonicmoe"))
    return _AVAILABLE


@lru_cache(maxsize=1)
def kernels():
    """SonicMoE's functional package, imported through the shared Quack runtime."""
    from ...quack_runtime import initialize
    from ...quack_runtime.autotune import sonicmoe_functional

    initialize()
    return sonicmoe_functional()


def supports(
    h2, residual, router_weight, w13_experts, w2_experts, *, surface, **kwargs
):
    if surface == "explicit":
        return SupportResult.no("the SonicMoE implementations are semantic-only")
    compute_precision = kwargs.pop("compute_precision", None)
    # The builtin's shape rules, which accept FP8 containers for bf16 math;
    # the precision itself is resolved below.
    base = builtin_supports(
        h2,
        residual,
        router_weight,
        w13_experts,
        w2_experts,
        surface=surface,
        compute_precision="bf16",
        **kwargs,
    )
    if not base:
        return base
    if not h2.is_cuda:
        return SupportResult.no("SonicMoE runs on CUDA devices")
    if torch.cuda.get_device_capability(h2.device)[0] not in CAPABILITIES:
        return SupportResult.no("SonicMoE serves compute capability 9.x and 10.x")
    if not available():
        return SupportResult.no("the quack and sonicmoe extras are not installed")
    plain = [w for w in (w13_experts, w2_experts) if not fp8.is_container(w)]
    if any(
        bf16.compute_dtype(tensor) is not torch.bfloat16
        for tensor in (h2, residual, *plain)
    ):
        return SupportResult.no("SonicMoE computes in bfloat16")
    if h2.shape[-1] % ROW_ALIGNMENT or w2_experts.shape[2] % ROW_ALIGNMENT:
        return SupportResult.no("the model and expert widths must be multiples of 8")
    if not (w13_experts.is_contiguous() and w2_experts.is_contiguous()):
        return SupportResult.no("expert weights must be contiguous")
    if compute_precision not in (None, "bf16", *fp8.PRECISIONS):
        return SupportResult.no(f"unknown compute precision {compute_precision!r}")
    precision = fp8.resolve(compute_precision, w13_experts, w2_experts)
    if precision in fp8.PRECISIONS:
        return fp8.supports(h2, w13_experts, w2_experts, precision)
    return SupportResult.yes()


def metadata(route_ids, experts):
    """SonicMoE's routing metadata ``(offsets, gather, scatter, reverse)``.

    ``offsets [E + 1]`` bounds each expert's rows in expert-sorted order;
    ``gather [rows * top_k]`` names the token of each sorted row; ``scatter``
    and ``reverse`` map (token, slot) pairs to and from sorted rows.
    """
    functional_kernels = kernels()
    assignments = route_ids.numel()
    device = route_ids.device
    frequency = torch.empty(experts, dtype=torch.int32, device=device)
    offsets = torch.empty(experts + 1, dtype=torch.int32, device=device)
    gather = torch.empty(assignments, dtype=torch.int32, device=device)
    scatter = torch.empty_like(gather)
    reverse = torch.empty_like(gather)
    functional_kernels.triton_kernels.TC_topk_router_metadata_triton(
        route_ids, experts, frequency, offsets, gather, scatter, reverse
    )
    return offsets, gather, scatter, reverse


@torch.library.custom_op("mlops::sonicmoe_routing_metadata", mutates_args=())
def routing_metadata(
    route_ids: torch.Tensor, experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`metadata` as a functional operator, for compositions that route in
    the traced graph (expert LoRA): SonicMoE's kernel writes into the
    buffers it is handed, which a traced graph would carry as a mutating
    operator."""
    return metadata(route_ids, int(experts))


@routing_metadata.register_fake
def _routing_metadata_fake(route_ids, experts):
    count = route_ids.numel()
    return (
        route_ids.new_empty((experts + 1,), dtype=torch.int32),
        *(route_ids.new_empty((count,), dtype=torch.int32) for _ in range(3)),
    )


@flop_formula(routing_metadata)
def _routing_metadata_flops(route_ids, experts, out_val=None):
    del route_ids, experts, out_val
    return 0


def up_projection(
    h2_flat, w13_experts, offsets, gather, compute_precision="native", limit=None
):
    """Gather the routed rows, project through ``w13`` and apply SwiGLU in one
    kernel: returns ``(h, a)``, the saved pre-activation -- whose gate/up
    columns Quack keeps interleaved -- and the activation, clamped at
    ``limit`` when given."""
    from quack.gemm_interface import gemm_act

    if compute_precision in fp8.PRECISIONS:
        return fp8.up_projection(h2_flat, w13_experts, offsets, gather, limit=limit)

    assignments, width = gather.shape[0], w13_experts.shape[1] // 2
    h = torch.empty(assignments, 2 * width, dtype=h2_flat.dtype, device=h2_flat.device)
    a = torch.empty(assignments, width, dtype=h2_flat.dtype, device=h2_flat.device)
    if limit is not None:
        from mlops.kernels.quack_moe.swiglu_limit import swiglu_limit

        swiglu_limit(
            h2_flat,
            w13_experts.transpose(1, 2),
            None,
            out={"D": h, "postact": a},
            cu_seqlens_m=offsets,
            A_idx=gather,
            concat_layout=("B",),
            limit=float(limit),
        )
        return h, a
    gemm_act(
        h2_flat,
        w13_experts.transpose(1, 2),
        activation="swiglu",
        cu_seqlens_m=offsets,
        A_idx=gather,
        preact_out=h,
        postact_out=a,
        store_preact=True,
        concat_layout=("B",),
    )
    return h, a


def activation(h, limit=None):
    """SwiGLU of the interleaved pre-activation, as the down projection reads
    it, clamped at ``limit`` when given."""
    if h.is_cuda:
        from mlops.kernels.quack_moe.swiglu_rows import swiglu_rows

        return swiglu_rows(h, limit=limit)
    pairs = h.view(h.shape[0], h.shape[1] // 2, 2)
    gate, up = pairs[..., 0].float(), pairs[..., 1].float()
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return (functional.silu(gate) * up).to(h.dtype)


def down_projection(
    a,
    w2_experts,
    route_weights,
    offsets,
    reverse,
    residual_flat,
    compute_precision="native",
    from_preactivation=False,
    limit=None,
):
    """Project through ``w2``, weight each row by its route weight, sum the
    rows of every token and add the base. ``a`` is the activation, or with
    ``from_preactivation`` the interleaved ``h`` the FP8 path activates
    (clamped at ``limit``) and quantizes in one pass."""
    from quack.gemm_interface import gemm

    functional_kernels = kernels()
    width = residual_flat.shape[1]
    if compute_precision in fp8.PRECISIONS:
        y = (
            fp8.project_down_from_preactivation(a, w2_experts, offsets, limit=limit)
            if from_preactivation
            else fp8.project_down(a, w2_experts, offsets)
        )
    else:
        y = torch.empty(a.shape[0], width, dtype=a.dtype, device=a.device)
        gemm(a, w2_experts.transpose(1, 2), out=y, cu_seqlens_m=offsets)
    combined = torch.empty_like(residual_flat)
    functional_kernels.forward._router_forward(
        y=y,
        o=combined,
        topk_scores=route_weights.reshape(-1).float(),
        s_reverse_scatter_idx=reverse,
        num_activated_expert_per_token_offset=None,
        varlen_K_max=route_weights.shape[1],
        H=width,
        is_varlen_K=False,
    )
    return combined + residual_flat


def down_backward(
    grad_output_flat,
    h,
    w2_experts,
    route_weights,
    offsets,
    gather,
    scatter,
    grad_dtype,
    compute_precision="native",
    need_weight=True,
    limit=None,
):
    """Return ``(dh, grad_route_weights, dw2)`` from the output cotangent
    (``dw2`` ``None`` unless ``need_weight``), the SwiGLU clamped at
    ``limit`` when given.

    One SonicMoE kernel takes the cotangent back through ``w2`` and SwiGLU to
    the pre-activation gradient, the route-weight gradient and the
    route-scaled activation ``a'``; a varlen GEMM over the gathered tokens then
    gives ``dw2 = grad_output.T @ a'`` per expert, at ``grad_dtype``.
    """
    from quack.gemm_interface import gemm

    functional_kernels = kernels()
    rows, top_k = route_weights.shape
    scores = route_weights.reshape(-1).float().contiguous()
    if compute_precision in fp8.PRECISIONS:
        # scatter names each sorted row's (token, slot) entry.
        slots = scatter.long()
        dh, a_prime, dscore = fp8.down_backward_activations(
            grad_output_flat, h, w2_experts, scores[slots], offsets, gather, limit=limit
        )
        ds = torch.empty_like(scores)
        ds[slots] = dscore.to(ds.dtype)
    elif limit is not None:
        # SonicMoE's gated backward with the clamped SwiGLU: the same GEMM
        # (the cotangent gathered through ``A_idx``, ``w2`` as stored), the
        # route score per sorted row, its gradient reduced in the epilogue.
        from mlops.kernels.quack_moe.swiglu_limit import (
            dswiglu_limit_scored,
        )

        slots = scatter.long()
        dh = torch.empty_like(h)
        a_prime = torch.empty(
            h.shape[0], w2_experts.shape[2], dtype=h.dtype, device=h.device
        )
        result = dswiglu_limit_scored(
            grad_output_flat,
            w2_experts,
            h,
            out={"D": dh, "postact": a_prime},
            score=scores[slots].contiguous(),
            limit=float(limit),
            cu_seqlens_m=offsets,
            A_idx=gather,
            dynamic_scheduler=False,
        )
        ds = torch.empty_like(scores)
        ds[slots] = result["dscore"].to(ds.dtype)
    else:
        dh = torch.empty_like(h)
        ds = torch.empty_like(scores)
        a_prime = torch.empty(
            h.shape[0], w2_experts.shape[2], dtype=h.dtype, device=h.device
        )
        functional_kernels.backward._down_projection_backward_act(
            dout=grad_output_flat,
            h=h,
            w2=w2_experts.permute(1, 2, 0),
            dh=dh,
            ds=ds,
            b2=None,
            db2=None,
            a_prime=a_prime,
            topk_scores=scores,
            expert_frequency_offset=offsets,
            x_gather_idx=gather,
            s_scatter_idx=scatter,
            activation_type="swiglu",
        )
    dw2 = None
    if need_weight:
        dw2 = torch.empty(
            w2_experts.shape,
            dtype=w2_experts.dtype if grad_dtype is None else grad_dtype,
            device=h.device,
        )
        gemm(
            grad_output_flat.T,
            a_prime,
            out=dw2,
            cu_seqlens_k=offsets,
            A_idx=gather,
            batch_idx_permute=None,
            dynamic_scheduler=True,
        )
    return dh, ds.view(rows, top_k).to(route_weights.dtype), dw2


def up_backward(
    h2_flat,
    w13_experts,
    dh,
    offsets,
    gather,
    reverse,
    top_k,
    grad_dtype,
    compute_precision="native",
    need_weight=True,
):
    """Return ``(grad_h2_flat, dw13)`` from the pre-activation gradient: the
    rows' gradient through ``w13``, ``dw13 = dh.T @ x`` per expert at
    ``grad_dtype`` (``None`` unless ``need_weight``), and the rows summed back
    onto their tokens."""
    from quack.gemm_interface import gemm

    functional_kernels = kernels()
    width = h2_flat.shape[1]
    dh = dh.contiguous()  # an autograd cotangent need not be; the GEMM needs it so
    if compute_precision in fp8.PRECISIONS:
        dx_expanded = fp8.input_gradient_rows(dh, w13_experts, offsets, gate_up=True)
    else:
        dx_expanded = torch.empty(dh.shape[0], width, dtype=dh.dtype, device=dh.device)
        functional_kernels.backward._up_projection_backward_act(
            w1=w13_experts.permute(1, 2, 0),
            dx_expanded=dx_expanded,
            dh=dh,
            db1=None,
            expert_frequency_offset=offsets,
            is_glu_activation=True,
            concat_layout=True,
        )
    dw13 = None
    if need_weight:
        dw13 = torch.empty(
            w13_experts.shape,
            dtype=w13_experts.dtype if grad_dtype is None else grad_dtype,
            device=dh.device,
        )
        gemm(
            h2_flat.T,
            dh,
            out=dw13.transpose(1, 2),
            cu_seqlens_k=offsets,
            A_idx=gather,
            batch_idx_permute=None,
            dynamic_scheduler=True,
            concat_layout=("out",),
        )
    grad_h2 = torch.empty_like(h2_flat)
    functional_kernels.backward._token_broadcast_backward(
        dx_reduced=grad_h2,
        dx_expanded=dx_expanded,
        s_reverse_scatter_idx=reverse,
        num_activated_expert_per_token_offset=None,
        varlen_K_max=top_k,
        H=width,
        is_varlen_K=False,
    )
    return grad_h2, dw13


def normalized(lengths, rows):
    """The sequence lengths as ints, checked to cover every hidden row."""
    result = tuple(int(length) for length in lengths)
    if not result or any(length <= 0 for length in result) or sum(result) != rows:
        raise ValueError("MoE lengths must be positive and cover every hidden row")
    return result


def forward(
    h2,
    residual,
    router_weight,
    w13_experts,
    w2_experts,
    router_bias,
    *,
    top_k,
    routing_mode,
    n_group,
    topk_group,
    routed_scaling,
    lengths,
    weight_precision,
    compute_precision="native",
    swiglu_limit=None,
):
    """Return output, auxiliary values and the VJP residuals.

    The ordered result is ``(output, aux, counts, probability_sum, logits,
    route_weights, route_ids, offsets, gather, scatter, reverse, h)``.
    """
    rows = h2.numel() // h2.shape[-1]
    normalized_lengths = normalized(lengths, rows)
    experts = router_weight.shape[0]
    with torch.no_grad():
        h2_flat = h2.reshape(-1, h2.shape[-1]).contiguous()
        residual_flat = residual.reshape_as(h2_flat).contiguous()
        logits = router_logits(h2_flat, router_weight)
        route_weights, route_ids = route(
            logits,
            int(top_k),
            str(routing_mode),
            bias=None if router_bias.numel() == 0 else router_bias,
            n_group=int(n_group),
            topk_group=int(topk_group),
            routed_scaling=float(routed_scaling),
            weight_precision=str(weight_precision),
        )
        offsets, gather, scatter, reverse = metadata(route_ids, experts)
        h, a = up_projection(
            h2_flat, w13_experts, offsets, gather, compute_precision, swiglu_limit
        )
        output = down_projection(
            a,
            w2_experts,
            route_weights,
            offsets,
            reverse,
            residual_flat,
            compute_precision,
        )
        aux, counts, probability_sum = routing_statistics(
            logits,
            route_ids,
            routing_mode=str(routing_mode),
            lengths=normalized_lengths,
            experts=experts,
        )
        return (
            output.reshape(h2.shape),
            aux,
            counts,
            probability_sum,
            logits,
            route_weights,
            route_ids,
            offsets,
            gather,
            scatter,
            reverse,
            h,
        )


def backward(
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
    offsets,
    gather,
    scatter,
    reverse,
    h,
    *,
    top_k,
    routing_mode,
    lengths,
    weight_grad_dtype=None,
    compute_precision="native",
    need_router=True,
    need_w13=True,
    need_w2=True,
    swiglu_limit=None,
):
    """Return VJPs for ``h2, residual, router_weight, w13, w2`` in order, the
    weights' at ``weight_grad_dtype`` (their own when ``None``); a weight
    nobody asked a gradient for gets a placeholder and no GEMM."""
    with torch.no_grad():
        h2_flat = h2.reshape(-1, h2.shape[-1]).contiguous()
        grad_output_flat = grad_output.reshape_as(h2_flat).contiguous()
        dh, grad_route_weights, grad_w2 = down_backward(
            grad_output_flat,
            h,
            w2_experts,
            route_weights,
            offsets,
            gather,
            scatter,
            weight_grad_dtype,
            compute_precision,
            need_weight=need_w2,
            limit=swiglu_limit,
        )
        grad_h2, grad_w13 = up_backward(
            h2_flat,
            w13_experts,
            dh,
            offsets,
            gather,
            reverse,
            int(top_k),
            weight_grad_dtype,
            compute_precision,
            need_weight=need_w13,
        )
        grad_logits = routing_gradient(
            grad_aux,
            grad_probability_sum,
            grad_route_weights,
            route_weights,
            route_ids,
            logits,
            routing_mode=str(routing_mode),
            lengths=lengths,
        )
        grad_logits_storage = grad_logits.to(logits.dtype)
        grad_router = (
            router_weight_gradient(
                grad_logits_storage,
                h2_flat,
                router_weight,
                weight_grad_dtype,
            )
            if need_router
            else skipped_gradient(router_weight, weight_grad_dtype)
        )
        add_router_input_gradient_(grad_h2, grad_logits_storage, router_weight)
        # Under FP8 the weights are component tuples; the gradient dtype is
        # then always resolved, so the placeholders never need theirs.
        return (
            grad_h2.reshape(h2.shape),
            grad_output.clone(),
            grad_router,
            grad_w13
            if grad_w13 is not None
            else skipped_gradient(
                h2,
                weight_grad_dtype
                if weight_grad_dtype is not None
                else w13_experts.dtype,
            ),
            grad_w2
            if grad_w2 is not None
            else skipped_gradient(
                h2,
                weight_grad_dtype
                if weight_grad_dtype is not None
                else w2_experts.dtype,
            ),
        )


@torch.library.custom_op("mlops::moe_sonicmoe_fwd", mutates_args=())
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
    weight_grad_dtype: torch.dtype | None,
    compute_precision: str,
    swiglu_limit: float | None,
    weight_components: list[torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del weight_grad_dtype
    w13_experts, w2_experts = fp8.weights(
        compute_precision, weight_components, w13_experts, w2_experts
    )
    return forward(
        h2,
        residual,
        router_weight,
        w13_experts,
        w2_experts,
        router_bias,
        top_k=top_k,
        routing_mode=routing_mode,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling=routed_scaling,
        lengths=lengths,
        weight_precision=weight_precision,
        compute_precision=compute_precision,
        swiglu_limit=swiglu_limit,
    )


def residual_fakes(h2, router_weight, w13_experts, top_k, weight_precision):
    """Fake ``(logits, route_weights, route_ids, offsets, gather, scatter,
    reverse, h)`` for the forward operators' metadata."""
    rows = h2.numel() // h2.shape[-1]
    experts = router_weight.shape[0]
    assignments = rows * int(top_k)
    route_dtype = torch.float32 if weight_precision == "float32" else h2.dtype
    return (
        torch.empty((rows, experts), dtype=torch.float32, device=h2.device),
        torch.empty((rows, int(top_k)), dtype=route_dtype, device=h2.device),
        torch.empty((rows, int(top_k)), dtype=torch.int32, device=h2.device),
        torch.empty(experts + 1, dtype=torch.int32, device=h2.device),
        torch.empty(assignments, dtype=torch.int32, device=h2.device),
        torch.empty(assignments, dtype=torch.int32, device=h2.device),
        torch.empty(assignments, dtype=torch.int32, device=h2.device),
        torch.empty(
            (assignments, w13_experts.shape[1]), dtype=h2.dtype, device=h2.device
        ),
    )


def statistic_fakes(h2, router_weight):
    """Fake ``(aux, counts, probability_sum)``."""
    experts = router_weight.shape[0]
    return (
        torch.empty((), dtype=torch.float32, device=h2.device),
        torch.empty(experts, dtype=torch.int64, device=h2.device),
        torch.empty(experts, dtype=torch.float32, device=h2.device),
    )


@_forward_op.register_fake
def _forward_fake(
    h2,
    residual,
    router_weight,
    w13_experts,
    w2_experts,
    router_bias,
    top_k,
    routing_mode,
    n_group,
    topk_group,
    routed_scaling,
    lengths,
    weight_precision,
    weight_grad_dtype,
    compute_precision,
    swiglu_limit,
    weight_components,
):
    del residual, w2_experts, router_bias, routing_mode, n_group, topk_group
    del routed_scaling, lengths, weight_grad_dtype, compute_precision
    del swiglu_limit, weight_components
    return (
        torch.empty_like(h2),
        *statistic_fakes(h2, router_weight),
        *residual_fakes(h2, router_weight, w13_experts, top_k, weight_precision),
    )


@torch.library.custom_op("mlops::moe_sonicmoe_bwd", mutates_args=())
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
    offsets: torch.Tensor,
    gather: torch.Tensor,
    scatter: torch.Tensor,
    reverse: torch.Tensor,
    h: torch.Tensor,
    top_k: int,
    routing_mode: str,
    lengths: list[int],
    weight_grad_dtype: torch.dtype | None,
    compute_precision: str,
    swiglu_limit: float | None,
    weight_components: list[torch.Tensor],
    need_router: bool,
    need_w13: bool,
    need_w2: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w13_experts, w2_experts = fp8.weights(
        compute_precision, weight_components, w13_experts, w2_experts
    )
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
        offsets,
        gather,
        scatter,
        reverse,
        h,
        top_k=top_k,
        routing_mode=routing_mode,
        lengths=lengths,
        weight_grad_dtype=weight_grad_dtype,
        compute_precision=compute_precision,
        need_router=need_router,
        need_w13=need_w13,
        need_w2=need_w2,
        swiglu_limit=swiglu_limit,
    )


@_backward_op.register_fake
def _backward_fake(
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
    offsets,
    gather,
    scatter,
    reverse,
    h,
    top_k,
    routing_mode,
    lengths,
    weight_grad_dtype,
    compute_precision,
    swiglu_limit,
    weight_components,
    need_router,
    need_w13,
    need_w2,
):
    del grad_aux, grad_probability_sum, logits, route_weights, route_ids
    del offsets, gather, scatter, reverse, h, top_k, routing_mode, lengths
    del compute_precision, swiglu_limit, weight_components
    return (
        torch.empty_like(h2),
        torch.empty_like(grad_output),
        torch.empty_like(router_weight, dtype=weight_grad_dtype)
        if need_router
        else skipped_gradient(router_weight, weight_grad_dtype),
        torch.empty_like(w13_experts, dtype=weight_grad_dtype)
        if need_w13
        else skipped_gradient(w13_experts, weight_grad_dtype),
        torch.empty_like(w2_experts, dtype=weight_grad_dtype)
        if need_w2
        else skipped_gradient(w2_experts, weight_grad_dtype),
    )


@flop_formula(_forward_op)
def _forward_flops(
    h2,
    residual,
    router_weight,
    w13_experts,
    w2_experts,
    router_bias,
    top_k,
    *_rest,
    out_val=None,
    **_kwargs,
):
    del router_bias, out_val
    return logical.moe(
        h2,
        residual,
        router_weight,
        w13_experts,
        w2_experts,
        top_k=top_k,
        entrypoint="forward",
    ).logical_flops


@flop_formula(_backward_op)
def _backward_flops(
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
    offsets,
    gather,
    scatter,
    reverse,
    h,
    top_k,
    *_rest,
    out_val=None,
    **_kwargs,
):
    del grad_aux, grad_probability_sum, logits, route_weights, route_ids
    del offsets, gather, scatter, reverse, h, out_val
    return logical.moe(
        h2,
        grad_output,
        router_weight,
        w13_experts,
        w2_experts,
        top_k=top_k,
        entrypoint="backward",
    ).logical_flops


def _setup_context(ctx, inputs, output):
    (
        h2,
        _residual,
        router_weight,
        w13_experts,
        w2_experts,
        _router_bias,
        top_k,
        routing_mode,
        _n_group,
        _topk_group,
        _routed_scaling,
        lengths,
        _weight_precision,
        weight_grad_dtype,
        compute_precision,
        swiglu_limit,
        weight_components,
    ) = inputs
    (
        _result,
        _aux,
        counts,
        _probability_sum,
        logits,
        route_weights,
        route_ids,
        offsets,
        gather,
        scatter,
        reverse,
        h,
    ) = output
    ctx.save_for_backward(
        h2,
        router_weight,
        w13_experts,
        w2_experts,
        logits,
        route_weights,
        route_ids,
        offsets,
        gather,
        scatter,
        reverse,
        h,
        *weight_components,
    )
    ctx.top_k = int(top_k)
    ctx.routing_mode = routing_mode
    ctx.lengths = list(lengths)
    ctx.weight_grad_dtype = weight_grad_dtype
    ctx.compute_precision = compute_precision
    ctx.swiglu_limit = swiglu_limit
    ctx.mark_non_differentiable(
        counts, logits, route_weights, route_ids, offsets, gather, scatter, reverse, h
    )


def _autograd_backward(
    ctx,
    grad_output,
    grad_aux,
    _grad_counts,
    grad_probability_sum,
    _grad_logits,
    _grad_route_weights,
    _grad_route_ids,
    _grad_offsets,
    _grad_gather,
    _grad_scatter,
    _grad_reverse,
    _grad_h,
):
    saved_tensors = ctx.saved_tensors
    residuals, components = saved_tensors[:12], saved_tensors[12:]
    h2, router_weight = residuals[:2]
    if grad_aux is None:
        grad_aux = torch.zeros((), dtype=torch.float32, device=h2.device)
    if grad_probability_sum is None:
        grad_probability_sum = torch.zeros(
            router_weight.shape[0], dtype=torch.float32, device=h2.device
        )
    # Inputs: h2, residual, router_weight, w13_experts, w2_experts, ...
    needs = ctx.needs_input_grad
    grad_h2, grad_residual, grad_router, grad_w13, grad_w2 = _backward_op(
        grad_output,
        grad_aux,
        grad_probability_sum,
        *residuals,
        ctx.top_k,
        ctx.routing_mode,
        ctx.lengths,
        ctx.weight_grad_dtype,
        ctx.compute_precision,
        ctx.swiglu_limit,
        list(components),
        bool(needs[2]),
        bool(needs[3]),
        bool(needs[4]),
    )
    return (
        grad_h2,
        grad_residual,
        requested(grad_router, needs[2]),
        requested(grad_w13, needs[3]),
        requested(grad_w2, needs[4]),
        *([None] * 11),
        [None] * len(components),
    )


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)
fp8.register_container_dispatch(_forward_op, _backward_op)


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
    compute_precision=None,
    swiglu_limit=None,
):
    from ...moe import _production_shared_base

    rows = h2.numel() // h2.shape[-1]
    normalized_lengths = (rows,) if lengths is None else tuple(int(x) for x in lengths)
    bias = (
        torch.empty(0, dtype=h2.dtype, device=h2.device)
        if router_bias is None
        else router_bias
    )
    base = _production_shared_base(
        h2,
        residual,
        shared_w13,
        shared_w2,
        shared_gate_weight,
        shared_mode,
        swiglu_limit,
    )
    precision, w13_experts, w2_experts, grad_dtype = fp8.prepare(
        compute_precision, w13_experts, w2_experts, h2.dtype
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
        grad_dtype,
        precision,
        None if swiglu_limit is None else float(swiglu_limit),
        [],
    )
    return outputs[:4]


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="moe",
        implementation_id="sonicmoe.moe",
        provider="sonicmoe",
        priority=95,
        deterministic=True,
        supports=supports,
        apply=apply,
    )
)

__all__ = [
    "CAPABILITIES",
    "IMPLEMENTATION",
    "ROW_ALIGNMENT",
    "activation",
    "apply",
    "available",
    "backward",
    "down_backward",
    "down_projection",
    "forward",
    "kernels",
    "metadata",
    "supports",
    "up_backward",
    "up_projection",
]
