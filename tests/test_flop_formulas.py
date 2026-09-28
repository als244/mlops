"""Every registered operator declares its logical work, and the counts add up.

The formulas run on fake tensors on the CPU: they read shapes and static
arguments and nothing else, so nothing here needs a device.
"""

from __future__ import annotations

import importlib

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.flop_counter import FlopCounterMode

from mlops.dispatch import (
    estimate_implementation,
    has_flop_formula,
    implementation_registry,
)
from mlops.dispatch import logical_costs as logical
from mlops.dispatch.costs import operation_estimators
from mlops.providers import ensure_implementations_registered

# Operators that are not implementations register on import like the rest.
importlib.import_module("mlops.preparation.packed_sequence")
ensure_implementations_registered()

ops = torch.ops.mlops

#: The operations with an opaque operator somewhere in the catalog. Each has a
#: canonical estimator, which is where every operator's formula comes from.
OPAQUE_OPERATIONS = frozenset(
    {
        "adamw",
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
    }
)


@pytest.fixture
def fake():
    with FakeTensorMode() as mode:
        yield mode


def _count(operator, *args, **kwargs) -> int:
    with FlopCounterMode(display=False) as counter:
        operator(*args, **kwargs)
    return counter.get_total_flops()


def _tensor(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype)


def _f32(*shape):
    return _tensor(*shape, dtype=torch.float32)


def _i32(*shape):
    return _tensor(*shape, dtype=torch.int32)


def _i64(*shape):
    return _tensor(*shape, dtype=torch.int64)


def test_every_registered_operator_has_a_flop_formula():
    names = sorted(ops._dir)
    assert names
    missing = [name for name in names if not has_flop_formula(getattr(ops, name))]
    assert not missing, missing


def test_every_opaque_operation_has_a_canonical_estimator():
    estimators = operation_estimators()
    assert OPAQUE_OPERATIONS <= set(estimators)
    assert set(estimators) <= set(implementation_registry())


def test_canonical_estimates_reach_estimate_implementation():
    x = torch.randn(6, 16)
    weight = torch.randn(16)
    forward = estimate_implementation("rms_norm", x, weight)
    backward = estimate_implementation("rms_norm", x, weight, entrypoint="backward")
    assert forward.logical_flops == 5 * 96 + 3 * 6
    assert backward.logical_flops == 10 * 96 + 6


def test_normalizations_count_a_few_operations_per_element(fake):
    x, weight = _tensor(6, 16), _tensor(16)
    rstd = _f32(6)
    assert _count(ops.rms_norm_builtin_triton_fwd, x, weight, 1e-5, None) == 498
    assert _count(ops.rms_norm_builtin_triton_bwd, x, x, weight, rstd, None) == 966
    # Another provider of the same operation reports the same logical work.
    assert _count(ops.rms_norm_liger_fwd, x, weight, 1e-5) == 498
    assert _count(ops.rms_norm_liger_bwd, x, x, weight, rstd) == 966
    layer_forward = ops.layer_norm_builtin_triton_fwd
    layer_backward = ops.layer_norm_builtin_triton_bwd
    assert _count(layer_forward, x, weight, weight, 1e-5, None) == 792
    assert _count(layer_backward, x, x, weight, rstd, rstd, None) == 12 * 96 + 12
    wide = _tensor(8, 2, 16)
    assert _count(ops.l2_norm_fla_fwd, wide, 1e-6) == 3 * 256 + 16
    assert _count(ops.l2_norm_fla_bwd, wide, wide, _f32(8, 2), 1e-6) == 5 * 256
    gated_forward = ops.gated_rms_norm_fla_fwd
    gated_backward = ops.gated_rms_norm_fla_bwd
    assert _count(gated_forward, wide, wide, weight, 1e-5) == 10 * 256 + 48
    assert (
        _count(gated_backward, wide, wide, wide, weight, _f32(16), 1e-5)
        == 18 * 256 + 16
    )


def test_activations_count_the_same_work_in_every_variant(fake):
    gate, up = _tensor(6, 32), _tensor(6, 32)
    packed = _tensor(6, 64)
    forwards = (ops.swiglu_builtin_triton_fwd, ops.swiglu_builtin_triton_legacy_fwd)
    backwards = (ops.swiglu_builtin_triton_bwd, ops.swiglu_builtin_triton_legacy_bwd)
    for forward in forwards:
        assert _count(forward, gate, up) == 5 * 192
    for backward in backwards:
        assert _count(backward, gate, gate, up) == 10 * 192
    packed_forwards = (
        ops.packed_swiglu_builtin_triton_fwd,
        ops.packed_swiglu_builtin_triton_legacy_fwd,
    )
    packed_backwards = (
        ops.packed_swiglu_builtin_triton_bwd,
        ops.packed_swiglu_builtin_triton_legacy_bwd,
    )
    for forward in packed_forwards:
        assert _count(forward, packed) == 5 * 192
    for backward in packed_backwards:
        assert _count(backward, gate, packed) == 10 * 192
    assert _count(ops.gelu_builtin_aten_explicit_backward_fwd, gate) == 8 * 192
    assert _count(ops.gelu_builtin_aten_explicit_backward_bwd, gate, gate) == 12 * 192


def test_rotations_count_two_products_and_a_sum_per_rotated_element(fake):
    x = _tensor(6, 2, 16)
    positions = _i32(6)
    cosine, sine = _f32(6, 16), _f32(6, 16)
    forwards = (
        ops.rope_builtin_triton_table_fwd,
        ops.rope_builtin_triton_analytic_fwd,
    )
    backwards = (
        ops.rope_builtin_triton_table_bwd,
        ops.rope_builtin_triton_analytic_bwd,
    )
    for forward in forwards:
        assert _count(forward, x, positions, 1e4, cosine, sine) == 3 * 192
    for backward in backwards:
        assert _count(backward, x, positions, 1e4, cosine, sine) == 3 * 192
    half = _f32(6, 8)
    partial_forward = ops.partial_rope_builtin_triton_fwd
    partial_backward = ops.partial_rope_builtin_triton_bwd
    assert _count(partial_forward, x, positions, 1e4, 8, half, half) == 288
    assert _count(partial_backward, x, positions, 1e4, 8, half, half) == 288


def test_gathers_do_no_arithmetic_and_scatters_add(fake):
    tokens = _i64(1, 4)
    weight = _tensor(17, 16)
    assert _count(ops.embedding_builtin_deterministic_fwd, tokens, weight, None) == 0
    grad = _tensor(1, 4, 16)
    assert _count(ops.embedding_builtin_deterministic_bwd, tokens, grad, 17, None) == 64
    assert _count(ops.prepare_packed_sequence_metadata, _f32(4), [2, 2], 2) == 0


def test_losses_count_the_logits_they_touch(fake):
    logits = _f32(7, 31)
    targets = _i64(7)
    losses = _f32(7)
    forward = ops.cross_entropy_builtin_triton_fwd
    backward = ops.cross_entropy_builtin_triton_bwd
    assert _count(forward, logits, targets, -100) == 4 * 217 + 14
    assert _count(backward, losses, logits, targets, losses, -100) == 4 * 217
    hidden, head = _tensor(7, 16), _tensor(31, 16)
    assert (
        _count(ops.head_loss_builtin_chunked_fwd, hidden, head, targets, 4, 7, None)
        == 6 * 7 * 16 * 31 + 6 * 7 * 31
    )


def test_attention_is_bounded_by_tokens_times_the_longest_sequence(fake):
    q, k, v = _tensor(8, 4, 64), _tensor(8, 2, 64), _tensor(8, 2, 64)
    cu_seqlens = _i32(4)
    lse = _f32(4, 8)
    pairs = 8 * 5
    forward = ops.flash_attention_builtin_aten_fwd
    backward = ops.flash_attention_builtin_aten_bwd
    causal = _count(forward, q, k, v, cu_seqlens, 5, True, None, False)
    full = _count(forward, q, k, v, cu_seqlens, 5, False, None, False)
    assert causal == 2 * 4 * 64 * pairs
    assert full == 4 * 4 * 64 * pairs
    assert (
        _count(backward, q, q, k, v, q, lse, cu_seqlens, 5, True, None, False)
        == 5 * 4 * 64 * pairs
    )
    sq, sk, sv = _tensor(8, 2, 16), _tensor(8, 2, 16), _tensor(8, 2, 12)
    indices = _i32(8, 2)
    sparse_forward = ops.dsa_attention_builtin_sparse_fwd
    sparse_backward = ops.dsa_attention_builtin_sparse_bwd
    selected = 2 * 8 * 2 * (16 + 12)
    assert _count(sparse_forward, sq, sk, sv, indices, [8]) == 2 * selected
    assert (
        _count(sparse_backward, sv, sq, sk, sv, indices, _f32(2, 8), [8])
        == 5 * selected
    )


def test_hybrid_operators_count_their_taps_chunks_and_state(fake):
    x, weight = _tensor(8, 16), _tensor(16, 1, 3)
    cumulative = _i64(3)
    conv_forward = ops.causal_conv_silu_fla_fwd
    conv_backward = ops.causal_conv_silu_fla_bwd
    assert _count(conv_forward, x, weight, cumulative) == (2 * 3 + 5) * 128
    assert _count(conv_backward, x, x, weight, cumulative) == (4 * 3 + 6) * 128
    q, k, v = _tensor(8, 2, 16), _tensor(8, 2, 16), _tensor(8, 4, 16)
    beta, a = _tensor(8, 4), _tensor(8, 4)
    a_log, dt_bias = _f32(4), _f32(4)
    empty_cumulative, empty_chunks = _i64(0), _i64(0, 2)
    chunk = logical.LINEAR_ATTENTION_CHUNK
    per_token = 4 * 16 * 16 + 2 * chunk * (16 + 16)
    work = 8 * 4 * per_token
    inputs = (q, k, v, beta, a, a_log, dt_bias)
    assert (
        _count(
            ops.linear_attention_fla_gated_delta_rule_fwd,
            *inputs,
            empty_cumulative,
            empty_chunks,
            0.25,
        )
        == work + 8 * 8 * 4
    )
    assert (
        _count(
            ops.linear_attention_fla_gated_delta_rule_bwd,
            v,
            *inputs,
            _f32(1, 8, 4),
            _tensor(1, 8, 4, chunk),
            empty_cumulative,
            empty_chunks,
            0.25,
        )
        == 5 * work // 2 + 8 * 8 * 4
    )


def test_mixture_of_experts_charges_every_assignment_once(fake):
    h2, residual = _tensor(8, 32), _tensor(8, 32)
    router = _tensor(4, 32).T
    w13, w2 = _tensor(4, 32, 64), _tensor(4, 32, 32)
    bias = _tensor(0)
    rows, width, experts, top_k, packed, expert_width = 8, 32, 4, 2, 64, 32
    assignments = rows * top_k
    router_flops = 2 * rows * width * experts + 3 * rows * experts
    up = 2 * assignments * width * packed
    activation = 5 * assignments * expert_width
    down = 2 * assignments * expert_width * width
    combine = 2 * assignments * width
    forward = router_flops + up + activation + down + combine
    static = (top_k, "softmax_then_topk", 1, 1, 1.0, [8], "float32", None)
    for fused in (ops.moe_builtin_grouped_gemm_fwd, ops.moe_scattermoe_fwd):
        assert _count(fused, h2, residual, router, w13, w2, bias, *static) == forward

    # The composed provider splits the same work into two regions.
    logits, route_weights = _tensor(8, 4), _f32(8, 2)
    route_ids, order, offsets, slots = _i32(8, 2), _i32(16), _i32(5), _i32(8, 2)
    h13 = _tensor(16, 64)
    prepare_forward = ops.moe_builtin_grouped_gemm_prepare_fwd
    finish_forward = ops.moe_builtin_grouped_gemm_finish_fwd
    prepare = _count(prepare_forward, h2, router, w13, bias, *static)
    finish = _count(
        finish_forward,
        h13,
        w2,
        route_weights,
        order,
        offsets,
        slots,
        residual,
        top_k,
        None,
    )
    assert prepare == router_flops + up
    assert finish == activation + down + combine
    assert prepare + finish == forward

    backward = (
        2 * router_flops
        - 3 * rows * experts
        + 2 * up
        + 2 * activation
        + 2 * down
        + combine
    )
    residuals = (logits, route_weights, route_ids, order, offsets, slots, h13)
    for fused in (ops.moe_builtin_grouped_gemm_bwd, ops.moe_scattermoe_bwd):
        assert (
            _count(
                fused,
                h2,
                _f32(()),
                _f32(4),
                h2,
                router,
                w13,
                w2,
                *residuals,
                top_k,
                "softmax_then_topk",
                [8],
                None,
            )
            == backward
        )
    prepare_backward = _count(
        ops.moe_builtin_grouped_gemm_prepare_bwd,
        _f32(()),
        _f32(4),
        route_weights,
        h13,
        h2,
        router,
        w13,
        *residuals[:-1],
        top_k,
        "softmax_then_topk",
        [8],
        None,
    )
    finish_backward = _count(
        ops.moe_builtin_grouped_gemm_finish_bwd,
        h2, h13, w2, route_weights, order, offsets, slots, top_k, None,
    )
    donated = _count(
        ops.moe_builtin_grouped_gemm_finish_bwd_donate_h13,
        h2, h13, w2, route_weights, order, offsets, slots, top_k, None,
    )
    assert prepare_backward + finish_backward == backward
    assert donated == finish_backward


def test_optimizer_updates_count_per_element(fake):
    state = tuple(_f32(10) for _ in range(4))
    step = _f32(1)
    scalars = tuple(torch.tensor(0.5, dtype=torch.float64) for _ in range(5))
    static = (1.0, *scalars, False, False, False, 0)
    assert _count(ops.adamw, *state, step, *static) == 200
    assert _count(ops.master_adamw, state[0], *state, step, *static) == 200
