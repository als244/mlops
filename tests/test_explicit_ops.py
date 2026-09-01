"""Contracts for the public autograd-independent operation entry points."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import mlops
import pytest
import torch

from mlops import (
    embedding as semantic_embedding,
    flash_attention as semantic_flash_attention,
    gelu as semantic_gelu,
    layer_norm as semantic_layer_norm,
    moe as semantic_moe,
    packed_swiglu as semantic_packed_swiglu,
    partial_rope as semantic_partial_rope,
    rms_norm as semantic_rms_norm,
    rope as semantic_rope,
    swiglu as semantic_swiglu,
)
from mlops import explicit
from mlops.kernels.flash_attention import (
    native_flash_attention_supported,
)
from mlops.dispatch import (
    explain_implementation,
    implementation_registry,
    use_implementation,
)


PACKAGE_ROOT = Path(mlops.__file__).resolve().parent


def _registered_forward_names() -> set[str]:
    return {
        operation
        for operation, implementations in implementation_registry().items()
        if any(implementation.forward is not None for implementation in implementations.values())
    }


def _assert_no_autograd(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        assert tensor.grad_fn is None
        assert not tensor.requires_grad


def test_every_opaque_forward_has_one_explicit_operation_module():
    assert set(explicit.__all__) == _registered_forward_names()
    for name in explicit.__all__:
        module = importlib.import_module(
            f"mlops.explicit.{name}"
        )
        assert module.__all__ == ["backward", "forward"]
        assert callable(module.forward)
        assert callable(module.backward)
        assert module.forward.__doc__
        assert module.backward.__doc__


def test_registered_wrappers_live_below_semantic_operations():
    semantic_sources = [
        (path, path.read_text())
        for path in sorted(PACKAGE_ROOT.glob("*.py"))
    ]
    for path, source in semantic_sources:
        assert "@torch.library.custom_op" not in source, path

    owner_sources = []
    provider_root = PACKAGE_ROOT / "providers"
    for path in sorted(provider_root.rglob("*.py")):
        source = path.read_text()
        if "@torch.library.custom_op" not in source:
            continue
        owner_sources.append((path, source))

    assert owner_sources
    for name in explicit.__all__:
        path = PACKAGE_ROOT / "explicit" / f"{name}.py"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            assert not (
                isinstance(function, ast.Attribute)
                and function.attr in {"backward", "grad"}
                and isinstance(function.value, ast.Attribute)
                and isinstance(function.value.value, ast.Name)
                and function.value.value.id == "torch"
                and function.value.attr == "autograd"
            ), path
        assert "resolve_implementation" in path.read_text(), path


def test_explicit_rms_norm_matches_registered_autograd_vjp():
    torch.manual_seed(1)
    x = torch.randn(5, 16, requires_grad=True)
    weight = torch.randn(16, requires_grad=True)
    cotangent = torch.randn_like(x)
    with use_implementation("rms_norm", "native_torch.rms_norm"):
        expected_output = semantic_rms_norm(x, weight)
    expected_gradients = torch.autograd.grad(
        expected_output,
        (x, weight),
        cotangent,
    )

    output, rstd = explicit.rms_norm.forward(x, weight)
    gradients = explicit.rms_norm.backward(cotangent, x, weight, rstd)

    assert torch.equal(output, expected_output.detach())
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    _assert_no_autograd(output, rstd, *gradients)
    assert x.grad is None and weight.grad is None


def test_explicit_layer_norm_matches_registered_autograd_vjp():
    torch.manual_seed(2)
    x = torch.randn(5, 16, requires_grad=True)
    weight = torch.randn(16, requires_grad=True)
    bias = torch.randn(16, requires_grad=True)
    cotangent = torch.randn_like(x)
    with use_implementation("layer_norm", "native_torch.layer_norm"):
        expected_output = semantic_layer_norm(x, weight, bias)
    expected_gradients = torch.autograd.grad(
        expected_output,
        (x, weight, bias),
        cotangent,
    )

    output, mean, rstd = explicit.layer_norm.forward(x, weight, bias)
    gradients = explicit.layer_norm.backward(cotangent, x, weight, mean, rstd)

    assert torch.equal(output, expected_output.detach())
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    _assert_no_autograd(output, mean, rstd, *gradients)


def test_explicit_swiglu_variants_match_registered_autograd_vjps():
    torch.manual_seed(3)
    gate = torch.randn(5, 16, requires_grad=True)
    up = torch.randn(5, 16, requires_grad=True)
    cotangent = torch.randn_like(gate)
    with use_implementation("swiglu", "native_torch.swiglu"):
        expected_output = semantic_swiglu(gate, up)
    expected_gradients = torch.autograd.grad(
        expected_output,
        (gate, up),
        cotangent,
    )
    output = explicit.swiglu.forward(gate, up)
    gradients = explicit.swiglu.backward(cotangent, gate, up)
    assert torch.equal(output, expected_output.detach())
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    _assert_no_autograd(output, *gradients)

    packed = torch.cat((gate.detach(), up.detach()), dim=-1).requires_grad_()
    with use_implementation("packed_swiglu", "native_torch.packed_swiglu"):
        expected_packed_output = semantic_packed_swiglu(packed)
    expected_packed_gradient = torch.autograd.grad(
        expected_packed_output,
        packed,
        cotangent,
    )[0]
    packed_output = explicit.packed_swiglu.forward(packed)
    packed_gradient = explicit.packed_swiglu.backward(cotangent, packed)
    assert torch.equal(packed_output, expected_packed_output.detach())
    torch.testing.assert_close(
        packed_gradient, expected_packed_gradient, rtol=2e-6, atol=2e-6
    )
    _assert_no_autograd(packed_output, packed_gradient)


def test_explicit_embedding_gelu_and_rope_vjps_match_registered_autograd():
    torch.manual_seed(4)
    tokens = torch.tensor([[1, 3, 1, 5]])
    table = torch.randn(11, 8, requires_grad=True)
    expected_embedding = semantic_embedding(tokens, table)
    embedding_cotangent = torch.randn_like(expected_embedding)
    expected_table_gradient = torch.autograd.grad(
        expected_embedding,
        table,
        embedding_cotangent,
    )[0]
    embedded = explicit.embedding.forward(tokens, table)
    table_gradient = explicit.embedding.backward(
        tokens,
        embedding_cotangent,
        table.shape[0],
    )
    assert torch.equal(embedded, expected_embedding.detach())
    assert torch.equal(table_gradient, expected_table_gradient)

    gelu_input = torch.randn(4, 8, requires_grad=True)
    gelu_cotangent = torch.randn_like(gelu_input)
    expected_gelu = semantic_gelu(gelu_input)
    expected_gelu_gradient = torch.autograd.grad(
        expected_gelu,
        gelu_input,
        gelu_cotangent,
    )[0]
    gelu_output = explicit.gelu.forward(gelu_input)
    gelu_gradient = explicit.gelu.backward(gelu_cotangent, gelu_input)
    assert torch.equal(gelu_output, expected_gelu.detach())
    assert torch.equal(gelu_gradient, expected_gelu_gradient)

    rope_input = torch.randn(4, 2, 8, requires_grad=True)
    positions = torch.arange(4, dtype=torch.int32)
    rope_cotangent = torch.randn_like(rope_input)
    inverse = 1.0 / (
        10_000.0
        ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8)
    )
    frequencies = torch.outer(torch.arange(4, dtype=torch.float32), inverse)
    table = torch.cat((frequencies, frequencies), dim=-1)
    cosine, sine = table.cos(), table.sin()
    with use_implementation("rope", "native_torch.rope"):
        expected_rope = semantic_rope(
            rope_input, positions, 10_000.0, cosine, sine
        )
    expected_rope_gradient = torch.autograd.grad(
        expected_rope,
        rope_input,
        rope_cotangent,
    )[0]
    rope_output = explicit.rope.forward(
        rope_input, positions, 10_000.0, cosine, sine
    )
    rope_gradient = explicit.rope.backward(
        rope_cotangent, positions, 10_000.0, cosine, sine
    )
    assert torch.equal(rope_output, expected_rope.detach())
    assert torch.equal(rope_gradient, expected_rope_gradient)

    partial_inverse = 1.0 / (
        10_000.0
        ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4)
    )
    partial_frequencies = torch.outer(
        torch.arange(4, dtype=torch.float32), partial_inverse
    )
    partial_table = torch.cat(
        (partial_frequencies, partial_frequencies), dim=-1
    )
    partial_cosine, partial_sine = partial_table.cos(), partial_table.sin()
    with use_implementation("partial_rope", "native_torch.partial_rope"):
        expected_partial = semantic_partial_rope(
            rope_input,
            positions,
            10_000.0,
            4,
            partial_cosine,
            partial_sine,
        )
    expected_partial_gradient = torch.autograd.grad(
        expected_partial,
        rope_input,
        rope_cotangent,
    )[0]
    partial_output = explicit.partial_rope.forward(
        rope_input,
        positions,
        10_000.0,
        4,
        partial_cosine,
        partial_sine,
    )
    partial_gradient = explicit.partial_rope.backward(
        rope_cotangent,
        positions,
        10_000.0,
        4,
        partial_cosine,
        partial_sine,
    )
    assert torch.equal(partial_output, expected_partial.detach())
    assert torch.equal(partial_gradient, expected_partial_gradient)
    _assert_no_autograd(
        embedded,
        table_gradient,
        gelu_output,
        gelu_gradient,
        rope_output,
        rope_gradient,
        partial_output,
        partial_gradient,
    )


def test_explicit_backward_does_not_mutate_saved_arguments():
    torch.manual_seed(5)
    x = torch.randn(5, 16, requires_grad=True)
    weight = torch.randn(16, requires_grad=True)
    cotangent = torch.randn_like(x)
    _output, rstd = explicit.rms_norm.forward(x, weight)
    snapshots = tuple(value.detach().clone() for value in (x, weight, rstd))

    first = explicit.rms_norm.backward(cotangent, x, weight, rstd)
    second = explicit.rms_norm.backward(cotangent, x, weight, rstd)

    for value, snapshot in zip((x, weight, rstd), snapshots, strict=True):
        assert torch.equal(value, snapshot)
    for first_value, second_value in zip(first, second, strict=True):
        assert torch.equal(first_value, second_value)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_explicit_flash_attention_matches_registered_autograd_vjp():
    torch.manual_seed(6)
    q = torch.randn(8, 4, 64, device="cuda", dtype=torch.bfloat16).requires_grad_()
    k = torch.randn(8, 2, 64, device="cuda", dtype=torch.bfloat16).requires_grad_()
    v = torch.randn(8, 2, 64, device="cuda", dtype=torch.bfloat16).requires_grad_()
    lengths = (3, 5)
    cu_seqlens = torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32)
    use_native = native_flash_attention_supported(q, k, v)
    if not use_native:
        pytest.skip("explicit FlashAttention VJP requires native flash")
    cotangent = torch.randn_like(q)

    expected_output = semantic_flash_attention(q, k, v, lengths)
    expected_gradients = torch.autograd.grad(
        expected_output,
        (q, k, v),
        cotangent,
    )
    output, lse = explicit.flash_attention.forward(
        q,
        k,
        v,
        cu_seqlens,
        5,
        lengths,
    )
    gradients = explicit.flash_attention.backward(
        cotangent,
        q,
        k,
        v,
        output,
        lse,
        cu_seqlens,
        5,
        lengths,
    )

    assert torch.equal(output, expected_output.detach())
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        assert torch.equal(actual, expected)
    _assert_no_autograd(output, lse, *gradients)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_explicit_moe_matches_registered_multi_root_autograd_vjp():
    torch.manual_seed(7)
    h2 = torch.randn(8, 32, device="cuda", dtype=torch.bfloat16).requires_grad_()
    residual = torch.randn_like(h2, requires_grad=True)
    router = torch.randn(32, 4, device="cuda", dtype=torch.float32).requires_grad_()
    w13 = torch.randn(4, 32, 64, device="cuda", dtype=torch.bfloat16).requires_grad_()
    w2 = torch.randn(4, 32, 32, device="cuda", dtype=torch.bfloat16).requires_grad_()
    primals = (h2, residual, router, w13, w2)
    lengths = (8,)

    output, aux, _counts, probability_sum = semantic_moe(
        *primals,
        top_k=2,
        routing_mode="softmax_then_topk",
        lengths=lengths,
    )
    grad_output = torch.randn_like(output)
    grad_aux = torch.tensor(0.37, device="cuda")
    grad_probability_sum = torch.randn_like(probability_sum)
    expected_gradients = torch.autograd.grad(
        (output, aux, probability_sum),
        primals,
        (grad_output, grad_aux, grad_probability_sum),
    )

    explicit_outputs = explicit.moe.forward(
        *primals,
        top_k=2,
        routing_mode="softmax_then_topk",
        lengths=lengths,
    )
    (
        explicit_output,
        explicit_aux,
        _explicit_counts,
        explicit_probability_sum,
        logits,
        route_weights,
        route_ids,
        order,
        offsets,
        slots,
        h13,
    ) = explicit_outputs
    gradients = explicit.moe.backward(
        grad_output,
        grad_aux,
        grad_probability_sum,
        h2,
        router,
        w13,
        w2,
        logits,
        route_weights,
        route_ids,
        order,
        offsets,
        slots,
        h13,
        top_k=2,
        routing_mode="softmax_then_topk",
        lengths=lengths,
    )

    assert torch.equal(explicit_output, output.detach())
    assert torch.equal(explicit_aux, aux.detach())
    assert torch.equal(explicit_probability_sum, probability_sum.detach())
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        assert torch.equal(actual, expected)
    _assert_no_autograd(*explicit_outputs, *gradients)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_trace_visible_moe_composition_matches_opaque_builtin():
    torch.manual_seed(17)
    base = (
        torch.randn(8, 32, device="cuda", dtype=torch.bfloat16),
        torch.randn(8, 32, device="cuda", dtype=torch.bfloat16),
        torch.randn(32, 4, device="cuda", dtype=torch.float32),
        torch.randn(4, 32, 64, device="cuda", dtype=torch.bfloat16),
        torch.randn(4, 32, 32, device="cuda", dtype=torch.bfloat16),
    )
    grad_output = torch.randn_like(base[0])
    grad_aux = torch.tensor(0.37, device="cuda")
    grad_probability_sum = torch.randn(4, device="cuda")

    resolution_kwargs = {
        "top_k": 2,
        "routing_mode": "softmax_then_topk",
        "lengths": (8,),
    }
    assert explain_implementation(
        "moe", *base, **resolution_kwargs
    ).selected == "builtin.moe.grouped_gemm_composed"
    assert explain_implementation(
        "moe", *base, surface="explicit", **resolution_kwargs
    ).selected == "builtin.moe.grouped_gemm"

    def run(implementation_id):
        primals = tuple(value.detach().clone().requires_grad_() for value in base)
        with use_implementation("moe", implementation_id):
            output, aux, counts, probability_sum = semantic_moe(
                *primals,
                **resolution_kwargs,
            )
        gradients = torch.autograd.grad(
            (output, aux, probability_sum),
            primals,
            (grad_output, grad_aux, grad_probability_sum),
        )
        return (output, aux, counts, probability_sum), gradients

    opaque_outputs, opaque_gradients = run("builtin.moe.grouped_gemm")
    composed_outputs, composed_gradients = run(
        "builtin.moe.grouped_gemm_composed"
    )
    for actual, expected in zip(
        composed_outputs, opaque_outputs, strict=True
    ):
        assert torch.equal(actual, expected)
    for actual, expected in zip(
        composed_gradients, opaque_gradients, strict=True
    ):
        assert torch.equal(actual, expected)
