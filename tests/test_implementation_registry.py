"""Tests for exact, context-local operation implementation dispatch."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from collections.abc import Mapping

import pytest
import torch

from mlops.dispatch import (
    Implementation,
    SupportResult,
    capture_dispatch,
    dispatch_manifest,
    estimate_implementation,
    GradcheckCase,
    gradcheck_implementation,
    gradcheck_implementations,
    explain_implementation,
    implementation_pairs,
    implementation_registry,
    resolve_implementation,
    use_implementation,
    use_implementations,
)
from mlops.dispatch.registry import (
    register_implementation,
)
from mlops.rms_norm import rms_norm
from mlops.rope_tables import build_rope_tables


def test_registry_contains_exact_rms_norm_implementations():
    registry = implementation_registry()
    assert set(registry["rms_norm"]) == {
        "builtin.rms_norm.triton",
        "liger.rms_norm",
        "native_torch.rms_norm",
    }
    assert registry["rms_norm"]["liger.rms_norm"].provider == "liger"


def test_nested_overrides_restore_and_trace_exact_ids():
    x = torch.randn(2, 8)
    weight = torch.randn(8)
    with use_implementations({"rms_norm": "native_torch.rms_norm"}):
        assert explain_implementation("rms_norm", x, weight).selected == (
            "native_torch.rms_norm"
        )
        with use_implementation("rms_norm", "builtin.rms_norm.triton"):
            explanation = explain_implementation("rms_norm", x, weight)
            assert explanation.forced == "builtin.rms_norm.triton"
            assert explanation.selected is None
        assert explain_implementation("rms_norm", x, weight).selected == (
            "native_torch.rms_norm"
        )

        with capture_dispatch() as trace:
            rms_norm(x, weight)
    assert trace == {"rms_norm": {"native_torch.rms_norm": 1}}
    assert dict(dispatch_manifest(trace)) == {
        "rms_norm": "native_torch.rms_norm"
    }


def test_forced_unsupported_and_unknown_selections_fail_clearly():
    x = torch.randn(2, 8)
    weight = torch.randn(8)
    with use_implementation("rms_norm", "builtin.rms_norm.triton"):
        with pytest.raises(RuntimeError, match="forced implementation.*unsupported"):
            resolve_implementation("rms_norm", x, weight)
    with pytest.raises(ValueError, match="unknown implementation"):
        with use_implementation("rms_norm", "does.not.exist"):
            pass
    with pytest.raises(ValueError, match="unknown semantic operation"):
        resolve_implementation("does_not_exist", x)


def test_duplicate_ids_and_half_explicit_records_are_rejected():
    operation = "registry_test_duplicate"

    def supports(*_args, surface, **_kwargs):
        del surface
        return SupportResult.yes()

    record = Implementation(
        operation=operation,
        implementation_id="test.duplicate",
        provider="test",
        priority=0,
        deterministic=True,
        supports=supports,
        apply=lambda value: value,
    )
    register_implementation(record)
    with pytest.raises(ValueError, match="duplicate implementation ID"):
        register_implementation(record)
    with pytest.raises(ValueError, match="forward and backward"):
        Implementation(
            operation="bad",
            implementation_id="test.bad",
            provider="test",
            priority=0,
            deterministic=True,
            supports=supports,
            apply=lambda value: value,
            forward=lambda value: value,
        )


def test_apply_only_implementation_is_rejected_on_explicit_surface():
    operation = "registry_test_apply_only"
    register_implementation(
        Implementation(
            operation=operation,
            implementation_id="test.apply_only",
            provider="test",
            priority=0,
            deterministic=True,
            supports=lambda *_args, surface, **_kwargs: SupportResult.yes(),
            apply=lambda value: value,
        )
    )
    explanation = explain_implementation(operation, torch.ones(1), surface="explicit")
    assert explanation.selected is None
    assert "does not expose explicit" in explanation.candidates[
        "test.apply_only"
    ].reason


def test_automatic_selection_tie_breaks_by_exact_id():
    operation = "registry_test_tie"
    for implementation_id in ("test.z", "test.a"):
        register_implementation(
            Implementation(
                operation=operation,
                implementation_id=implementation_id,
                provider="test",
                priority=7,
                deterministic=True,
                supports=lambda *_args, surface, **_kwargs: SupportResult.yes(),
                apply=lambda value: value,
            )
        )
    assert resolve_implementation(operation, torch.ones(1)).implementation_id == (
        "test.a"
    )


def test_registry_metadata_retains_no_tensors():
    def walk(value):
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(key)
                walk(item)
        elif is_dataclass(value):
            for field in fields(value):
                walk(getattr(value, field.name))

    walk(dict(implementation_registry()))


def test_cost_hints_distinguish_zero_from_unknown():
    x = torch.randn(2, 8)
    weight = torch.randn(8)
    native = estimate_implementation(
        "rms_norm",
        x,
        weight,
        implementation_id="native_torch.rms_norm",
    )
    assert native.logical_flops == 5 * x.numel() + 3 * x.shape[0]
    assert native.implementation_flops is None
    assert native.workspace_bytes is None
    assert not native.has_roofline_estimate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_builtin_rms_norm_cost_hints_are_scalar_only():
    x = torch.randn(5, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    hints = estimate_implementation(
        "rms_norm",
        x,
        weight,
        implementation_id="builtin.rms_norm.triton",
        entrypoint="backward",
    )
    assert hints.workspace_bytes == 64 * 4
    assert hints.has_roofline_estimate
    for field in fields(hints):
        assert not isinstance(getattr(hints, field.name), torch.Tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rope_tables_are_inputs_and_not_workspace():
    x = torch.randn(12, 2, 16, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(12, device="cuda", dtype=torch.int32)
    cosine, sine = build_rope_tables(12, 16, 10_000.0, x.device)
    hints = estimate_implementation(
        "rope",
        x,
        positions,
        10_000.0,
        cosine,
        sine,
        implementation_id="builtin.rope.triton_table",
    )
    assert hints.workspace_bytes == 0
    assert any("caller-owned inputs" in note for note in hints.notes)


def test_default_native_torch_pairs_use_exact_ids():
    pairs = implementation_pairs(("rms_norm",))
    assert dict(pairs["rms_norm"]["default"]) == {
        "rms_norm": "builtin.rms_norm.triton"
    }
    assert dict(pairs["rms_norm"]["native_torch"]) == {
        "rms_norm": "native_torch.rms_norm"
    }


def test_native_torch_rms_norm_passes_numerical_gradcheck():
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(4, dtype=torch.float64, requires_grad=True)
    result = gradcheck_implementation(
        "rms_norm", "native_torch.rms_norm", (x, weight)
    )
    assert result.status == "passed", result.reason


def test_gradcheck_suite_reports_passes_and_unsupported_implementations():
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(4, dtype=torch.float64, requires_grad=True)
    results = gradcheck_implementations(
        (
            GradcheckCase(
                "rms_norm",
                (x, weight),
                implementation_ids=(
                    "native_torch.rms_norm",
                    "builtin.rms_norm.triton",
                ),
            ),
        )
    )
    assert [(result.implementation_id, result.status) for result in results] == [
        ("native_torch.rms_norm", "passed"),
        ("builtin.rms_norm.triton", "skipped"),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "implementation_id",
    ("builtin.rms_norm.triton", "liger.rms_norm"),
)
def test_rms_norm_external_vjps_are_repeatable_and_non_mutating(implementation_id):
    torch.manual_seed(8173)
    x = torch.randn(11, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_output = torch.randn_like(x)
    originals = tuple(value.detach().clone() for value in (x, weight, grad_output))
    with use_implementation("rms_norm", implementation_id):
        output = rms_norm(x, weight)
        first = torch.autograd.grad(
            output, (x, weight), grad_output, retain_graph=True
        )
        second = torch.autograd.grad(output, (x, weight), grad_output)
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)
    for actual, expected in zip((x, weight, grad_output), originals, strict=True):
        assert torch.equal(actual.detach(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("implementation_id", "target"),
    (
        (
            "builtin.rms_norm.triton",
            "mlops.rms_norm_builtin_triton_fwd.default",
        ),
        (
            "liger.rms_norm",
            "mlops.rms_norm_liger_fwd.default",
        ),
    ),
)
def test_rms_norm_export_retains_exact_custom_op_target(implementation_id, target):
    class Wrapper(torch.nn.Module):
        def forward(self, x, weight):
            return rms_norm(x, weight)

    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    with use_implementation("rms_norm", implementation_id):
        exported = torch.export.export(Wrapper(), (x, weight), strict=True)
    targets = {
        str(node.target)
        for node in exported.graph.nodes
        if node.op == "call_function"
    }
    assert target in targets
