"""Correctness and capture contracts for standalone cross entropy."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch
import torch.nn.functional as functional

from mlops import cross_entropy, explicit
from mlops.dispatch import (
    estimate_implementation,
    gradcheck_implementation,
    implementation_registry,
    use_implementation,
)


def test_cross_entropy_registry_has_builtin_and_native_implementations():
    implementations = implementation_registry()["cross_entropy"]
    assert set(implementations) == {
        "builtin.cross_entropy.triton",
        "native_torch.cross_entropy",
    }


@pytest.mark.parametrize("reduction", ("none", "sum", "mean"))
@pytest.mark.parametrize("weighted", (False, True))
def test_native_cross_entropy_matches_torch(reduction, weighted):
    torch.manual_seed(101)
    logits = torch.randn(9, 17, requires_grad=True)
    targets = torch.tensor([1, 4, -100, 3, 2, 9, 0, 16, 8])
    weight = torch.rand(17) if weighted else None
    cotangent = torch.randn(9) if reduction == "none" else torch.randn(())

    with use_implementation("cross_entropy", "native_torch.cross_entropy"):
        actual = cross_entropy(
            logits,
            targets,
            weight=weight,
            ignore_index=-100,
            reduction=reduction,
        )
    expected = functional.cross_entropy(
        logits,
        targets,
        weight=weight,
        ignore_index=-100,
        reduction=reduction,
    )
    actual_gradient = torch.autograd.grad(actual, logits, cotangent)[0]
    expected_gradient = torch.autograd.grad(expected, logits, cotangent)[0]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_explicit_cross_entropy_matches_arbitrary_native_torch_vjp():
    torch.manual_seed(102)
    logits = torch.randn(7, 19, requires_grad=True)
    targets = torch.tensor([3, -7, 2, 11, 0, 18, 4])
    weight = torch.rand(19)
    cotangent = torch.randn(7)

    with use_implementation("cross_entropy", "native_torch.cross_entropy"):
        expected = cross_entropy(
            logits,
            targets,
            weight=weight,
            ignore_index=-7,
            reduction="none",
        )
        losses, logsumexp = explicit.cross_entropy.forward(
            logits,
            targets,
            weight,
            -7,
        )
        gradient = explicit.cross_entropy.backward(
            cotangent,
            logits,
            targets,
            logsumexp,
            weight,
            -7,
        )
    expected_gradient = torch.autograd.grad(expected, logits, cotangent)[0]

    assert losses.grad_fn is None and logsumexp.grad_fn is None
    assert gradient.grad_fn is None
    assert logits.grad is None
    torch.testing.assert_close(losses, expected.detach())
    torch.testing.assert_close(gradient, expected_gradient)


def test_native_cross_entropy_passes_registry_gradcheck():
    logits = torch.randn(4, 7, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([1, 3, 0, 6])
    result = gradcheck_implementation(
        "cross_entropy",
        "native_torch.cross_entropy",
        (logits, targets),
        kwargs={"reduction": "mean"},
    )
    assert result.status == "passed", result.reason


def test_cross_entropy_cost_hints_are_scalar_only():
    logits = torch.randn(5, 11)
    targets = torch.arange(5)
    hints = estimate_implementation(
        "cross_entropy",
        logits,
        targets,
        implementation_id="native_torch.cross_entropy",
        entrypoint="backward",
    )
    assert hints.logical_flops == 4 * logits.numel()
    assert hints.workspace_bytes is None
    for field in fields(hints):
        assert not isinstance(getattr(hints, field.name), torch.Tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_builtin_cross_entropy_forward_vjp_and_replay(dtype):
    torch.manual_seed(103)
    logits = torch.randn(
        13,
        97,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    targets = torch.tensor(
        [1, 4, -100, 3, 2, 9, 0, 96, 15, 8, 6, 41, 71],
        device="cuda",
    )
    cotangent = torch.randn(13, device="cuda")
    original_cotangent = cotangent.clone()

    with use_implementation("cross_entropy", "builtin.cross_entropy.triton"):
        losses = cross_entropy(logits, targets, reduction="none")
        first = torch.autograd.grad(
            losses,
            logits,
            cotangent,
            retain_graph=True,
        )[0]
        second = torch.autograd.grad(losses, logits, cotangent)[0]
        explicit_losses, logsumexp = explicit.cross_entropy.forward(
            logits, targets
        )
        explicit_gradient = explicit.cross_entropy.backward(
            cotangent, logits, targets, logsumexp
        )

    reference_logits = logits.detach().float().requires_grad_()
    reference_losses = functional.cross_entropy(
        reference_logits,
        targets,
        reduction="none",
    )
    reference_gradient = torch.autograd.grad(
        reference_losses,
        reference_logits,
        cotangent,
    )[0]
    tolerance = 4e-3 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(losses, reference_losses, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(explicit_losses, losses, rtol=0, atol=0)
    torch.testing.assert_close(first.float(), reference_gradient, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    torch.testing.assert_close(explicit_gradient, first, rtol=0, atol=0)
    assert torch.equal(cotangent, original_cotangent)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_builtin_cross_entropy_export_retains_exact_target():
    class Wrapper(torch.nn.Module):
        def forward(self, logits, targets):
            return cross_entropy(logits, targets)

    logits = torch.randn(5, 31, device="cuda", dtype=torch.bfloat16)
    targets = torch.arange(5, device="cuda")
    with use_implementation("cross_entropy", "builtin.cross_entropy.triton"):
        exported = torch.export.export(Wrapper(), (logits, targets), strict=True)
    targets_in_graph = {
        str(node.target)
        for node in exported.graph.nodes
        if node.op == "call_function"
    }
    assert "mlops.cross_entropy_builtin_triton_fwd.default" in targets_in_graph
