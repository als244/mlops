"""Architecture filtering and the PyTorch SDPA fallback for attention."""

from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import mlops
from mlops.dispatch import (
    capture_dispatch,
    dispatch_manifest,
    explain_implementation,
    resolve_implementation,
    use_implementations,
)
from mlops.providers.builtin import flash_attention as builtin


@pytest.mark.parametrize("capability", [(7, 0), (7, 5), (8, 0), (8, 6), (9, 0)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_architecture_filters_the_registered_attention_candidates(
    monkeypatch, capability, dtype
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(builtin, "_maybe_activate_fa3", lambda device: False)
    with FakeTensorMode():
        q, k, v = (torch.empty(64, 2, 32, device="cuda", dtype=dtype) for _ in range(3))
        offsets = torch.empty(4, device="cuda", dtype=torch.int32)
        explanation = explain_implementation("flash_attention", q, k, v, offsets, 32)
    optimized = "builtin.flash_attention.aten"
    fallback = "native_torch.flash_attention"
    assert explanation.forced is None
    assert explanation.selected == (optimized if capability >= (8, 0) else fallback)
    if capability < (8, 0):
        assert "compute capability 8.0" in explanation.candidates[optimized].reason


def test_an_unsupported_explicit_choice_reports_the_architecture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    with FakeTensorMode():
        q, k, v = (
            torch.empty(64, 2, 32, device="cuda", dtype=torch.float16) for _ in range(3)
        )
        offsets = torch.empty(4, device="cuda", dtype=torch.int32)
        with (
            use_implementations({"flash_attention": "builtin.flash_attention.aten"}),
            pytest.raises(
                RuntimeError, match="forced implementation.*compute capability 8.0"
            ),
        ):
            resolve_implementation("flash_attention", q, k, v, offsets, 32)


def test_cpu_fallback_does_not_query_cuda(monkeypatch):
    def unexpected_query(*args, **kwargs):
        raise AssertionError("CPU attention must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_query)
    q, k, v = (torch.empty(16, 2, 32) for _ in range(3))
    offsets = torch.tensor([0, 16], dtype=torch.int32)
    explanation = explain_implementation("flash_attention", q, k, v, offsets, 16)
    assert explanation.selected == "native_torch.flash_attention"


@pytest.mark.gpu
def test_pre_ampere_fallback_forward_backward_and_compilation():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability() >= (8, 0):
        pytest.skip("exercises the pre-Ampere fallback on real hardware")
    torch.manual_seed(7)
    q, k, v = (
        torch.randn(64, heads, 32, device="cuda", dtype=torch.float16).requires_grad_()
        for heads in (4, 2, 2)
    )
    offsets = torch.tensor([0, 13, 32, 64, 64], device="cuda", dtype=torch.int32)
    cotangent = torch.randn_like(q)
    with capture_dispatch() as trace:
        actual = mlops.flash_attention(q, k, v, offsets, 32)
    selected = dispatch_manifest(trace)
    assert dict(selected) == {"flash_attention": "native_torch.flash_attention"}
    gradients = torch.autograd.grad(actual, (q, k, v), cotangent)

    # An independent per-sequence SDPA expression checks the packed mask and GQA.
    outputs = []
    for start, stop in ((0, 13), (13, 32), (32, 64)):
        query = q[start:stop].transpose(0, 1).unsqueeze(0)
        key, value = (
            item[start:stop].repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0)
            for item in (k, v)
        )
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    expected = torch.cat(outputs)
    expected_gradients = torch.autograd.grad(expected, (q, k, v), cotangent)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
    for got, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(got, want, rtol=3e-3, atol=3e-3)

    # Explicitly selecting a supported identity remains available for capture.
    compiled = torch.compile(mlops.flash_attention, fullgraph=True)
    with use_implementations(selected):
        output = compiled(q, k, v, offsets, 32)
        compiled_gradients = torch.autograd.grad(output, (q, k, v), cotangent)
    torch.testing.assert_close(output, actual, rtol=3e-3, atol=3e-3)
    for got, want in zip(compiled_gradients, gradients, strict=True):
        torch.testing.assert_close(got, want, rtol=3e-3, atol=3e-3)

    # The same call also selects SDPA automatically during full-graph capture.
    torch._dynamo.reset()
    automatic = torch.compile(mlops.flash_attention, fullgraph=True)
    output = automatic(q, k, v, offsets, 32)
    automatic_gradients = torch.autograd.grad(output, (q, k, v), cotangent)
    torch.testing.assert_close(output, actual, rtol=3e-3, atol=3e-3)
    for got, want in zip(automatic_gradients, gradients, strict=True):
        torch.testing.assert_close(got, want, rtol=3e-3, atol=3e-3)


def test_compiled_resolution_does_not_execute_intermediate_tensors():
    calls = []

    @torch.library.custom_op("mlops_test::dispatch_input_probe", mutates_args=())
    def probe(value: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return value.clone()

    @probe.register_fake
    def probe_fake(value):
        return torch.empty_like(value)

    def attention(value, offsets):
        projected = probe(value)
        return mlops.flash_attention(projected, projected, projected, offsets, 16)

    value = torch.randn(16, 2, 32)
    offsets = torch.tensor([0, 16], dtype=torch.int32)
    compiled = torch.compile(attention, backend="eager", fullgraph=True)
    result = compiled(value, offsets)
    assert calls == [1]  # One execution; capture did not run the input producer.
    assert result.shape == value.shape
    assert torch.isfinite(result).all()
