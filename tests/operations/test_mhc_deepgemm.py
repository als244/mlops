"""Numerical and compilation qualification for fused four-stream preprocessing."""

import pytest
import torch

import mlops
from mlops.providers.deepgemm import hyper_connection as provider
from mlops.providers.native_torch.hyper_connection import pre as reference

pytestmark = pytest.mark.gpu


def require_backend():
    pytest.importorskip("deep_gemm")
    pytest.importorskip("tile_kernels")
    if torch.cuda.get_device_capability() not in ((9, 0), (10, 0)):
        pytest.skip("SM90/SM100 only")


def close(actual, expected):
    assert torch.isfinite(actual).all()
    a, b = actual.float(), expected.float()
    relative_rms = torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(
        b
    ).clamp_min(1e-12)
    assert relative_rms < 0.012, float(relative_rms)
    if actual.dtype == torch.bfloat16:
        torch.testing.assert_close(a, b, rtol=0.03, atol=0.003)
    else:
        torch.testing.assert_close(a, b, rtol=0.005, atol=0.0003)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("projection_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("train_parameters", [False, True])
def test_fused_pre_forward_backward(compiled, projection_dtype, train_parameters):
    require_backend()
    torch.manual_seed(7301)
    x = (
        torch.randn(2, 64, 4, 128, device="cuda", dtype=torch.bfloat16) * 0.3
    ).requires_grad_()
    w = (
        torch.randn(24, 512, device="cuda", dtype=projection_dtype) * 0.05
    ).requires_grad_(train_parameters)
    b = (torch.randn(24, device="cuda") * 0.1).requires_grad_(train_parameters)
    s = torch.full((3,), 0.4, device="cuda").requires_grad_(train_parameters)
    values = (x, w, b, s)
    copies = tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in values)
    expected = reference(*copies)
    fn = torch.compile(mlops.mhc_pre, fullgraph=True) if compiled else mlops.mhc_pre
    with mlops.dispatch.use_implementation("mhc_pre", "deepgemm.mhc_pre"):
        actual = fn(*values)
        cotangents = tuple(torch.randn_like(y) * 0.1 for y in expected)
        ag = torch.autograd.grad(
            actual, tuple(t for t in values if t.requires_grad), cotangents
        )
    eg = torch.autograd.grad(
        expected, tuple(t for t in copies if t.requires_grad), cotangents
    )
    for a, e in zip((*actual, *ag), (*expected, *eg), strict=True):
        close(a, e)
    assert actual[0].dtype == actual[1].dtype == torch.float32
    assert actual[2].dtype == torch.bfloat16


def test_split_projection_and_saved_state():
    require_backend()
    torch.manual_seed(11)
    x = torch.randn(
        128, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    w = torch.randn(24, 512, device="cuda", dtype=torch.bfloat16) * 0.01
    b = torch.zeros(24, device="cuda")
    s = torch.full((3,), 0.1, device="cuda")
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (saved.append(t), t)[1], lambda t: t
    ):
        actual = provider._forward(x, w, b, s, 1e-5, 1e-6, 20, 4)
    expected = reference(x, w, b, s)
    for a, e in zip(actual[:3], expected, strict=True):
        close(a, e)
    assert not any(t.dtype == torch.float32 and t.numel() == x.numel() for t in saved)
    torch.autograd.grad(actual[2].sum(), x)


def test_support_rejects_other_architectures_and_geometry(monkeypatch):
    require_backend()
    x = torch.empty(32, 4, 128, device="cuda", dtype=torch.bfloat16)
    args = (
        x,
        torch.empty(24, 512, device="cuda"),
        torch.empty(24, device="cuda"),
        torch.empty(3, device="cuda"),
    )
    monkeypatch.setattr(provider, "_available", lambda: True)
    for architecture in ((9, 0), (10, 0)):
        monkeypatch.setattr(
            torch.cuda,
            "get_device_capability",
            lambda *a, architecture=architecture: architecture,
        )
        assert provider._supports(*args, surface="semantic")
    for architecture in ((8, 0), (8, 9), (12, 0)):
        monkeypatch.setattr(
            torch.cuda,
            "get_device_capability",
            lambda *a, architecture=architecture: architecture,
        )
        assert not provider._supports(*args, surface="semantic")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    assert not provider._supports(x[:31], *args[1:], surface="semantic")


def test_custom_op_registration():
    require_backend()
    x = torch.randn(32, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(24, 512, device="cuda", dtype=torch.bfloat16) * 0.01
    b = torch.zeros(24, device="cuda")
    s = torch.full((3,), 0.1, device="cuda")
    torch.library.opcheck(provider._forward, (x, w, b, s, 1e-5, 1e-6, 20, 1))


def test_autocast_preserves_coefficients_and_gradients():
    require_backend()
    torch.manual_seed(119)
    x = torch.randn(32, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(24, 512, device="cuda", dtype=torch.bfloat16) * 0.01
    b = torch.zeros(24, device="cuda")
    s = torch.full((3,), 0.1, device="cuda")
    with mlops.dispatch.use_implementation("mhc_pre", "deepgemm.mhc_pre"):
        expected = mlops.mhc_pre(x, w, b, s)
        cotangents = tuple(torch.randn_like(y) for y in expected)
        eg = torch.autograd.grad(expected, x, cotangents)[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = mlops.mhc_pre(x, w, b, s)
            ag = torch.autograd.grad(actual, x, cotangents)[0]
    for a, e in zip((*actual, ag), (*expected, eg), strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    assert actual[0].dtype == actual[1].dtype == torch.float32


def test_default_dispatch_and_missing_dependency_fallback(monkeypatch):
    require_backend()
    x = torch.empty(32, 4, 128, device="cuda", dtype=torch.bfloat16)
    args = (
        x,
        torch.empty(24, 512, device="cuda"),
        torch.empty(24, device="cuda"),
        torch.empty(3, device="cuda"),
    )
    explain = mlops.dispatch.explain_implementation
    assert explain("mhc_pre", *args).selected == "deepgemm.mhc_pre"
    monkeypatch.setattr(provider, "_available", lambda: False)
    assert explain("mhc_pre", *args).selected == "tilekernels.mhc_pre.sinkhorn"
    assert explain("mhc_pre", x[:31], *args[1:]).selected == "native_torch.mhc_pre"
