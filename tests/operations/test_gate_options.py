"""Merged options preserve default math and use the standard operation surfaces."""

from importlib.util import find_spec

import pytest
import torch
import torch.nn.functional as F

from mlops import explicit, gated_rms_norm, packed_swiglu, swiglu
from mlops.dispatch import use_implementation


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_gated_norm_options(activation, device):
    if device == "cuda" and find_spec("fla") is None:
        pytest.skip("Fused gated RMSNorm requires mlops[fla]")
    torch.manual_seed(7)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    x = torch.randn(2, 7, 64, device=device, dtype=dtype, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    weight = torch.randn(64, device=device, dtype=torch.float32, requires_grad=True)
    args = (x, gate, weight)
    reference_args = tuple(t.detach().clone().requires_grad_() for t in args)
    rx, rg, rw = reference_args
    normalized = rx.float() * torch.rsqrt(
        rx.float().square().mean(-1, keepdim=True) + 1e-5
    )
    activated = rg.float().sigmoid() if activation == "sigmoid" else F.silu(rg.float())
    expected = (normalized * rw.float() * activated).to(dtype)
    fn = (
        torch.compile(gated_rms_norm, fullgraph=True)
        if device == "cuda"
        else gated_rms_norm
    )
    got = fn(*args, activation=activation)
    dy = torch.randn_like(got)
    expected_grads = torch.autograd.grad(expected, reference_args, dy)
    got_grads = torch.autograd.grad(got, args, dy)
    tol = 0.016 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(got, expected, rtol=tol, atol=tol)
    for actual, reference in zip(got_grads, expected_grads):
        torch.testing.assert_close(actual, reference, rtol=tol, atol=tol)
    if device == "cuda":
        ey, rstd = explicit.gated_rms_norm.forward(*args, activation=activation)
        eg = explicit.gated_rms_norm.backward(dy, *args, rstd, activation=activation)
        torch.testing.assert_close(ey, got, rtol=0, atol=0)
        for actual, reference in zip(eg, got_grads):
            torch.testing.assert_close(actual, reference, rtol=tol, atol=tol)
        assert not any(t.requires_grad for t in (ey, rstd, *eg))


@pytest.mark.parametrize("limit", [0.0, -1.0, float("inf"), float("nan")])
def test_clipping_rejects_invalid_limit_on_both_surfaces(limit):
    x = torch.ones(2, 8)
    for fn, args in (
        (swiglu, (x, x)),
        (packed_swiglu, (x,)),
        (explicit.swiglu.forward, (x, x)),
        (explicit.swiglu.backward, (x, x, x)),
        (explicit.packed_swiglu.forward, (x,)),
        (explicit.packed_swiglu.backward, (x[..., :4], x)),
    ):
        with pytest.raises(ValueError, match="finite and positive"):
            fn(*args, limit=limit)


def test_gated_norm_rejects_unknown_activation_on_both_surfaces():
    x, weight = torch.ones(2, 8), torch.ones(8)
    for fn, args in (
        (gated_rms_norm, (x, x, weight)),
        (explicit.gated_rms_norm.forward, (x, x, weight)),
        (explicit.gated_rms_norm.backward, (x, x, x, weight, torch.ones(2))),
    ):
        with pytest.raises(ValueError, match="activation"):
            fn(*args, activation="relu")


def test_clipping_torch_fallback_preserves_clamp_gradients():
    x = torch.tensor([[-2.0, -0.5, 0.0, 0.5, 2.0, 1.0, -1.0, -2.0]], requires_grad=True)
    ref = x.detach().clone().requires_grad_()
    with use_implementation("packed_swiglu", "native_torch.packed_swiglu"):
        y = packed_swiglu(x, limit=0.5)
    a, b = ref.chunk(2, -1)
    expected = F.silu(a.clamp(max=0.5)) * b.clamp(-0.5, 0.5)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(y.sum(), x)[0],
        torch.autograd.grad(expected.sum(), ref)[0],
        rtol=0,
        atol=0,
    )
