import pytest
import torch

from mlops import explicit, packed_swiglu, swiglu


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("limit", [0.4, 10.0])
def test_clipped_activation(dtype, compiled, limit):
    torch.manual_seed(18)
    x = (torch.randn(37, 66, device="cuda", dtype=dtype) * 12).requires_grad_()
    with torch.no_grad():
        x[0, :6] = torch.tensor([-10, 10, -11, 11, 0, 1], device="cuda", dtype=dtype)
    with torch.no_grad():
        x[0, :6] *= limit / 10
    ref = x.detach().clone().requires_grad_()
    fn = torch.compile(packed_swiglu, fullgraph=True) if compiled else packed_swiglu
    got = fn(x, limit=limit)
    a, b = ref.chunk(2, -1)
    expected = torch.nn.functional.silu(a.clamp(max=limit)) * b.clamp(-limit, limit)
    dy = torch.randn_like(got)
    (actual_grad,) = torch.autograd.grad(got, x, dy)
    (ref_grad,) = torch.autograd.grad(expected, ref, dy)
    tol = 0.01 if dtype == torch.bfloat16 else 0.002 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(got, expected, rtol=tol, atol=tol)
    torch.testing.assert_close(actual_grad, ref_grad, rtol=tol, atol=tol)


pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_clipping_packed_separate_and_explicit_agree(dtype):
    torch.manual_seed(92)
    packed = (torch.randn(2, 3, 66, device="cuda", dtype=dtype) * 3).requires_grad_()
    gate, up = [t.contiguous().detach().requires_grad_() for t in packed.chunk(2, -1)]
    dy = torch.randn_like(gate)
    limit = 0.7
    y = swiglu(gate, up, limit=limit)
    grads = torch.autograd.grad(y, (gate, up), dy)
    py = packed_swiglu(packed, limit=limit)
    pg = torch.autograd.grad(py, packed, dy)[0]
    ey = explicit.swiglu.forward(gate, up, limit=limit)
    eg = explicit.swiglu.backward(dy, gate, up, limit=limit)
    epy = explicit.packed_swiglu.forward(packed, limit=limit)
    epg = explicit.packed_swiglu.backward(dy, packed, limit=limit)
    torch.testing.assert_close(y, py, rtol=0, atol=0)
    torch.testing.assert_close(y, ey, rtol=0, atol=0)
    torch.testing.assert_close(py, epy, rtol=0, atol=0)
    torch.testing.assert_close(pg, epg, rtol=0, atol=0)
    for actual, packed_grad, explicit_grad in zip(grads, pg.chunk(2, -1), eg):
        torch.testing.assert_close(actual, packed_grad, rtol=0, atol=0)
        torch.testing.assert_close(actual, explicit_grad, rtol=0, atol=0)
    assert not any(t.requires_grad for t in (ey, *eg, epy, epg))
