"""cuLA training preserves MLOps' packed explicit-VJP and autograd contracts."""

import gc
import weakref

import pytest
import torch

import mlops
from mlops.dispatch import use_implementation

from .test_kda import recurrence

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    "provider",
    ["cula.kimi_delta_attention.sm90", "cula.kimi_delta_attention.sm90.recompute"],
)
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("beta_dtype", [torch.bfloat16, torch.float32])
def test_cula_sm90_values_gradients_and_metadata(compiled, beta_dtype, provider):
    pytest.importorskip("cula")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 provider")
    torch.manual_seed(719)
    lengths = (65, 127)
    shape = (sum(lengths), 4, 128)
    tensors = [
        mlops.l2_norm(torch.randn(shape, device="cuda", dtype=torch.bfloat16))
        for _ in range(2)
    ]
    tensors += [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16),
        torch.empty(shape, device="cuda").uniform_(-5, -0.01),
        torch.rand(shape[:2], device="cuda", dtype=beta_dtype),
    ]
    inputs = [t.requires_grad_() for t in tensors]
    reference = [t.detach().clone().requires_grad_() for t in tensors]
    cu = torch.tensor([0, 65, 192], device="cuda", dtype=torch.int64)
    chunks = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]], device="cuda", dtype=torch.int64
    )
    refs = [weakref.ref(t) for t in (cu, chunks)]
    fn = (
        torch.compile(mlops.kimi_delta_attention, fullgraph=True)
        if compiled
        else mlops.kimi_delta_attention
    )
    dy = torch.randn_like(inputs[2])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with (
        torch.cuda.stream(stream),
        use_implementation("kimi_delta_attention", provider),
    ):
        out = fn(*inputs, cu, chunks)
        grads = torch.autograd.grad(out, inputs, dy)
        if not compiled:
            result = mlops.explicit.kimi_delta_attention.forward(*inputs, cu, chunks)
            vjp = mlops.explicit.kimi_delta_attention.backward(
                dy, *inputs, *result[1:], cu, chunks
            )
            torch.testing.assert_close(out, result[0], rtol=0, atol=0)
            for a, b in zip(grads, vjp):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            del result, vjp
    torch.cuda.current_stream().wait_stream(stream)
    target = recurrence(*reference, lengths)
    expected_grads = torch.autograd.grad(target, reference, dy)
    errors = []
    for got, expected in zip((out, *grads), (target, *expected_grads)):
        assert torch.isfinite(got).all()
        error = (
            got.float() - expected.float()
        ).norm() / expected.float().norm().clamp_min(1e-8)
        errors.append(error.item())
    assert max(errors) < 0.02, errors
    del out, grads, cu, chunks
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_sm90_provider_priority_and_architecture_fallback(monkeypatch):
    pytest.importorskip("cula")
    from importlib.util import find_spec

    from torch._subclasses.fake_tensor import FakeTensorMode

    from mlops.dispatch import explain_implementation

    with FakeTensorMode():
        shape = (128, 64, 128)
        q = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        g = torch.empty(shape, device="cuda", dtype=torch.float32)
        beta = torch.empty(shape[:2], device="cuda", dtype=torch.bfloat16)
        cu = torch.empty(2, device="cuda", dtype=torch.int64)
        chunks = torch.empty(2, 2, device="cuda", dtype=torch.int64)
        args = (q, q, q, g, beta, cu, chunks)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
        expected = (
            "cula.kimi_delta_attention.sm90.recompute"
            if find_spec("cula._cudac_sm90") is not None
            else "cula.kimi_delta_attention.sm90"
        )
        assert (
            explain_implementation("kimi_delta_attention", *args).selected == expected
        )
        with use_implementation(
            "kimi_delta_attention", "cula.kimi_delta_attention.sm90"
        ):
            assert (
                explain_implementation("kimi_delta_attention", *args).selected
                == "cula.kimi_delta_attention.sm90"
            )
        # SM100 is deliberately not routed to a kernel validated only on SM90.
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 0))
        assert (
            explain_implementation("kimi_delta_attention", *args).selected
            == "fla.kimi_delta_attention.chunk"
        )
        key = torch.empty(128, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.empty(128, 64, device="cuda", dtype=torch.float32)
        position = torch.empty(4, 128, device="cuda", dtype=torch.bfloat16)
        ends = torch.empty(2, dtype=torch.int64)
        assert (
            explain_implementation(
                "pooled_topk", q, key, key, weights, position, ends, top_k=128
            ).selected
            == "builtin.pooled_topk.triton"
        )
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
        assert (
            explain_implementation(
                "pooled_topk", q, key, key, weights, position, ends, top_k=128
            ).selected
            == "cudnn.pooled_topk.sm90"
        )


@pytest.mark.parametrize(
    "provider",
    ["cula.kimi_delta_attention.sm90", "cula.kimi_delta_attention.sm90.recompute"],
)
def test_cula_autograd_saves_only_backward_inputs(provider):
    """Raw decays must not stay live solely for their already-known FP32 dtype."""
    pytest.importorskip("cula")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 provider")
    shape = (65, 4, 128)
    q, k = [
        mlops.l2_norm(torch.randn(shape, device="cuda", dtype=torch.bfloat16))
        for _ in range(2)
    ]
    v = torch.randn_like(q)
    g = torch.full(shape, -0.2, device="cuda", dtype=torch.float32)
    beta = torch.full(shape[:2], 0.5, device="cuda", dtype=torch.bfloat16)
    inputs = [t.requires_grad_() for t in (q, k, v, g, beta)]
    cu = torch.tensor([0, 65], device="cuda", dtype=torch.int32)
    chunks = torch.tensor([[0, 0], [0, 1]], device="cuda", dtype=torch.int32)
    saved = []
    versions = [t._version for t in (*inputs, cu, chunks)]

    def pack(t):
        saved.append(t)
        return t

    with use_implementation("kimi_delta_attention", provider):
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = mlops.kimi_delta_attention(*inputs, cu, chunks)
        assert all(t.data_ptr() != g.data_ptr() for t in saved)
        matrices = [t for t in saved if t.ndim == 3 and t.shape[-1] == 64]
        assert len(matrices) == (0 if provider.endswith(".recompute") else 2)
        dy = torch.randn_like(out)
        first = torch.autograd.grad(out, inputs, dy, retain_graph=True)
        second = torch.autograd.grad(out, inputs, dy)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert versions == [t._version for t in (*inputs, cu, chunks)]
        from mlops.providers.cula import kda, kda_native

        forward_op = (
            kda_native._forward if provider.endswith(".recompute") else kda._forward
        )
        torch.library.opcheck(
            forward_op,
            (*inputs, cu, chunks, 128**-0.5),
        )
