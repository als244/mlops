import gc
import weakref

import pytest
import torch

from mlops import explicit, kimi_delta_attention


def recurrence(q, k, v, g, beta, lengths):
    output = []
    start = 0
    for length in lengths:
        state = torch.zeros(
            q.shape[1], q.shape[2], v.shape[2], device=q.device, dtype=torch.float32
        )
        for t in range(start, start + length):
            state = state * g[t].float().exp().unsqueeze(-1)
            correction = (
                v[t].float() - (state * k[t].float().unsqueeze(-1)).sum(-2)
            ) * beta[t].float().unsqueeze(-1)
            state = state + k[t].float().unsqueeze(-1) * correction.unsqueeze(-2)
            output.append(
                (state * (q[t].float() * q.shape[-1] ** -0.5).unsqueeze(-1)).sum(-2)
            )
        start += length
    return torch.stack(output).to(v.dtype)


@pytest.mark.parametrize("lengths", [(79,), (31, 67)])
@pytest.mark.parametrize("compiled", [False, True])
def test_kda_values_gradients(lengths, compiled):
    torch.manual_seed(41)
    shape = (sum(lengths), 2, 64)
    values = [
        torch.nn.functional.normalize(
            torch.randn(shape, device="cuda"), dim=-1
        ).bfloat16()
        for _ in range(2)
    ]
    values += [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16),
        -torch.rand(shape, device="cuda") * 0.3 - 0.01,
        torch.rand(shape[:2], device="cuda", dtype=torch.bfloat16),
    ]
    # Exercise caller-owned strided inputs in the explicit VJP too.
    if len(lengths) > 1:
        values = [torch.stack((x, x), dim=-1)[..., 0] for x in values]
        assert all(not x.is_contiguous() for x in values)
    source = [x.requires_grad_() for x in values]
    reference = [x.detach().clone().requires_grad_() for x in values]
    ends = [0]
    for n in lengths:
        ends.append(ends[-1] + n)
    cumulative = torch.tensor(ends, device="cuda", dtype=torch.int32)
    chunks = torch.tensor(
        [(i, j) for i, n in enumerate(lengths) for j in range((n + 63) // 64)],
        device="cuda",
        dtype=torch.int32,
    )
    fn = (
        torch.compile(kimi_delta_attention, fullgraph=True)
        if compiled
        else kimi_delta_attention
    )
    got = fn(*source, cumulative, chunks)
    expected = recurrence(*reference, lengths)
    dy = torch.randn_like(got)
    actual_grad = torch.autograd.grad(got, source, dy)
    reference_grad = torch.autograd.grad(expected, reference, dy)
    if not compiled:
        output, gates, aqk, akk = explicit.kimi_delta_attention.forward(
            *source, cumulative, chunks
        )
        vjp = explicit.kimi_delta_attention.backward(
            dy, *source, gates, aqk, akk, cumulative, chunks
        )
        torch.testing.assert_close(output, got, rtol=0, atol=0)
        for actual, reference_vjp in zip(vjp, actual_grad):
            torch.testing.assert_close(actual, reference_vjp, rtol=0, atol=0)
        assert not any(t.requires_grad for t in (output, gates, aqk, akk, *vjp))
    errors = []
    for actual, target in zip((got, *actual_grad), (expected, *reference_grad)):
        error = (
            actual.float() - target.float()
        ).norm() / target.float().norm().clamp_min(1e-8)
        errors.append(float(error.detach()))
        assert error < 0.02, errors
    print("KDA output/dQ/dK/dV/dGate/dBeta relative L2:", errors, flush=True)


pytestmark = pytest.mark.gpu


def test_kda_does_not_retain_sequence_metadata():
    def invoke():
        shape = (5, 2, 64)
        q, k, v = (
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
            for _ in range(3)
        )
        gates = -torch.ones(shape, device="cuda") * 0.2
        beta = torch.ones(shape[:2], device="cuda", dtype=torch.bfloat16) * 0.5
        cumulative = torch.tensor([0, 5], device="cuda", dtype=torch.int64)
        chunks = torch.tensor([[0, 0]], device="cuda", dtype=torch.int64)
        references = weakref.ref(cumulative), weakref.ref(chunks)
        with torch.no_grad():
            output = kimi_delta_attention(q, k, v, gates, beta, cumulative, chunks)
            assert torch.isfinite(output).all()
        return references

    references = invoke()
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("entries", [1, 9, 1024, 1025])
def test_kda_offsets_read_current_metadata(dtype, entries):
    from mlops.kernels.kda.state import _offsets

    # Reuse one strided input with different contents. Identity-based caching
    # would return stale offsets on the second invocation.
    storage = torch.empty(2 * entries, device="cuda", dtype=dtype)
    cumulative = storage[::2]
    for shift in (0, 3):
        lengths = [(0, 1, 63, 64, 65, 129)[(i + shift) % 6] for i in range(entries - 1)]
        ends, expected = [11], [0]
        for length in lengths:
            ends.append(ends[-1] + length)
            expected.append(expected[-1] + (length + 63) // 64)
        cumulative.copy_(torch.tensor(ends, device="cuda", dtype=dtype))
        actual = _offsets(cumulative)
        torch.testing.assert_close(
            actual,
            torch.tensor(expected, device="cuda", dtype=torch.int64),
            rtol=0,
            atol=0,
        )
