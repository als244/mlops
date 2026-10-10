import pytest
import torch

from mlops.glm.kda import kimi_delta_attention


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
    errors = []
    for actual, target in zip((got, *actual_grad), (expected, *reference_grad)):
        error = (
            actual.float() - target.float()
        ).norm() / target.float().norm().clamp_min(1e-8)
        errors.append(float(error.detach()))
        assert error < 0.02, errors
    print("KDA output/dQ/dK/dV/dGate/dBeta relative L2:", errors, flush=True)


pytestmark = pytest.mark.gpu
