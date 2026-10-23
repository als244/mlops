"""KDA default/IEEE choices stay explicit in dispatch and captured operator IDs."""

import pytest
import torch

from mlops import kimi_delta_attention
from mlops.dispatch import explain_implementation, use_implementation

pytestmark = pytest.mark.gpu


def test_fast_default_and_explicit_ieee_have_separate_ops():
    import mlops

    torch.manual_seed(89)
    q = mlops.l2_norm(
        torch.randn(129, 2, 128, device="cuda", dtype=torch.bfloat16)
    ).requires_grad_()
    k = mlops.l2_norm(torch.randn_like(q)).requires_grad_()
    v = torch.randn_like(q).requires_grad_()
    g = (
        torch.empty(129, 2, 128, device="cuda", dtype=torch.float32)
        .uniform_(-5, -0.01)
        .requires_grad_()
    )
    beta = torch.rand(129, 2, device="cuda", dtype=torch.bfloat16).requires_grad_()
    cu = torch.tensor([0, 64, 129], device="cuda", dtype=torch.int64)
    chunks = torch.tensor([[0, 0], [1, 0], [1, 1]], device="cuda", dtype=torch.int64)
    args = (q, k, v, g, beta, cu, chunks)
    assert (
        explain_implementation("kimi_delta_attention", *args).selected
        != "fla.kimi_delta_attention.chunk.ieee"
    )
    dy = torch.randn_like(v)
    outputs = []
    for implementation in [
        "fla.kimi_delta_attention.chunk",
        "fla.kimi_delta_attention.chunk.ieee",
    ]:
        with use_implementation("kimi_delta_attention", implementation):
            out = kimi_delta_attention(*args)
            grads = torch.autograd.grad(out, args[:5], grad_outputs=dy)
            assert all(torch.isfinite(t).all() for t in (out, *grads))
            outputs.append(out)
            # A separate custom-op identity ensures compiler caches and manifests
            # distinguish the opt-in math policy.
            graph = torch.fx.experimental.proxy_tensor.make_fx(kimi_delta_attention)(
                *args
            )
            expected = "fla_ieee_fwd" if implementation.endswith(".ieee") else "fla_fwd"
            assert expected in graph.code
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("lengths", [(79,), (31, 67)])
def test_fast_kda_matches_upstream_fla_at_strong_decay(lengths):
    from fla.ops.kda import chunk_kda

    import mlops

    torch.manual_seed(812)
    shape = (sum(lengths), 2, 64)
    inputs = [
        mlops.l2_norm(torch.randn(shape, device="cuda", dtype=torch.bfloat16))
        for _ in range(2)
    ] + [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16),
        torch.empty(shape, device="cuda").uniform_(-3.5, -1.5),
        torch.rand(shape[:2], device="cuda", dtype=torch.bfloat16),
    ]
    inputs = [x.requires_grad_() for x in inputs]
    refs = [x.detach().clone().requires_grad_() for x in inputs]
    ends = [0]
    for n in lengths:
        ends.append(ends[-1] + n)
    cu = torch.tensor(ends, device="cuda", dtype=torch.int64)
    chunks = torch.tensor(
        [(i, j) for i, n in enumerate(lengths) for j in range((n + 63) // 64)],
        device="cuda",
        dtype=torch.int64,
    )
    with use_implementation("kimi_delta_attention", "fla.kimi_delta_attention.chunk"):
        out = kimi_delta_attention(*inputs, cu, chunks)
    target = chunk_kda(
        *(x.unsqueeze(0) for x in refs),
        cu_seqlens=cu,
        safe_gate=True,
        lower_bound=-5.0,
    )[0].squeeze(0)
    dy = torch.randn_like(out)
    gradients = torch.autograd.grad(out, inputs, dy)
    expected = torch.autograd.grad(target, refs, dy)
    errors = [
        ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)).item()
        for a, b in zip((out, *gradients), (target, *expected))
    ]
    assert max(errors) < 0.02, errors
