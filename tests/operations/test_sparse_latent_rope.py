"""Positional scores affect sparse attention, with gradients to all four inputs."""

import pytest
import torch

from mlops import explicit, sparse_latent_attention


def reference(q, kv, qp, kp, ids, scale):
    tokens = q.shape[0]
    selected = torch.zeros(tokens, tokens + 1, device=q.device, dtype=torch.bool)
    selected.scatter_(1, torch.where(ids >= 0, ids, tokens).long(), True)
    logits = torch.einsum("thd,sd->hts", q.float(), kv.float())
    logits += torch.einsum("thr,sr->hts", qp.float(), kp.float())
    probabilities = (
        (logits * scale).masked_fill(~selected[:, :-1], -torch.inf).softmax(-1)
    )
    return torch.einsum("hts,sd->thd", probabilities, kv.float()).to(q.dtype)


@pytest.mark.parametrize("heads,width", [(16, 64), (64, 512)])
@pytest.mark.parametrize("compiled", [False, True])
def test_positional_tail_attention_and_all_gradients(heads, width, compiled):
    torch.manual_seed(114)
    tokens = 47
    shapes = (
        (tokens, heads, width),
        (tokens, width),
        (tokens, heads, 64),
        (tokens, 64),
    )
    values = tuple(
        torch.randn(s, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for s in shapes
    )
    # Absorbing the key projection naturally produces this noncontiguous Q.
    values = (values[0].transpose(0, 1).contiguous().transpose(0, 1), *values[1:])
    expected_inputs = tuple(v.detach().clone().requires_grad_() for v in values)
    q, kv, qp, kp = values
    ids = torch.full((tokens, 33), -1, device="cuda", dtype=torch.int32)
    for row in range(tokens):
        start = 0 if row < 19 else 19
        selected = torch.arange(start, row + 1, 2, device="cuda")
        ids[row, : selected.numel()] = selected.to(torch.int32)
    fn = (
        torch.compile(sparse_latent_attention, fullgraph=True)
        if compiled
        else sparse_latent_attention
    )
    actual = fn(q, kv, ids, scale=256**-0.5, q_rope=qp, k_rope=kp)
    expected = reference(*expected_inputs, ids, 256**-0.5)
    dy = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, values, dy)
    expected_gradients = torch.autograd.grad(expected, expected_inputs, dy)
    errors = []
    for got, target in zip(
        (actual, *gradients), (expected, *expected_gradients), strict=True
    ):
        error = float(
            ((got.float() - target.float()).norm() / target.float().norm()).detach()
        )
        errors.append(error)
        assert error < 0.012, errors
    if not compiled:
        out, lse = explicit.sparse_latent_attention.forward(
            q, kv, ids, scale=256**-0.5, q_rope=qp, k_rope=kp
        )
        vjp = explicit.sparse_latent_attention.backward(
            dy, q, kv, ids, out, lse, scale=256**-0.5, q_rope=qp, k_rope=kp
        )
        torch.testing.assert_close(out, actual, rtol=0, atol=0)
        for got, target in zip(vjp, gradients, strict=True):
            torch.testing.assert_close(got, target, rtol=0.01, atol=0.01)


pytestmark = pytest.mark.gpu
