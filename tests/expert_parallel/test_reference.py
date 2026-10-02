"""The fixed-route reference has an independent per-token equation check."""

import torch
from torch.nn import functional as F

from mlops.expert_parallel.reference import (
    expert_computation,
    route,
    router_logits,
    routing_probabilities,
)


def test_fixed_routes_outputs_and_all_gradients():
    torch.manual_seed(17)
    # Includes a globally empty expert and deliberately non-top-k route order.
    x = torch.randn(4, 8, dtype=torch.bfloat16).requires_grad_()
    p = torch.rand(4, 2).requires_grad_()
    weights = [
        torch.randn(*shape).requires_grad_()
        for shape in ((3, 4, 8), (3, 4, 8), (3, 8, 4))
    ]
    ids = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.int32)
    output = expert_computation(x, ids, p, *weights)
    expected = []
    for token in range(x.shape[0]):
        terms = []
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            gate = F.linear(x[token : token + 1], weights[0][expert].bfloat16())
            up = F.linear(x[token : token + 1], weights[1][expert].bfloat16())
            hidden = (F.silu(gate.float()) * up.float()).bfloat16()
            raw = F.linear(hidden, weights[2][expert].bfloat16())
            terms.append((raw.float() * p[token, slot]).bfloat16().float())
        expected.append(sum(terms).bfloat16())
    expected = torch.cat(expected)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    dy = torch.randn_like(output)
    actual_grad = torch.autograd.grad(output, (x, p, *weights), dy)
    expected_grad = torch.autograd.grad(expected, (x, p, *weights), dy)
    for actual, reference in zip(actual_grad, expected_grad):
        # Separate BF16 matmul reductions can round at different additions.
        torch.testing.assert_close(actual, reference, atol=0.125, rtol=0.02)
    assert all(torch.count_nonzero(g[2]) == 0 for g in actual_grad[2:])


def test_router_can_be_checked_without_running_experts():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16)
    w = torch.tensor([[4.0, 1.0], [2.0, 4.0], [0.0, 2.0]], requires_grad=True)
    ids, p = route(x, w, top_k=2)
    assert ids.tolist() == [[0, 1], [1, 2]]
    torch.testing.assert_close(p.sum(-1), torch.ones(2))
    logits = router_logits(x, w)
    full = routing_probabilities(logits, ids, renormalize_topk=False)
    torch.testing.assert_close(full, logits.softmax(-1).gather(-1, ids.long()))
    assert bool((full.sum(-1) < 1).all())
    permuted = ids.flip(-1)
    torch.testing.assert_close(routing_probabilities(logits, permuted), p.flip(-1))


def test_reference_never_changes_supplied_routes():
    x = torch.ones(2, 4, dtype=torch.bfloat16)
    gate = up = torch.ones(3, 2, 4)
    down = torch.arange(1, 4).float()[:, None, None].expand(3, 4, 2)
    ids = torch.tensor([[2], [0]], dtype=torch.int32)
    p = torch.tensor([[0.5], [1.0]])
    saved_ids, saved_p = ids.clone(), p.clone()
    y = expert_computation(x, ids, p, gate, up, down)
    assert torch.equal(ids, saved_ids) and torch.equal(p, saved_p)
    torch.testing.assert_close(
        y[0].float() / y[1].float(), torch.full((4,), 1.5), atol=0.01, rtol=0
    )
