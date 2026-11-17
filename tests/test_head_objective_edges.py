"""Fused formula corner cases and tensor-valued controls under compilation."""

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch.nn import functional as F

from mlops import head_objective


def reference(logits, targets, fields, controls):
    current = -F.cross_entropy(logits, targets, reduction="none")
    mask, weights, old, advantages, reference = fields
    current, weights, old, advantages, reference = (
        torch.where(mask, value, 0.0)
        for value in (current, weights, old, advantages, reference)
    )
    clip, beta = controls
    delta = current - old
    # Equivalent surrogate bounds applied before exp avoid evaluating an
    # overflowing branch that clipping would discard.
    bounded = torch.where(
        advantages >= 0,
        delta.clamp_max(torch.log1p(clip)),
        delta.clamp_min(torch.log1p(-clip)),
    )
    difference = torch.where(beta != 0, reference - current, 0.0)
    return (
        weights
        * (-bounded.exp() * advantages + beta * (difference.expm1() - difference))
    ).sum()


def case(device):
    torch.manual_seed(708)
    rows, vocab = 13, 33
    logits = (torch.randn(rows, vocab, device=device) * 0.1).requires_grad_()
    targets = torch.arange(rows, device=device)
    current = -F.cross_entropy(logits.detach(), targets, reduction="none")
    difference = torch.tensor(
        [-1, -0.1, 0.05, 1, -1, -0.1, 0.05, 1, 100, -100, 0, 0.1, 0.3], device=device
    )
    advantages = torch.tensor(
        [1.0] * 4 + [-1.0] * 4 + [1.0, -1.0, 0.0, 1.0, -1.0], device=device
    )
    weights = torch.tensor(
        [1.0, -0.3, 0.4, 0.8, 1.2, 0.7, -0.6, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        device=device,
    )
    mask = torch.ones(rows, dtype=torch.bool, device=device)
    mask[-1] = False
    fields = [mask, weights, current - difference, advantages, current + 0.1]
    for field in fields[1:]:
        field[-1] = float("nan")
    return logits, targets, fields


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_policy_reuses_graph_with_changed_controls_and_masked_nan_fields():
    torch._dynamo.reset()
    logits, targets, fields = case("cuda")
    weight = torch.eye(logits.shape[1], device="cuda")
    controls = [torch.tensor(0.2, device="cuda"), torch.tensor(0.0, device="cuda")]

    def run(logits, fields, controls):
        return head_objective(
            logits,
            weight,
            targets,
            objective="clipped_policy",
            inputs=fields,
            controls=controls,
            chunk_size=4,
        )[0]

    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(run, backend=counter, fullgraph=True)
    oracle = torch.compile(reference, fullgraph=True)
    for clip, beta in ((0.2, 0.0), (0.17, 0.2), (0.63, 0.03), (0.0, 0.0)):
        controls[0].fill_(clip)
        controls[1].fill_(beta)
        actual = compiled(logits, fields, controls)
        expected = oracle(logits, targets, fields, controls)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
        actual_grad = torch.autograd.grad(actual, logits)[0]
        expected_grad = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=3e-6)
        assert torch.count_nonzero(actual_grad[-1]) == 0
    assert counter.frame_count == 1

    # This change is runtime tensor data too. A disabled penalty must not
    # evaluate the unused exponential, while an enabled overflow stays visible.
    fields[-1].fill_(1000.0)
    assert torch.isfinite(compiled(logits, fields, controls))
    fields[0].fill_(False)
    fields[0][0] = True
    controls[1].fill_(0.1)
    assert torch.isinf(compiled(logits, fields, controls))
    assert counter.frame_count == 1


@pytest.mark.parametrize("size", [0, 13])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_empty_or_fully_masked_objective_has_zero_gradients(size, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.zeros(size, 7, device=device, requires_grad=True)
    w = torch.zeros(11, 7, device=device, requires_grad=True)
    targets = torch.zeros(size, dtype=torch.int64, device=device)
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    data = torch.full((size,), float("nan"), device=device)
    for name, fields, controls in (
        ("cross_entropy", [mask, data], []),
        (
            "clipped_policy",
            [mask, data, data, data, data],
            [torch.tensor(0.2, device=device), torch.tensor(0.1, device=device)],
        ),
    ):
        loss, _ = head_objective(
            x,
            w,
            targets,
            objective=name,
            inputs=fields,
            controls=controls,
            chunk_size=3,
        )
        assert loss.item() == 0
        gradients = torch.autograd.grad(loss, (x, w))
        assert all(
            torch.isfinite(value).all() and torch.count_nonzero(value) == 0
            for value in gradients
        )
