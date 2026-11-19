"""Clipping boundaries and LoRA's local derivative, without communication."""

import pytest
import torch
from torch.nn import functional as F

pytest.importorskip("quack")

from mlops.expert_parallel.quack.lora.math import down_derivative
from mlops.expert_parallel.quack.shared_expert import (
    frozen_shared_backward,
    shared_forward,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]
BF16 = {"rtol": 2e-2, "atol": 2e-2}


@pytest.mark.parametrize("limit", [None, 0.25, 10.0])
def test_frozen_shared_expert_clamps_and_keeps_input_gradients(limit):
    torch.manual_seed(5)
    x, gate_weight, up_weight, down_weight = (
        torch.randn(shape, device="cuda").to(torch.bfloat16)
        for shape in ((11, 8), (12, 8), (12, 8), (8, 12))
    )
    leaf = x.float().requires_grad_()
    gate = F.linear(leaf, gate_weight.float()).to(torch.bfloat16).float()
    up = F.linear(leaf, up_weight.float()).to(torch.bfloat16).float()
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    expected = F.linear(F.silu(gate) * up, down_weight.float())
    actual, *state = shared_forward(x, gate_weight, up_weight, down_weight, limit)
    dy = torch.randn_like(actual)
    (dx,) = frozen_shared_backward(
        dy, x, *state, gate_weight, up_weight, down_weight, limit
    )
    torch.testing.assert_close(actual.float(), expected, **BF16)
    (expected_dx,) = torch.autograd.grad(expected, leaf, dy.float())
    # The activation and its gradient are rounded to BF16 before the
    # projections, as the operator rounds them; that is judged at the
    # gradient's scale, which a wrong clip would exceed many times over.
    scale = expected_dx.abs().max().item()
    torch.testing.assert_close(dx.float(), expected_dx, rtol=0, atol=2e-2 * scale)


@pytest.mark.parametrize("limit", [None, 0.25, 1.5, 10.0])
def test_routed_derivatives_match_autograd_at_and_outside_boundaries(limit):
    torch.manual_seed(7)
    edge = limit or 1.5
    preact = (torch.randn(13, 16, device="cuda") * edge * 2).to(torch.bfloat16)
    preact[0, :8] = torch.tensor(
        [edge, edge, -edge, -edge, 2 * edge, 2 * edge, -2 * edge, -2 * edge],
        device="cuda",
    )
    leaf = preact.float().requires_grad_()
    probabilities = torch.rand(13, device="cuda", requires_grad=True)
    gradient = torch.randn(13, 8, device="cuda").to(torch.bfloat16)
    gate, up = leaf[:, 0::2], leaf[:, 1::2]
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    activation = F.silu(gate) * up
    expected = torch.autograd.grad(
        activation * probabilities[:, None], (leaf, probabilities), gradient.float()
    )
    actual, scaled, score = down_derivative(
        preact, gradient, probabilities.detach(), limit
    )
    torch.testing.assert_close(actual, expected[0].to(torch.bfloat16))
    torch.testing.assert_close(score, expected[1])
    torch.testing.assert_close(
        scaled, (activation * probabilities[:, None]).detach().to(torch.bfloat16)
    )
    if limit is not None:
        assert actual[0, 4:6].count_nonzero() == 0
        assert actual[0, 6] != 0  # The gate has no lower clamp.
        assert actual[0, 7] == 0


@pytest.mark.parametrize("limit", [0.25, 1.5, 10.0])
def test_routed_row_kernel_matches_compiled_autograd(limit):
    torch.manual_seed(9)
    preact = (torch.randn(128, 256, device="cuda") * 2 * limit).to(torch.bfloat16)
    preact[0, :8] = torch.tensor(
        [limit, limit, -limit, -limit, 2 * limit, 2 * limit, -2 * limit, -2 * limit],
        device="cuda",
    )
    leaf = preact.float().requires_grad_()
    probabilities = torch.rand(128, device="cuda", requires_grad=True)
    gradient = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)

    def reference(value, weights):
        gate, up = value[:, 0::2], value[:, 1::2]
        activation = F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)
        return activation * weights[:, None]

    expected = torch.autograd.grad(
        torch.compile(reference, fullgraph=True, dynamic=False)(leaf, probabilities),
        (leaf, probabilities),
        gradient.float(),
    )
    actual, _, score = down_derivative(preact, gradient, probabilities.detach(), limit)
    torch.testing.assert_close(actual, expected[0].to(torch.bfloat16))
    # The row kernel sums <r, a> in its own order: FP32 rounding only.
    torch.testing.assert_close(score, expected[1], rtol=1e-4, atol=1e-4)
