"""Clipping boundaries and LoRA's local derivative, without communication."""

import pytest
import torch
from torch.nn import functional as F

pytest.importorskip("quack")

from mlops.expert_parallel.quack.lora.math import _down_derivative
from mlops.expert_parallel.quack.shared_expert import (
    frozen_shared_backward,
    shared_forward,
)


@pytest.mark.parametrize("limit", [None, 0.25, 10.0])
def test_frozen_shared_expert_clamps_and_keeps_input_gradients(limit):
    torch.manual_seed(5)
    x = torch.randn(11, 8, requires_grad=True)
    gate_weight, up_weight, down_weight = (
        torch.randn(shape) for shape in ((12, 8), (12, 8), (8, 12))
    )
    gate, up = F.linear(x, gate_weight), F.linear(x, up_weight)
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    expected = F.linear(F.silu(gate) * up, down_weight)
    actual, *state = shared_forward(x, gate_weight, up_weight, down_weight, limit)
    dy = torch.randn_like(actual)
    (dx,) = frozen_shared_backward(
        dy, x, *state, gate_weight, up_weight, down_weight, limit
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(dx, torch.autograd.grad(expected, x, dy)[0])


@pytest.mark.parametrize("limit", [None, 1.5, 10.0])
def test_routed_derivatives_match_autograd_at_and_outside_boundaries(limit):
    torch.manual_seed(7)
    edge = limit or 1.5
    preact = (torch.randn(13, 16) * edge * 2).to(torch.bfloat16).float()
    preact[0, :8] = torch.tensor(
        [edge, edge, -edge, -edge, 2 * edge, 2 * edge, -2 * edge, -2 * edge]
    )
    preact.requires_grad_()
    probabilities = torch.rand(13, requires_grad=True)
    gradient = torch.randn(13, 8).to(torch.bfloat16)
    gate, up = preact[:, 0::2], preact[:, 1::2]
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    activation = F.silu(gate) * up
    expected = torch.autograd.grad(
        activation * probabilities[:, None], (preact, probabilities), gradient.float()
    )
    actual, scaled, score = _down_derivative(preact, gradient, probabilities, limit)
    torch.testing.assert_close(actual, expected[0].to(torch.bfloat16))
    torch.testing.assert_close(score, expected[1])
    torch.testing.assert_close(
        scaled, (activation * probabilities[:, None]).to(torch.bfloat16)
    )
    if limit is not None:
        assert actual[0, 4:6].count_nonzero() == 0
        assert actual[0, 6] != 0  # The gate has no lower clamp.
        assert actual[0, 7] == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_routed_derivatives_match_compiled_autograd():
    from mlops.expert_parallel.quack.lora.math import compiled

    torch.manual_seed(9)
    preact = (torch.randn(128, 256, device="cuda") * 3).to(torch.bfloat16).float()
    preact.requires_grad_()
    probabilities = torch.rand(128, device="cuda", requires_grad=True)
    gradient = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)

    def reference(value, weights):
        gate, up = value[:, 0::2], value[:, 1::2]
        activation = F.silu(gate.clamp(max=1.5)) * up.clamp(-1.5, 1.5)
        return activation * weights[:, None]

    expected = torch.autograd.grad(
        torch.compile(reference, fullgraph=True)(preact, probabilities),
        (preact, probabilities),
        gradient.float(),
    )
    actual, _, score = compiled(_down_derivative)(preact, gradient, probabilities, 1.5)
    torch.testing.assert_close(actual, expected[0].to(torch.bfloat16))
    torch.testing.assert_close(score, expected[1], rtol=1e-5, atol=1e-5)
