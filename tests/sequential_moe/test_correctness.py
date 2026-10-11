"""Sequential experts versus independent PyTorch math with identical routes."""

import pytest
import torch
from torch.nn import functional as F

from mlops.sequential_moe import (
    SequentialMoE,
    SequentialMoEConfig,
    SequentialMoELoRA,
)
from mlops.sequential_moe.precision.parameters import payload

FACTORS = ("gate_up_a", "gate_up_b", "down_a", "down_b")
pytestmark = pytest.mark.gpu


def relative_error(actual, expected):
    return (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-8)


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("skewed", [False, True])
def test_outputs_and_gradients_with_empty_and_multichunk_experts(lora, dtype, skewed):
    if not torch.cuda.is_available():
        pytest.skip("Sequential experts require CUDA")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 arithmetic is unavailable")
    torch.manual_seed(601)
    config = SequentialMoEConfig(
        64,
        96,
        4,
        2,
        rank=8,
        alpha=5,
        chunk_capacity=256,
        row_multiple=256,
        activation_dtype=dtype,
        factor_dtype=dtype,
        gradient_dtype=torch.float32,
        weight_storage="dense",
        gemm_precision="dense",
    )
    model = (SequentialMoELoRA if lora else SequentialMoE)(config)
    if lora:
        with torch.no_grad():
            for expert in model.experts:
                for name in FACTORS:
                    payload(getattr(expert, name)).normal_(0, 0.04)
    x = torch.randn(613, 64, device="cuda", dtype=dtype, requires_grad=True)
    xr = x.detach().float().requires_grad_()
    first = torch.arange(len(x), device="cuda") % (1 if skewed else 4)
    ids = torch.stack((first, (first + 1) % 4), -1)
    probs = torch.randn(len(x), 2, device="cuda").softmax(-1).requires_grad_()
    pr = probs.detach().clone().requires_grad_()
    initial = (torch.randn_like(x) * 0.1).requires_grad_()
    initial_before = initial.detach().clone()
    initial_ref = initial.detach().float().requires_grad_()
    expected = initial_ref
    snapshots = []
    for i, expert in enumerate(model.experts):
        values = [
            payload(getattr(expert, name))
            .detach()
            .float()
            .requires_grad_(name in FACTORS if lora else True)
            for name in (("gate_up", "down", *FACTORS) if lora else ("gate_up", "down"))
        ]
        snapshots.append(values)
        rows, slots = torch.where(ids == i)
        w13, w2, *factors = values
        gu = F.linear(xr[rows], w13)
        if factors:
            a13, b13, a2, b2 = factors
            gu = gu + config.scale * F.linear(F.linear(xr[rows], a13), b13)
        gate, up = gu.chunk(2, -1)
        hidden = F.silu(gate) * up
        y = F.linear(hidden, w2)
        if factors:
            y = y + config.scale * F.linear(F.linear(hidden, a2), b2)
        expected = expected.index_add(0, rows, y * pr[rows, slots, None])
    actual = model.forward_routed(x, ids, probs, accumulator=initial)
    torch.testing.assert_close(initial, initial_before, rtol=0, atol=0)
    dy = torch.randn_like(actual)
    actual.backward(dy)
    expected.backward(dy.float())
    pairs = [(actual, expected), (x.grad, xr.grad), (probs.grad, pr.grad)]
    torch.testing.assert_close(initial.grad.float(), initial_ref.grad, rtol=0, atol=0)
    for expert, snapshot in zip(model.experts, snapshots, strict=True):
        for name, value in zip(
            ("gate_up", "down", *FACTORS) if lora else ("gate_up", "down"),
            snapshot,
            strict=True,
        ):
            parameter = getattr(expert, name)
            if not value.requires_grad:
                assert parameter.grad is None
            else:
                assert parameter.grad.dtype == torch.float32
                pairs.append((parameter.grad, value.grad))
    errors = [float(relative_error(a, b).detach()) for a, b in pairs]
    assert max(errors) < (0.04 if dtype == torch.bfloat16 else 0.009), errors
