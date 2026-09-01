"""Destination-storage contracts for the optimized packed SwiGLU VJP."""

from __future__ import annotations

import pytest
import torch

from mlops.kernels.swiglu import (
    swiglu_packed_backward,
    swiglu_packed_backward_out,
)
from mlops.providers.builtin.moe_composed import (
    _finish_backward,
    _finish_backward_donate_h13_op,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_packed_backward_destination_may_reuse_dead_source(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    generator = torch.Generator(device=device).manual_seed(71)
    packed = torch.randn(
        17,
        64,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    grad = torch.randn(
        17,
        32,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    expected = swiglu_packed_backward(
        grad,
        packed,
        variant="triton",
    )
    donated = packed.clone()
    actual = swiglu_packed_backward_out(
        grad,
        donated,
        donated,
        variant="triton",
    )
    assert actual.data_ptr() == donated.data_ptr()
    assert torch.equal(actual, expected)


def test_packed_backward_destination_rejects_layout_mismatch():
    packed = torch.randn(4, 16)
    grad = torch.randn(4, 8)
    destination = torch.empty(4, 17)[:, :16]
    with pytest.raises(ValueError, match="destination must match"):
        swiglu_packed_backward_out(
            grad,
            packed,
            destination,
            variant="triton",
        )


@pytest.mark.gpu
def test_planning_moe_vjp_donates_only_the_declared_activation():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(19)
    tokens, hidden, top_k, experts, expert_width = 8, 32, 2, 4, 32
    assignments = tokens * top_k
    grad_output = torch.randn(
        tokens, hidden, device="cuda", dtype=torch.bfloat16
    )
    h13 = torch.randn(
        assignments, 2 * expert_width, device="cuda", dtype=torch.bfloat16
    )
    w2 = torch.randn(
        experts, expert_width, hidden, device="cuda", dtype=torch.bfloat16
    )
    route_weights = torch.rand(tokens, top_k, device="cuda")
    order = torch.arange(assignments, device="cuda", dtype=torch.int32)
    offsets = torch.arange(
        0,
        assignments + 1,
        assignments // experts,
        device="cuda",
        dtype=torch.int32,
    )
    slots = order.reshape(tokens, top_k)

    expected = _finish_backward(
        grad_output,
        h13,
        w2,
        route_weights,
        order,
        offsets,
        slots,
        top_k=top_k,
    )
    donated_h13 = h13.clone()
    actual_tail = _finish_backward_donate_h13_op(
        grad_output,
        donated_h13,
        w2,
        route_weights,
        order,
        offsets,
        slots,
        top_k,
    )
    assert torch.equal(donated_h13, expected[0])
    for actual, reference in zip(actual_tail, expected[1:], strict=True):
        assert torch.equal(actual, reference)
