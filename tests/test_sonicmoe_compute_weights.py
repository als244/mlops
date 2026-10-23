"""CPU tests for the proposed physical-BF16/logical-FP32 operator boundary."""

import pytest
import torch
from mlops.expert_parallel.parameters import BF16ComputeWeight
from mlops.providers.sonicmoe import bf16, fp8
from mlops.providers.sonicmoe.lora import _aligned_factors, interleaved_rows


@torch.library.custom_op("mlops_bf16_boundary_test::value", mutates_args=())
def value(weight: torch.Tensor, components: list[torch.Tensor]) -> torch.Tensor:
    assert weight.dtype == torch.bfloat16
    return weight.sum() + sum(c.sum() for c in components)


@value.register_fake
def fake(weight, components):
    return torch.empty((), dtype=torch.bfloat16, device=weight.device)


def setup(ctx, inputs, output):
    ctx.shape = inputs[0].shape
    ctx.count = len(inputs[1])


def backward(ctx, dy):
    return torch.full(ctx.shape, 1.001, dtype=torch.float32) * dy, [None] * ctx.count


value.register_autograd(backward, setup_context=setup)
fp8.register_container_dispatch(value)


def parameter(shape):
    return torch.nn.Parameter(
        BF16ComputeWeight(torch.ones(shape, dtype=torch.bfloat16))
    )


def test_logical_boundary_preserves_fp32_gradient_and_existing_components():
    w = parameter((4, 4))
    output = value(w, [torch.tensor(3.0)])
    assert output.item() == 19
    output.backward()
    torch.testing.assert_close(w.grad, torch.full((4, 4), 1.001))
    assert w._data.dtype == torch.bfloat16


@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_factor_scale_and_interleave_preserve_gradient_precision(scale):
    w = parameter((2, 8, 4))
    transformed = bf16.scaled(interleaved_rows(w), scale)
    value(transformed, []).backward()
    torch.testing.assert_close(w.grad, torch.full((2, 8, 4), 1.001) * scale)
    assert transformed._data.dtype == torch.bfloat16


def test_rank_four_padding_keeps_logical_gradient_dtype():
    a, b = parameter((2, 4, 16)), parameter((2, 16, 4))
    pa, pb = _aligned_factors(a, b)
    assert pa.shape == (2, 8, 16) and pb.shape == (2, 16, 8)
    (value(pa, []) + value(pb, [])).backward()
    for weight in (a, b):
        torch.testing.assert_close(weight.grad, torch.full(weight.shape, 1.001))


def test_bf16_router_storage_preserves_fp32_logit_and_parameter_gradient():
    from mlops.kernels.router import routed_logits

    x = torch.randn(7, 16, dtype=torch.bfloat16, requires_grad=True)
    w = torch.nn.Parameter(BF16ComputeWeight(torch.randn(3, 16, dtype=torch.bfloat16)))
    dy = torch.randn(7, 3)
    output = routed_logits(x, w, weight_grad_dtype=torch.float32)
    assert output.dtype == torch.float32
    output.backward(dy)
    torch.testing.assert_close(output, x.detach().float() @ w._data.float().t())
    torch.testing.assert_close(w.grad, dy.t() @ x.detach().float())
    assert w.grad.dtype == torch.float32 and w._data.dtype == torch.bfloat16
    torch.testing.assert_close(x.grad, (dy @ w._data.float()).bfloat16())
