"""Packed routers must remain trainable when exported and differentiated."""

import pytest
import torch
from torch.export.graph_signature import InputKind

from mlops.expert_parallel.parameters import BF16ComputeWeight
from mlops.kernels.router import routed_logits


class Router(torch.nn.Module):
    def __init__(self, packed):
        super().__init__()
        weight = torch.randn(3, 16, dtype=torch.bfloat16)
        self.weight = torch.nn.Parameter(
            BF16ComputeWeight(weight) if packed else weight
        )
        self.bias = torch.nn.Parameter(torch.randn(3))

    def forward(self, x):
        return routed_logits(x, self.weight, self.bias)


@pytest.mark.parametrize("packed", [False, True])
def test_export_preserves_router_parameter_and_gradients(packed):
    torch.manual_seed(739)
    model = Router(packed)
    x = torch.randn(7, 16, dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(7, 3)
    exported = torch.export.export(model, (x,), strict=True)
    assert not any(
        spec.kind is InputKind.CONSTANT_TENSOR
        for spec in exported.graph_signature.input_specs
    )
    weight = model.weight._data if packed else model.weight
    result = torch.compile(exported.module(), backend="aot_eager", fullgraph=True)(x)
    gradients = torch.autograd.grad(result, (x, model.weight, model.bias), dy)
    torch.testing.assert_close(result, x.float() @ weight.float().t() + model.bias)
    expected = (
        (dy @ weight.float()).to(x.dtype),
        (dy.t() @ x.float()).to(model.weight.dtype),
        dy.sum(0),
    )
    for actual, reference in zip(gradients, expected, strict=True):
        torch.testing.assert_close(actual, reference)
    assert gradients[1].dtype == model.weight.dtype
