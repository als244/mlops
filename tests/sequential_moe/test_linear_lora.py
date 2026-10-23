"""Compressed dense LoRA preserves frozen bytes and ordinary linear VJPs."""

import pytest
import torch
from torch import nn

from mlops.lora import LoRAConfig, LoRALinear
from mlops.sequential_moe.precision.linear import FrozenLinear, FrozenLoRALinear
from mlops.sequential_moe.precision.storage import materialize, pack

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("alpha", [5, 8])
@pytest.mark.parametrize("storage", ["fp8", "nvfp4"])
@pytest.mark.parametrize("compiled", [False, True])
def test_compressed_lora_matches_decompressed_linear(storage, compiled, alpha):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("Requires BF16 CUDA")
    torch.manual_seed(431)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * 0.03
    parts, spec = pack(weight, storage, nvfp4_block_rows=1)
    config = LoRAConfig(rank=8, alpha=alpha, factor_dtype=torch.float32)
    candidate = FrozenLoRALinear(FrozenLinear(parts, spec), config)
    dense = nn.Linear(128, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        dense.weight.copy_(materialize(parts, spec, torch.bfloat16))
    dense.requires_grad_(False)
    reference = LoRALinear(dense, config)
    with torch.no_grad():
        candidate.lora_b.normal_(std=0.03)
        reference.lora_a.copy_(candidate.lora_a)
        reference.lora_b.copy_(candidate.lora_b)
    originals = [
        value.detach().reshape(-1).view(torch.uint8).clone()
        for value in candidate.components
    ]
    x = torch.randn(19, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    fn = torch.compile(candidate, fullgraph=True) if compiled else candidate
    actual, expected = fn(x), reference(xr)
    dy = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, (x, candidate.lora_a, candidate.lora_b), dy)
    targets = torch.autograd.grad(
        expected, (xr, reference.lora_a, reference.lora_b), dy
    )
    for got, target in zip((actual, *gradients), (expected, *targets), strict=True):
        error = (got.float() - target.float()).norm() / target.float().norm().clamp_min(
            1e-8
        )
        assert error < 0.015, float(error)
    assert gradients[1].dtype == gradients[2].dtype == torch.float32
    assert all(not p.requires_grad and p.grad is None for p in candidate.components)
    for actual, expected in zip(candidate.components, originals, strict=True):
        torch.testing.assert_close(
            actual.reshape(-1).view(torch.uint8), expected, rtol=0, atol=0
        )
