"""MoE backwards work with non-reentrant activation recomputation."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import mlops
from mlops.dispatch.context import use_implementation
from mlops.providers.sonicmoe import moe as sonicmoe_moe


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "implementation",
    [
        "builtin.moe.grouped_gemm",
        "builtin.moe.grouped_gemm_composed",
        "sonicmoe.moe",
        "sonicmoe.moe.composed",
    ],
)
def test_checkpoint_preserves_moe_output_and_gradients(implementation):
    torch.manual_seed(1729)
    sizes = [(64, 128), (64, 128), (8, 128), (8, 256, 128), (8, 128, 128)]
    original = [
        (torch.randn(size, device="cuda") * 0.1).to(
            torch.float32 if index == 2 else torch.bfloat16
        )
        for index, size in enumerate(sizes)
    ]
    if "sonicmoe" in implementation and not sonicmoe_moe.supports(
        *original, surface="semantic", top_k=2, routing_mode="topk_then_softmax"
    ):
        pytest.skip("SonicMoE unavailable for this device/environment")

    def operation(*inputs):
        # Re-establish the implementation inside the checkpointed function:
        # autograd recomputation may run on a different Python context.
        with use_implementation("moe", implementation):
            output, auxiliary, _, probabilities = mlops.moe(
                *inputs, top_k=2, routing_mode="topk_then_softmax"
            )
        return output, auxiliary, probabilities

    results = []
    with use_implementation("moe", implementation):
        for recompute in (False, True):
            inputs = [value.detach().clone().requires_grad_() for value in original]
            output, auxiliary, probabilities = (
                checkpoint(operation, *inputs, use_reentrant=False)
                if recompute
                else operation(*inputs)
            )
            loss = (
                output.float().square().mean()
                + auxiliary * 0.1
                + probabilities.square().mean() * 0.01
            )
            loss.backward()
            results.append((output.detach(), *(value.grad for value in inputs)))
    for actual, expected in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
