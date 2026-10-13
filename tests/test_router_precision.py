"""The router GEMM forms its logits one way on every MoE path.

`mlops.kernels.router` gives every implementation the trainers' router:
operands at the activation's dtype, fp32 sums and logits, the activation's
gradient at its dtype, the weight's gradient summed in fp32 from two bf16
pieces of the cotangent. The tests pin the numbers against fp32 references,
the sharing across paths, and the fp32 fallback off CUDA.
"""

from __future__ import annotations

import pytest
import torch

import mlops
from mlops.dispatch.context import use_implementation
from mlops.kernels import router
from mlops.kernels.moe_reference import native_torch_moe
from mlops.lora.experts import expert_lora

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
ROWS, WIDTH, EXPERTS = 4096, 256, 32


def _fp32_reference(hidden, weight, grad_logits):
    """Exact fp32 logits and gradients from the same bf16-rounded operands."""
    x, w = hidden.to(torch.float32), weight.to(hidden.dtype).to(torch.float32)
    logits = x @ w.t()
    if grad_logits is None:
        return logits, None, None
    return logits, grad_logits @ w, grad_logits.t() @ x


@pytest.mark.gpu
@cuda
def test_logits_sum_bf16_products_in_fp32():
    generator = torch.Generator(device="cuda").manual_seed(0)
    hidden = torch.randn(ROWS, WIDTH, device="cuda", generator=generator).to(
        torch.bfloat16
    )
    weight = (
        torch.randn(EXPERTS, WIDTH, device="cuda", generator=generator) * WIDTH**-0.5
    )
    logits = router.router_logits(hidden, weight)
    expected, _, _ = _fp32_reference(hidden, weight, None)
    assert logits.dtype is torch.float32
    # The products of bf16 values are exact in fp32; only the summation order differs.
    torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-5)
    bias = torch.randn(EXPERTS, device="cuda", generator=generator)
    torch.testing.assert_close(
        router.router_logits(hidden, weight, bias),
        expected + bias,
        atol=1e-4,
        rtol=1e-5,
    )


@pytest.mark.gpu
@cuda
def test_gradients_match_fp32_references():
    generator = torch.Generator(device="cuda").manual_seed(1)
    hidden = torch.randn(ROWS, WIDTH, device="cuda", generator=generator).to(
        torch.bfloat16
    )
    weight = (
        torch.randn(EXPERTS, WIDTH, device="cuda", generator=generator) * WIDTH**-0.5
    )
    grad_logits = torch.randn(ROWS, EXPERTS, device="cuda", generator=generator)
    _, expected_hidden, expected_weight = _fp32_reference(hidden, weight, grad_logits)
    grad_hidden = router.router_input_gradient(grad_logits, weight, hidden)
    assert grad_hidden.dtype is torch.bfloat16
    torch.testing.assert_close(
        grad_hidden, expected_hidden.to(torch.bfloat16), atol=2e-2, rtol=2e-2
    )
    grad_weight = router.router_weight_gradient(grad_logits, hidden, torch.float32)
    assert grad_weight.dtype is torch.float32
    # Two bf16 pieces of the cotangent carry it to ~16 significant bits: the
    # weight's gradient lands within fp32 summation-order noise of exact...
    torch.testing.assert_close(grad_weight, expected_weight, atol=5e-3, rtol=1e-4)
    # ...where one piece alone does not.
    one_piece = (grad_logits.to(torch.bfloat16).t().to(torch.float32)) @ hidden.to(
        torch.float32
    )
    assert (one_piece - expected_weight).abs().max() > (
        grad_weight - expected_weight
    ).abs().max() * 4
    accumulated = torch.zeros_like(hidden)
    router.add_router_input_gradient_(accumulated, grad_logits, weight)
    torch.testing.assert_close(accumulated, grad_hidden)


@pytest.mark.gpu
@cuda
def test_autograd_form_returns_the_same_gradients():
    generator = torch.Generator(device="cuda").manual_seed(2)
    hidden = (
        torch.randn(ROWS, WIDTH, device="cuda", generator=generator)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    weight = (
        torch.randn(EXPERTS, WIDTH, device="cuda", generator=generator) * WIDTH**-0.5
    ).requires_grad_()
    bias = torch.randn(EXPERTS, device="cuda", generator=generator).requires_grad_()
    logits = router.routed_logits(hidden, weight, bias)
    grad_logits = torch.randn_like(logits)
    logits.backward(grad_logits)
    assert torch.equal(
        hidden.grad, router.router_input_gradient(grad_logits, weight.detach(), hidden)
    )
    assert torch.equal(
        weight.grad,
        router.router_weight_gradient(grad_logits, hidden.detach(), torch.float32),
    )
    assert torch.equal(bias.grad, grad_logits.sum(0))
    weight_bf16 = weight.detach().to(torch.bfloat16).requires_grad_()
    router.routed_logits(hidden.detach(), weight_bf16).backward(grad_logits)
    assert weight_bf16.grad.dtype is torch.bfloat16
    router.routed_logits(
        hidden.detach(),
        weight.detach().requires_grad_(),
        weight_grad_dtype=torch.bfloat16,
    )


def test_off_cuda_the_products_are_fp32():
    hidden = torch.randn(64, WIDTH).to(torch.bfloat16)
    weight = torch.randn(EXPERTS, WIDTH)
    expected = hidden.to(torch.float32) @ weight.t()
    assert torch.equal(router.router_logits(hidden, weight), expected)
    assert router.operand_dtype(hidden) is torch.float32
    grad_logits = torch.randn(64, EXPERTS)
    assert torch.equal(
        router.router_weight_gradient(grad_logits, hidden, torch.float32),
        grad_logits.t() @ hidden.to(torch.float32),
    )


@pytest.mark.gpu
@cuda
def test_every_moe_path_routes_identically():
    """The same logits bitwise on the builtin and SonicMoE operators, the
    native reference and the builtin LoRA composition."""
    generator = torch.Generator(device="cuda").manual_seed(3)
    rows, width, experts, expert_width, top_k = 512, 64, 8, 48, 2

    def randn(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(
            dtype
        )

    h2 = randn(rows, width)
    residual = randn(rows, width)
    router_weight = randn(experts, width, scale=width**-0.5, dtype=torch.float32)
    w13 = randn(experts, 2 * expert_width, width, scale=width**-0.5)
    w2 = randn(experts, width, expert_width, scale=expert_width**-0.5)
    outputs = {}
    for implementation in (
        "builtin.moe.grouped_gemm",
        "builtin.moe.grouped_gemm_composed",
        "sonicmoe.moe",
        "sonicmoe.moe.composed",
    ):
        with use_implementation("moe", implementation):
            try:
                outputs[implementation] = mlops.moe(
                    h2,
                    residual,
                    router_weight,
                    w13,
                    w2,
                    top_k=top_k,
                    routing_mode="softmax_then_topk",
                )[2]
            except Exception as error:  # an implementation this GPU does not serve
                if "SonicMoE" not in str(error) and "sonicmoe" not in implementation:
                    raise
    counts = list(outputs.values())
    assert len(counts) >= 2
    for other in counts[1:]:
        assert torch.equal(counts[0], other)
    reference_counts = native_torch_moe(
        h2,
        residual,
        router_weight,
        w13,
        w2,
        torch.empty(0, device="cuda"),
        None,
        None,
        None,
        top_k=top_k,
        routing_mode="softmax_then_topk",
        n_group=1,
        topk_group=1,
        routed_scaling=1.0,
        lengths=None,
        shared_mode="add",
    )[2]
    assert torch.equal(counts[0], reference_counts)
    rank = 4
    factors = [
        randn(experts, rank, width, scale=width**-0.5),
        torch.zeros(
            experts, 2 * expert_width, rank, device="cuda", dtype=torch.bfloat16
        ),
        randn(experts, rank, expert_width, scale=expert_width**-0.5),
        torch.zeros(experts, width, rank, device="cuda", dtype=torch.bfloat16),
    ]
    lora_counts = expert_lora(
        h2,
        residual,
        router_weight,
        w13,
        w2,
        *factors,
        top_k=top_k,
        routing_mode="softmax_then_topk",
        scale=1.0,
        implementation="builtin",
    )[2]
    assert torch.equal(counts[0], lora_counts)
