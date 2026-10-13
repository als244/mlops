"""Frozen parameters get no gradient work from the MoE backward operators.

Every ``moe`` implementation's backward honors ``needs_input_grad``: a router
or expert weight that does not require a gradient gets ``None``, its
weight-gradient GEMM never launches, and the gradients that are computed are
the ones the all-trainable step computes.
"""

from __future__ import annotations

import pytest
import torch

import mlops
from mlops.dispatch.context import use_implementation
from mlops.providers import moe_common
from mlops.providers.builtin import moe_composed as builtin_composed
from mlops.providers.sonicmoe import moe as sonicmoe_moe

ROWS, WIDTH, EXPERTS, EXPERT_WIDTH, TOP_K = 64, 64, 4, 48, 2
IMPLEMENTATIONS = (
    "builtin.moe.grouped_gemm",
    "builtin.moe.grouped_gemm_composed",
    "sonicmoe.moe",
    "sonicmoe.moe.composed",
)
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _inputs(seed, *, router, experts):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.bfloat16, scale=1.0, grad=True):
        values = torch.randn(*shape, device="cuda", generator=generator) * scale
        return values.to(dtype).requires_grad_(grad)

    return (
        randn(ROWS, WIDTH),
        randn(ROWS, WIDTH),
        randn(EXPERTS, WIDTH, dtype=torch.float32, scale=WIDTH**-0.5, grad=router),
        randn(EXPERTS, 2 * EXPERT_WIDTH, WIDTH, scale=WIDTH**-0.5, grad=experts),
        randn(EXPERTS, WIDTH, EXPERT_WIDTH, scale=EXPERT_WIDTH**-0.5, grad=experts),
    )


def _supported(implementation):
    if "sonicmoe" not in implementation:
        return True
    probe = tuple(value.detach() for value in _inputs(0, router=True, experts=True))
    return bool(
        sonicmoe_moe.supports(
            *probe, surface="semantic", top_k=TOP_K, routing_mode="softmax_then_topk"
        )
    )


def _step(implementation, seed, *, router, experts):
    primals = _inputs(seed, router=router, experts=experts)
    with use_implementation("moe", implementation):
        output, aux, _counts, _probability_sum = mlops.moe(
            *primals, top_k=TOP_K, routing_mode="softmax_then_topk"
        )
    torch.manual_seed(seed)
    (output.float() * torch.randn_like(output, dtype=torch.float32)).sum().add(
        0.3 * aux
    ).backward()
    return primals


@pytest.mark.gpu
@cuda
@pytest.mark.parametrize("frozen", ["router", "experts", "both"])
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_frozen_parameters_skip_their_gradients(implementation, frozen, monkeypatch):
    if not _supported(implementation):
        pytest.skip("SonicMoE needs a Hopper or Blackwell GPU with its extras")
    weight_gemms = []
    if "sonicmoe" in implementation:
        from quack import gemm_interface

        real = gemm_interface.gemm

        def counting(*args, **kwargs):
            if kwargs.get("cu_seqlens_k") is not None:
                weight_gemms.append("varlen-K")
            return real(*args, **kwargs)

        monkeypatch.setattr(gemm_interface, "gemm", counting)
    else:
        real = moe_common.grouped_mm_wgrad

        def counting(*args, **kwargs):
            weight_gemms.append("grouped wgrad")
            return real(*args, **kwargs)

        monkeypatch.setattr(moe_common, "grouped_mm_wgrad", counting)
        monkeypatch.setattr(builtin_composed, "grouped_mm_wgrad", counting)
    router, experts = frozen != "router" and frozen != "both", frozen == "router"
    h2, residual, router_weight, w13, w2 = _step(
        implementation, 5, router=router, experts=experts
    )
    assert (router_weight.grad is None) == (not router)
    assert (w13.grad is None) == (not experts) and (w2.grad is None) == (not experts)
    if not experts:
        assert weight_gemms == []
    weight_gemms.clear()
    reference = _step(implementation, 5, router=True, experts=True)
    assert len(weight_gemms) == 2
    # The gradients still computed are exactly the all-trainable step's.
    assert torch.equal(h2.grad, reference[0].grad)
    assert torch.equal(residual.grad, reference[1].grad)
    if router:
        assert torch.equal(router_weight.grad, reference[2].grad)
    if experts:
        assert torch.equal(w13.grad, reference[3].grad)
        assert torch.equal(w2.grad, reference[4].grad)
