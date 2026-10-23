"""GLM's clamped SwiGLU on the SonicMoE paths.

The clamp -- the gate capped at the limit, the up projection clipped to
``[-limit, limit]``, no gradient through a clipped value, the builtin
``packed_swiglu(limit=)`` semantics -- makes the activation's gradient jump
where a pre-activation crosses the limit, so two paths that round a
pre-activation differently can disagree there by a whole element. These
checks therefore hand every form the same pre-activation: the Triton row
kernels against a PyTorch reference, and each Quack epilogue (bf16 and FP8)
against the row kernels on the pre-activation its own GEMM produced or
consumed. The operators' end-to-end agreement with the builtin is in
``test_sonicmoe_moe.py`` and ``test_sonicmoe_lora.py``.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from mlops.providers.sonicmoe import moe as sonicmoe_moe

# FP8 rows need 16-element widths; the expert width is neither the model
# width nor half of it.
ROWS, WIDTH, EXPERTS, EXPERT_WIDTH, TOP_K = 96, 128, 4, 48, 2
LIMIT = 1.5  # exact in bf16; the pre-activations below cross it often


def _supported(compute_precision=None) -> bool:
    if not torch.cuda.is_available():
        return False
    probe = (
        torch.empty(ROWS, WIDTH, device="cuda", dtype=torch.bfloat16),
        torch.empty(ROWS, WIDTH, device="cuda", dtype=torch.bfloat16),
        torch.empty(EXPERTS, WIDTH, device="cuda"),
        torch.empty(
            EXPERTS, 2 * EXPERT_WIDTH, WIDTH, device="cuda", dtype=torch.bfloat16
        ),
        torch.empty(EXPERTS, WIDTH, EXPERT_WIDTH, device="cuda", dtype=torch.bfloat16),
    )
    return bool(
        sonicmoe_moe.supports(
            *probe,
            surface="semantic",
            top_k=TOP_K,
            routing_mode="softmax_then_topk",
            compute_precision=compute_precision,
        )
    )


pytestmark = pytest.mark.gpu
hopper = pytest.mark.skipif(
    not _supported(),
    reason="needs a Hopper or Blackwell GPU with the quack and sonicmoe extras",
)
fp8_hopper = pytest.mark.skipif(
    not _supported("fp8_current"),
    reason="needs an SM90 GPU with the quack and sonicmoe extras",
)


def _reference(h, limit=LIMIT):
    """The clamped SwiGLU of an interleaved ``h [rows, 2I]`` in FP32,
    differentiable in ``h``."""
    gate, up = h.float()[:, 0::2], h.float()[:, 1::2]
    return functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)


def _relative(actual, expected):
    actual, expected = actual.detach().float(), expected.detach().float()
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-6))


def _preactivation(seed):
    """An interleaved ``h`` spread well past the limit; its first row holds
    the pairs ``(L, L)``, ``(-L, -L)``, ``(2L, 2L)``, ``(-2L, -2L)``: both ends
    of the limit exactly, then past them."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn(ROWS, 2 * EXPERT_WIDTH, device="cuda", generator=generator)
    h = h * 2 * LIMIT
    edges = (LIMIT, LIMIT, -LIMIT, -LIMIT, 2 * LIMIT, 2 * LIMIT, -2 * LIMIT, -2 * LIMIT)
    h[0, : len(edges)] = torch.tensor(edges, device="cuda")
    return h.to(torch.bfloat16)


def _routed(seed):
    """Token rows, expert weights whose pre-activations spread past the limit,
    route weights and SonicMoE's metadata for them."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        values = torch.randn(*shape, device="cuda", generator=generator) * scale
        return values.to(torch.bfloat16)

    hidden = randn(ROWS, WIDTH)
    w13 = randn(EXPERTS, 2 * EXPERT_WIDTH, WIDTH, scale=2 * LIMIT * WIDTH**-0.5)
    w2 = randn(EXPERTS, WIDTH, EXPERT_WIDTH, scale=EXPERT_WIDTH**-0.5)
    choice = torch.rand(ROWS, EXPERTS, device="cuda", generator=generator)
    ids = choice.topk(TOP_K, dim=-1).indices.to(torch.int32)
    weights = torch.rand(ROWS, TOP_K, device="cuda", generator=generator)
    offsets, gather, scatter, _reverse = sonicmoe_moe.metadata(ids, EXPERTS)
    return hidden, w13, w2, weights, offsets, gather, scatter


@hopper
def test_the_row_kernels_clamp_as_the_builtin_does():
    from mlops.kernels.quack_moe import swiglu_rows

    torch.manual_seed(0)
    h = _preactivation(0)
    expected = _reference(h)
    activation = swiglu_rows.swiglu_rows(h, limit=LIMIT)
    torch.testing.assert_close(activation.float(), expected, rtol=1e-2, atol=1e-2)
    # Without the limit the activation is far from it: the clamp took effect.
    assert (swiglu_rows.swiglu_rows(h).float() - expected).abs().max() > 1.0
    data, scales = swiglu_rows.swiglu_quantize_rows_fp8(h, limit=LIMIT)
    assert _relative(data.float() * scales[:, None], expected) < 4e-2

    leaf = h.detach().requires_grad_()
    cotangent = torch.randn(ROWS, EXPERT_WIDTH, device="cuda", dtype=torch.bfloat16)
    (expected_dh,) = torch.autograd.grad(_reference(leaf), leaf, cotangent.float())
    dh = swiglu_rows.swiglu_rows_backward(cotangent, h, limit=LIMIT)
    torch.testing.assert_close(dh.float(), expected_dh.float(), rtol=1e-2, atol=1e-2)
    # At the limit the gradient passes; past it none does, except the gate's
    # below -L, which is not clipped.
    passed = dh[0, :8] != 0
    assert passed.tolist() == [True, True, True, True, False, False, True, False]

    score = torch.rand(ROWS, device="cuda")
    dscore, routed_dh, a_prime = swiglu_rows.routed_swiglu_rows_backward(
        cotangent, h, score, scaled_activation=True, limit=LIMIT
    )
    (expected_routed,) = torch.autograd.grad(
        _reference(leaf), leaf, cotangent.float() * score[:, None]
    )
    torch.testing.assert_close(
        routed_dh.float(), expected_routed.float(), rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        dscore, (cotangent.float() * expected).sum(-1), rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        a_prime.float(), expected * score[:, None], rtol=1e-2, atol=1e-2
    )


@hopper
def test_the_gated_epilogues_clamp_as_the_row_kernels_do():
    from quack.gemm_interface import gemm

    from mlops.kernels.quack_moe.swiglu_rows import (
        routed_swiglu_rows_backward,
        swiglu_rows,
    )

    torch.manual_seed(1)
    hidden, w13, w2, weights, offsets, gather, scatter = _routed(1)
    h, a = sonicmoe_moe.up_projection(hidden, w13, offsets, gather, limit=LIMIT)
    torch.testing.assert_close(a, swiglu_rows(h, limit=LIMIT), rtol=1e-2, atol=1e-2)
    assert (a.float() - swiglu_rows(h).float()).abs().max() > 0.5
    plain_h, _ = sonicmoe_moe.up_projection(hidden, w13, offsets, gather)
    torch.testing.assert_close(h, plain_h)  # the limit leaves h alone

    grad = torch.randn(ROWS, WIDTH, device="cuda", dtype=torch.bfloat16)
    dh, grad_weights, dw2 = sonicmoe_moe.down_backward(
        grad, h, w2, weights, offsets, gather, scatter, None, limit=LIMIT
    )
    # The same row gradient by a plain GEMM, then the row kernel on the same h.
    r = torch.empty(h.shape[0], EXPERT_WIDTH, device="cuda", dtype=torch.bfloat16)
    gemm(grad, w2, out=r, cu_seqlens_m=offsets, A_idx=gather)
    slots = scatter.long()
    score_rows = weights.reshape(-1)[slots].contiguous()
    ds, expected_dh, a_prime = routed_swiglu_rows_backward(
        r, h, score_rows, scaled_activation=True, limit=LIMIT
    )
    assert _relative(dh, expected_dh) < 1e-2
    expected_weights = torch.empty_like(weights.reshape(-1))
    expected_weights[slots] = ds
    assert _relative(grad_weights, expected_weights.view_as(weights)) < 1e-2
    expected_dw2 = torch.empty_like(w2)
    gemm(
        grad.T,
        a_prime,
        out=expected_dw2,
        cu_seqlens_k=offsets,
        A_idx=gather,
        batch_idx_permute=None,
        dynamic_scheduler=False,
    )
    assert _relative(dw2, expected_dw2) < 1e-2


def _fp8_weight(weight, *, interleaved=False):
    from mlops.expert_parallel.quack.parameters.fp8 import QuackFP8Weight
    from mlops.providers.sonicmoe.fp8 import Weight

    container = QuackFP8Weight.quantize(weight, interleaved=interleaved)
    return Weight(
        container._data,
        container._transposed,
        container._scale,
        container._transposed_scale,
    )


@fp8_hopper
def test_the_fp8_epilogues_clamp_as_the_row_kernels_do():
    from mlops.kernels.quack_moe.swiglu_rows import (
        routed_swiglu_rows_backward,
        swiglu_rows,
    )
    from mlops.providers.sonicmoe import fp8

    torch.manual_seed(2)
    hidden, w13, w2, weights, offsets, gather, scatter = _routed(2)
    w13, w2 = _fp8_weight(w13, interleaved=True), _fp8_weight(w2)
    h, a = fp8.up_projection(hidden, w13, offsets, gather, limit=LIMIT)
    torch.testing.assert_close(a, swiglu_rows(h, limit=LIMIT), rtol=1e-2, atol=1e-2)
    assert (a.float() - swiglu_rows(h).float()).abs().max() > 0.5

    # A LoRA term enters before the activation, interleaved like h.
    term = torch.randn_like(h) * 0.5
    h_lora, a_lora = fp8.lora_up_projection(
        hidden, w13, term, offsets, gather=gather, limit=LIMIT
    )
    torch.testing.assert_close(
        a_lora, swiglu_rows(h_lora, limit=LIMIT), rtol=1e-2, atol=1e-2
    )
    assert _relative(h_lora.float() - term.float(), h) < 1e-2

    # Activating and quantizing in one pass, or activating, then quantizing.
    y = fp8.project_down_from_preactivation(h, w2, offsets, limit=LIMIT)
    expected_y = fp8.project_down(swiglu_rows(h, limit=LIMIT), w2, offsets)
    assert _relative(y, expected_y) < 2e-2

    grad = torch.randn(ROWS, WIDTH, device="cuda", dtype=torch.bfloat16)
    score_rows = weights.reshape(-1)[scatter.long()].contiguous()
    dh, a_prime, dscore = fp8.down_backward_activations(
        grad, h, w2, score_rows, offsets, gather, limit=LIMIT
    )
    # The same FP8 row gradient, rounded to bf16, then the row kernel.
    r = fp8.input_gradient_rows(grad, w2, offsets, gather=gather)
    expected_ds, expected_dh, expected_a_prime = routed_swiglu_rows_backward(
        r, h, score_rows, scaled_activation=True, limit=LIMIT
    )
    assert _relative(dh, expected_dh) < 2e-2
    assert _relative(a_prime, expected_a_prime) < 1e-2
    assert _relative(dscore, expected_ds) < 2e-2
