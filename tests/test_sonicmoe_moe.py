"""The SonicMoE implementations of ``moe`` against the builtin and the oracle."""

from __future__ import annotations

import pytest
import torch

import mlops
from mlops.dispatch import resolve_implementation
from mlops.dispatch.context import use_implementation, weight_gradients_at
from mlops.providers.sonicmoe import moe as sonicmoe_moe

# Neither width equals the other or half of it, so an orientation slip
# changes a shape; both are multiples of 8.
ROWS, WIDTH, EXPERTS, EXPERT_WIDTH, TOP_K = 96, 64, 8, 40, 2
MODES = {
    "softmax_then_topk": {},
    "topk_then_softmax": {},
    "sigmoid_noaux_tc": {"n_group": 2, "topk_group": 1, "routed_scaling": 1.5},
}
SINGLE, COMPOSED = "sonicmoe.moe", "sonicmoe.moe.composed"
BUILTIN, ORACLE = "builtin.moe.grouped_gemm_composed", "native_torch.moe"


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda")
    probe = (
        torch.empty(ROWS, WIDTH, device=device, dtype=torch.bfloat16),
        torch.empty(ROWS, WIDTH, device=device, dtype=torch.bfloat16),
        torch.empty(EXPERTS, WIDTH, device=device),
        torch.empty(
            EXPERTS, 2 * EXPERT_WIDTH, WIDTH, device=device, dtype=torch.bfloat16
        ),
        torch.empty(EXPERTS, WIDTH, EXPERT_WIDTH, device=device, dtype=torch.bfloat16),
    )
    return bool(
        sonicmoe_moe.supports(
            *probe, surface="semantic", top_k=TOP_K, routing_mode="softmax_then_topk"
        )
    )


hopper = pytest.mark.skipif(
    not _supported(),
    reason="needs a Hopper or Blackwell GPU with the quack and sonicmoe extras",
)


def _inputs(seed, mode, *, lengths=None, shared=False):
    """Primals (requiring gradients) and the keyword options of one case."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.bfloat16, scale=1.0):
        values = torch.randn(*shape, device="cuda", generator=generator) * scale
        return values.to(dtype)

    primals = (
        randn(ROWS, WIDTH).requires_grad_(),
        randn(ROWS, WIDTH).requires_grad_(),
        randn(EXPERTS, WIDTH, dtype=torch.float32, scale=WIDTH**-0.5).requires_grad_(),
        randn(EXPERTS, 2 * EXPERT_WIDTH, WIDTH, scale=WIDTH**-0.5).requires_grad_(),
        randn(EXPERTS, WIDTH, EXPERT_WIDTH, scale=EXPERT_WIDTH**-0.5).requires_grad_(),
    )
    options = {"top_k": TOP_K, "routing_mode": mode, "lengths": lengths, **MODES[mode]}
    if mode == "sigmoid_noaux_tc":
        options["router_bias"] = randn(EXPERTS, dtype=torch.float32, scale=0.1)
    if shared:
        options["shared_w13"] = randn(2 * 16, WIDTH, scale=WIDTH**-0.5)
        options["shared_w2"] = randn(WIDTH, 16, scale=16**-0.5)
        options["shared_gate_weight"] = randn(1, WIDTH, scale=WIDTH**-0.5)
        options["shared_mode"] = "sigmoid"
    return primals, options


def _run(implementation, seed, mode, *, lengths=None, shared=False, grad_dtype=None):
    """Outputs and the gradients of a loss touching output, aux and the sums."""
    primals, options = _inputs(seed, mode, lengths=lengths, shared=shared)
    with use_implementation("moe", implementation), weight_gradients_at(grad_dtype):
        output, aux, counts, probability_sum = mlops.moe(*primals, **options)
    torch.manual_seed(seed)
    upstream = torch.randn_like(output)
    vector = torch.randn(EXPERTS, device="cuda")
    loss = (
        (output.float() * upstream).sum() + 0.3 * aux + (probability_sum * vector).sum()
    )
    gradients = torch.autograd.grad(loss, primals)
    return (output, aux, counts, probability_sum, *gradients)


def _assert_close(actual, expected, *, bf16=3e-2, fp32=1e-2):
    """Relative Frobenius error within the dtype's bound; integers exact."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if not actual.is_floating_point():
        assert torch.equal(actual, expected)
        return
    difference = (actual.float() - expected.float()).norm()
    scale = expected.float().norm().clamp_min(1e-6)
    bound = bf16 if actual.dtype is torch.bfloat16 else fp32
    assert float(difference / scale) <= bound, (
        f"relative error {float(difference / scale):.3e} exceeds {bound:.1e}"
    )


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("mode", sorted(MODES))
@pytest.mark.parametrize("implementation", [COMPOSED, SINGLE])
@pytest.mark.parametrize("grad_dtype", [None, torch.float32])
def test_sonicmoe_agrees_with_the_builtin(mode, implementation, grad_dtype):
    actual = _run(implementation, 11, mode, grad_dtype=grad_dtype)
    expected = _run(BUILTIN, 11, mode, grad_dtype=grad_dtype)
    for value, reference in zip(actual, expected, strict=True):
        _assert_close(value, reference)


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("mode", sorted(MODES))
def test_sonicmoe_agrees_with_the_oracle(mode):
    actual = _run(COMPOSED, 23, mode)
    expected = _run(ORACLE, 23, mode)
    for value, reference in zip(actual, expected, strict=True):
        _assert_close(value, reference)


@pytest.mark.gpu
@hopper
def test_sequence_lengths_and_the_shared_expert_match_the_builtin():
    for implementation in (COMPOSED, SINGLE):
        actual = _run(
            implementation, 5, "sigmoid_noaux_tc", lengths=(32, 64), shared=True
        )
        expected = _run(BUILTIN, 5, "sigmoid_noaux_tc", lengths=(32, 64), shared=True)
        for value, reference in zip(actual, expected, strict=True):
            _assert_close(value, reference)


@pytest.mark.gpu
@hopper
def test_the_single_operator_matches_the_composition():
    single = _run(SINGLE, 7, "topk_then_softmax")
    composed = _run(COMPOSED, 7, "topk_then_softmax")
    for value, reference in zip(single, composed, strict=True):
        _assert_close(value, reference, bf16=1e-2, fp32=1e-3)


@pytest.mark.gpu
@hopper
def test_runs_are_bitwise_repeatable():
    for implementation in (COMPOSED, SINGLE):
        first = _run(implementation, 3, "softmax_then_topk")
        second = _run(implementation, 3, "softmax_then_topk")
        for value, again in zip(first, second, strict=True):
            assert torch.equal(value, again)


@pytest.mark.gpu
@hopper
def test_dispatch_prefers_sonicmoe_here():
    primals, options = _inputs(1, "softmax_then_topk")
    implementation = resolve_implementation(
        "moe", *primals, surface="semantic", **options
    )
    assert implementation.implementation_id == COMPOSED
    explicit = resolve_implementation("moe", *primals, surface="explicit", **options)
    assert explicit.provider == "builtin"


def test_supports_declines_elsewhere(monkeypatch):
    """Other devices, capabilities and dtypes go to the builtin."""
    cpu = (
        torch.empty(4, WIDTH, dtype=torch.bfloat16),
        torch.empty(4, WIDTH, dtype=torch.bfloat16),
        torch.empty(EXPERTS, WIDTH),
        torch.empty(EXPERTS, 2 * EXPERT_WIDTH, WIDTH, dtype=torch.bfloat16),
        torch.empty(EXPERTS, WIDTH, EXPERT_WIDTH, dtype=torch.bfloat16),
    )
    options = {
        "surface": "semantic",
        "top_k": TOP_K,
        "routing_mode": "softmax_then_topk",
    }
    assert "CUDA" in sonicmoe_moe.supports(*cpu, **options).reason
    explicit = {**options, "surface": "explicit"}
    assert "semantic" in sonicmoe_moe.supports(*cpu, **explicit).reason
    if not torch.cuda.is_available():
        return
    cuda = tuple(tensor.cuda() for tensor in cpu)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    assert "9.x and 10.x" in sonicmoe_moe.supports(*cuda, **options).reason
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(sonicmoe_moe, "available", lambda: True)
    halves = (cuda[0].half(), cuda[1].half(), cuda[2], cuda[3].half(), cuda[4].half())
    assert "bfloat16" in sonicmoe_moe.supports(*halves, **options).reason
    narrow = (
        cuda[0][:, :60].contiguous(),
        cuda[1][:, :60].contiguous(),
        cuda[2][:, :60].contiguous(),
        cuda[3][:, :, :60].contiguous(),
        cuda[4][:, :60].contiguous(),
    )
    assert "multiples of 8" in sonicmoe_moe.supports(*narrow, **options).reason
