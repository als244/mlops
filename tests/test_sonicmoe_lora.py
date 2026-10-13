"""Expert LoRA on SonicMoE's kernels against the builtin path."""

from __future__ import annotations

import pytest
import torch

from mlops.dispatch.context import weight_gradients_at
from mlops.lora.experts import expert_lora
from mlops.providers.sonicmoe import lora as sonicmoe_lora

ROWS, WIDTH, EXPERTS, EXPERT_WIDTH, RANK, TOP_K = 96, 64, 8, 40, 8, 2
MODES = ("softmax_then_topk", "topk_then_softmax")


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    hidden = torch.empty(ROWS, WIDTH, device="cuda", dtype=torch.bfloat16)
    gate_up = torch.empty(
        EXPERTS, 2 * EXPERT_WIDTH, WIDTH, device="cuda", dtype=torch.bfloat16
    )
    down = torch.empty(
        EXPERTS, WIDTH, EXPERT_WIDTH, device="cuda", dtype=torch.bfloat16
    )
    return bool(sonicmoe_lora.supports(hidden, gate_up, down))


hopper = pytest.mark.skipif(
    not _supported(),
    reason="needs a Hopper or Blackwell GPU with the quack and sonicmoe extras",
)


def _inputs(seed, *, train_base, expert_width=EXPERT_WIDTH):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.bfloat16, scale=1.0, grad=True):
        values = torch.randn(*shape, device="cuda", generator=generator) * scale
        return values.to(dtype).requires_grad_(grad)

    hidden = randn(ROWS, WIDTH)
    residual = randn(ROWS, WIDTH)
    router = randn(EXPERTS, WIDTH, dtype=torch.float32, scale=WIDTH**-0.5)
    gate_up = randn(
        EXPERTS, 2 * expert_width, WIDTH, scale=WIDTH**-0.5, grad=train_base
    )
    down = randn(
        EXPERTS, WIDTH, expert_width, scale=expert_width**-0.5, grad=train_base
    )
    a13 = randn(EXPERTS, RANK, WIDTH, dtype=torch.float32, scale=WIDTH**-0.5)
    # Nonzero B factors, so the LoRA term and its gradients are exercised.
    b13 = randn(EXPERTS, 2 * expert_width, RANK, dtype=torch.float32, scale=RANK**-0.5)
    a2 = randn(
        EXPERTS, RANK, expert_width, dtype=torch.float32, scale=expert_width**-0.5
    )
    b2 = randn(EXPERTS, WIDTH, RANK, dtype=torch.float32, scale=RANK**-0.5)
    return (hidden, residual, router, gate_up, down, a13, b13, a2, b2)


def _run(
    implementation,
    seed,
    mode,
    *,
    train_base,
    grad_dtype=None,
    expert_width=EXPERT_WIDTH,
):
    primals = _inputs(seed, train_base=train_base, expert_width=expert_width)
    with weight_gradients_at(grad_dtype):
        output, aux, counts, probability_sum = expert_lora(
            *primals,
            top_k=TOP_K,
            routing_mode=mode,
            scale=0.5,
            implementation=implementation,
        )
    torch.manual_seed(seed)
    upstream = torch.randn_like(output)
    loss = (output.float() * upstream).sum() + 0.3 * aux
    trainable = [value for value in primals if value.requires_grad]
    gradients = torch.autograd.grad(loss, trainable)
    return (output, aux, counts, probability_sum, *gradients)


def _assert_close(actual, expected, *, bf16=3e-2, fp32=1e-2):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if not actual.is_floating_point():
        assert torch.equal(actual, expected)
        return
    actual, expected = actual.detach(), expected.detach()
    difference = (actual.float() - expected.float()).norm()
    scale = expected.float().norm().clamp_min(1e-6)
    bound = bf16 if actual.dtype is torch.bfloat16 else fp32
    assert float(difference / scale) <= bound, (
        f"relative error {float(difference / scale):.3e} exceeds {bound:.1e}"
    )


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("train_base", [False, True])
@pytest.mark.parametrize("grad_dtype", [None, torch.float32])
def test_sonicmoe_expert_lora_matches_the_builtin(mode, train_base, grad_dtype):
    actual = _run("sonicmoe", 3, mode, train_base=train_base, grad_dtype=grad_dtype)
    expected = _run("builtin", 3, mode, train_base=train_base, grad_dtype=grad_dtype)
    for value, reference in zip(actual, expected, strict=True):
        _assert_close(value, reference)


@pytest.mark.gpu
@hopper
def test_a_frozen_base_gets_no_weight_gradient_kernel(monkeypatch):
    from quack import gemm_interface

    calls = []
    real = gemm_interface.gemm

    base_shapes = {
        (EXPERTS, 2 * EXPERT_WIDTH, WIDTH),
        (EXPERTS, WIDTH, EXPERT_WIDTH),
    }

    def counting(*args, **kwargs):
        # A varlen-K GEMM into a base-weight-shaped output is the base's
        # weight gradient; the factors' gradients are varlen-K GEMMs too.
        out = kwargs.get("out")
        if kwargs.get("cu_seqlens_k") is not None and out is not None:
            if tuple(out.shape) in base_shapes or tuple(out.mT.shape) in base_shapes:
                calls.append("weight gradient")
        return real(*args, **kwargs)

    monkeypatch.setattr(gemm_interface, "gemm", counting)
    _run("sonicmoe", 5, "softmax_then_topk", train_base=False)
    assert calls == []


@pytest.mark.gpu
@hopper
def test_auto_takes_the_sonicmoe_path_here():
    primals = _inputs(1, train_base=False)
    hidden, _residual, _router, gate_up, down, *_ = primals
    assert sonicmoe_lora.supports(hidden, gate_up, down)
    auto = _run("auto", 1, "softmax_then_topk", train_base=False)
    sonic = _run("sonicmoe", 1, "softmax_then_topk", train_base=False)
    for value, reference in zip(auto, sonic, strict=True):
        assert torch.equal(value, reference)


def test_the_sonicmoe_path_declines_elsewhere():
    hidden = torch.empty(4, WIDTH, dtype=torch.bfloat16)
    gate_up = torch.empty(EXPERTS, 2 * EXPERT_WIDTH, WIDTH, dtype=torch.bfloat16)
    down = torch.empty(EXPERTS, WIDTH, EXPERT_WIDTH, dtype=torch.bfloat16)
    assert "CUDA" in sonicmoe_lora.supports(hidden, gate_up, down).reason
    factors = (
        torch.empty(EXPERTS, WIDTH),
        gate_up,
        down,
        torch.empty(EXPERTS, RANK, WIDTH),
        torch.empty(EXPERTS, 2 * EXPERT_WIDTH, RANK),
        torch.empty(EXPERTS, RANK, EXPERT_WIDTH),
        torch.empty(EXPERTS, WIDTH, RANK),
    )
    with pytest.raises(ValueError, match="does not apply here"):
        expert_lora(
            hidden,
            hidden,
            *factors,
            top_k=TOP_K,
            routing_mode="softmax_then_topk",
            scale=0.5,
            implementation="sonicmoe",
        )
    with pytest.raises(ValueError, match="unknown expert LoRA implementation"):
        expert_lora(
            hidden,
            hidden,
            *factors,
            top_k=TOP_K,
            routing_mode="softmax_then_topk",
            scale=0.5,
            implementation="other",
        )


# ---- FP8 base weights -------------------------------------------------------


# E4M3 rows need a 16-element pitch: a wider expert than the bf16 cases'.
FP8_EXPERT_WIDTH = 48


def _fp8_inputs(seed, *, train_base):
    """The same primals with the base as ``QuackFP8Weight`` containers."""
    from mlops.expert_parallel.quack.parameters.fp8 import QuackFP8Weight

    hidden, residual, router, gate_up, down, *factors = _inputs(
        seed, train_base=train_base, expert_width=FP8_EXPERT_WIDTH
    )
    gate_up = QuackFP8Weight.quantize(gate_up.detach(), interleaved=True)
    down = QuackFP8Weight.quantize(down.detach())
    gate_up.requires_grad_(train_base)
    down.requires_grad_(train_base)
    return (hidden, residual, router, gate_up, down, *factors)


def _run_fp8(implementation, seed, mode, *, train_base, compute_precision=None):
    primals = _fp8_inputs(seed, train_base=train_base)
    output, aux, counts, probability_sum = expert_lora(
        *primals,
        top_k=TOP_K,
        routing_mode=mode,
        scale=0.5,
        implementation=implementation,
        compute_precision=compute_precision,
    )
    torch.manual_seed(seed)
    upstream = torch.randn_like(output)
    loss = (output.float() * upstream).sum() + 0.3 * aux
    trainable = [value for value in primals if value.requires_grad]
    gradients = torch.autograd.grad(loss, trainable)
    return (output, aux, counts, probability_sum, *gradients)


def _fp8_supported() -> bool:
    if not _supported():
        return False
    hidden = torch.empty(ROWS, WIDTH, device="cuda", dtype=torch.bfloat16)
    gate_up = torch.empty(
        EXPERTS, 2 * FP8_EXPERT_WIDTH, WIDTH, device="cuda", dtype=torch.bfloat16
    )
    down = torch.empty(
        EXPERTS, WIDTH, FP8_EXPERT_WIDTH, device="cuda", dtype=torch.bfloat16
    )
    return bool(
        sonicmoe_lora.supports(hidden, gate_up, down, compute_precision="fp8_current")
    )


fp8_hopper = pytest.mark.skipif(
    not _fp8_supported(), reason="needs an SM90 GPU with the quack and sonicmoe extras"
)
# E4M3 rows and weights on both projections: the bf16 path's bounds, widened
# as the moe FP8 tests' are.
FP8_BOUNDS = {"bf16": 8e-2, "fp32": 8e-2}


@pytest.mark.gpu
@fp8_hopper
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("train_base", [False, True])
def test_an_fp8_base_matches_the_bf16_base(mode, train_base):
    actual = _run_fp8("sonicmoe", 5, mode, train_base=train_base)
    expected = _run(
        "builtin", 5, mode, train_base=train_base, expert_width=FP8_EXPERT_WIDTH
    )
    for value, reference in zip(actual, expected, strict=True):
        _assert_close(value, reference, **FP8_BOUNDS)
    if train_base:
        # The containers' gradients (after the four outputs: hidden, residual,
        # router, gate_up, down, ...) are plain tensors at the logical dtype.
        grad_gate_up, grad_down = actual[7], actual[8]
        assert grad_gate_up.dtype is grad_down.dtype is torch.bfloat16
        assert grad_gate_up.shape == (EXPERTS, 2 * FP8_EXPERT_WIDTH, WIDTH)
        assert grad_down.shape == (EXPERTS, WIDTH, FP8_EXPERT_WIDTH)


@pytest.mark.gpu
@fp8_hopper
def test_the_precision_option_is_apart_from_the_weights():
    stored = _run_fp8("sonicmoe", 5, "softmax_then_topk", train_base=False)
    per_call = _run(
        "sonicmoe",
        5,
        "softmax_then_topk",
        train_base=False,
        expert_width=FP8_EXPERT_WIDTH,
    )  # bf16 base, bf16 math
    primals = _inputs(5, train_base=False, expert_width=FP8_EXPERT_WIDTH)
    output, *_ = expert_lora(
        *primals,
        top_k=TOP_K,
        routing_mode="softmax_then_topk",
        scale=0.5,
        implementation="sonicmoe",
        compute_precision="fp8_current",
    )
    # bf16 weights quantized per call compute exactly what the containers do.
    assert torch.equal(output, stored[0])
    dequantized = _run_fp8(
        "sonicmoe", 5, "softmax_then_topk", train_base=False, compute_precision="bf16"
    )
    for value, reference in zip(dequantized, per_call, strict=True):
        # Only the weights' quantization error separates bf16 math on the
        # dequantized containers from bf16 math on the bf16 weights.
        _assert_close(value, reference, bf16=5e-2, fp32=5e-2)
    builtin = _run_fp8(
        "builtin", 5, "softmax_then_topk", train_base=False, compute_precision="bf16"
    )
    for value, reference in zip(builtin, dequantized, strict=True):
        _assert_close(value, reference)
    with pytest.raises(ValueError, match="compute_precision='bf16'"):
        _run_fp8("builtin", 5, "softmax_then_topk", train_base=False)
