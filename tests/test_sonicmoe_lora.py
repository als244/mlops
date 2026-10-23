"""Expert LoRA on SonicMoE's kernels against the builtin path."""

from __future__ import annotations

import pytest
import torch

from mlops.dispatch.context import weight_gradients_at
from mlops.lora.experts import expert_lora
from mlops.providers.sonicmoe import lora as sonicmoe_lora

ROWS, WIDTH, EXPERTS, EXPERT_WIDTH, RANK, TOP_K = 96, 64, 8, 40, 8, 2
MODES = {
    "softmax_then_topk": {},
    "topk_then_softmax": {},
    # GLM's: sigmoid scores, a correction bias (drawn per case), expert
    # groups and the routed scaling.
    "sigmoid_noaux_tc": {"n_group": 2, "topk_group": 1, "routed_scaling": 1.5},
}
# GLM's clamped SwiGLU, about 2.5 standard deviations out. The builtin rounds
# the pre-activation twice (the base and the LoRA term apart), SonicMoE once,
# so a value within a rounding step of the limit can be clipped on one path
# only, which moves its gradient by a whole element; this far out such
# values are rare while a few percent still clip.
LIMIT = 2.75


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


def _inputs(seed, *, train_base, expert_width=EXPERT_WIDTH, rank=RANK):
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
    a13 = randn(EXPERTS, rank, WIDTH, dtype=torch.float32, scale=WIDTH**-0.5)
    # Nonzero B factors, so the LoRA term and its gradients are exercised.
    b13 = randn(EXPERTS, 2 * expert_width, rank, dtype=torch.float32, scale=rank**-0.5)
    a2 = randn(
        EXPERTS, rank, expert_width, dtype=torch.float32, scale=expert_width**-0.5
    )
    b2 = randn(EXPERTS, WIDTH, rank, dtype=torch.float32, scale=rank**-0.5)
    return (hidden, residual, router, gate_up, down, a13, b13, a2, b2)


def _routing(seed, mode):
    """The routing options of ``mode``, the sigmoid mode's correction bias
    drawn from a generator of its own."""
    options = dict(MODES[mode])
    if mode == "sigmoid_noaux_tc":
        generator = torch.Generator(device="cuda").manual_seed(seed + 1000)
        options["router_bias"] = (
            torch.randn(EXPERTS, device="cuda", generator=generator) * 0.1
        )
    return options


def _run(
    implementation,
    seed,
    mode,
    *,
    train_base,
    grad_dtype=None,
    expert_width=EXPERT_WIDTH,
    lengths=None,
    swiglu_limit=None,
    rank=RANK,
):
    primals = _inputs(seed, train_base=train_base, expert_width=expert_width, rank=rank)
    with weight_gradients_at(grad_dtype):
        output, aux, counts, probability_sum = expert_lora(
            *primals,
            top_k=TOP_K,
            routing_mode=mode,
            scale=0.5,
            implementation=implementation,
            lengths=lengths,
            swiglu_limit=swiglu_limit,
            **_routing(seed, mode),
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
@pytest.mark.parametrize("train_base", [False, True])
@pytest.mark.parametrize("grad_dtype", [None, torch.float32])
def test_glm_expert_lora_matches_the_builtin(train_base, grad_dtype):
    """GLM's routing, two sequences' balance statistics and the clamp."""
    options = {"lengths": (32, 64), "swiglu_limit": LIMIT, "grad_dtype": grad_dtype}
    actual = _run("sonicmoe", 9, "sigmoid_noaux_tc", train_base=train_base, **options)
    expected = _run("builtin", 9, "sigmoid_noaux_tc", train_base=train_base, **options)
    for value, reference in zip(actual, expected, strict=True):
        _assert_close(value, reference)


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("implementation", ["builtin", "sonicmoe"])
def test_zero_factors_are_mlops_moe(implementation):
    """With zero B factors expert LoRA is `mlops.moe`, GLM's routing, its
    per-sequence statistics and the clamp included."""
    import mlops

    options = {
        **_routing(13, "sigmoid_noaux_tc"),
        "lengths": (32, 64),
        "swiglu_limit": LIMIT,
    }

    def run(lora):
        primals = _inputs(13, train_base=True)
        hidden, residual, router, gate_up, down, _a13, b13, _a2, b2 = primals
        with torch.no_grad():
            b13.zero_()
            b2.zero_()
        common = {"top_k": TOP_K, "routing_mode": "sigmoid_noaux_tc", **options}
        if lora:
            outputs = expert_lora(
                *primals, scale=0.5, implementation=implementation, **common
            )
        else:
            outputs = mlops.moe(hidden, residual, router, gate_up, down, **common)
        torch.manual_seed(13)
        upstream = torch.randn_like(outputs[0])
        loss = (outputs[0].float() * upstream).sum() + 0.3 * outputs[1]
        bases = (hidden, residual, router, gate_up, down)
        return (*outputs, *torch.autograd.grad(loss, bases))

    for value, reference in zip(run(True), run(False), strict=True):
        _assert_close(value, reference)


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("implementation", ["builtin", "sonicmoe"])
def test_the_limit_changes_the_result(implementation):
    clamped = _run(
        implementation, 7, "softmax_then_topk", train_base=True, swiglu_limit=0.75
    )
    plain = _run(implementation, 7, "softmax_then_topk", train_base=True)
    # output and the gradients of the hidden rows and the base weights
    for index in (0, 4, 7, 8):
        difference = (clamped[index].float() - plain[index].float()).norm()
        assert float(difference / plain[index].float().norm()) > 5e-2, index


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


def _run_fp8(
    implementation,
    seed,
    mode,
    *,
    train_base,
    compute_precision=None,
    swiglu_limit=None,
):
    primals = _fp8_inputs(seed, train_base=train_base)
    output, aux, counts, probability_sum = expert_lora(
        *primals,
        top_k=TOP_K,
        routing_mode=mode,
        scale=0.5,
        implementation=implementation,
        compute_precision=compute_precision,
        swiglu_limit=swiglu_limit,
        **_routing(seed, mode),
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
@pytest.mark.parametrize("train_base", [False, True])
def test_glm_on_an_fp8_base(train_base):
    """The clamp reaches the FP8 kernels. FP8 moves pre-activations by a few
    percent, enough to cross a limit the bf16 run did not, so the comparison
    with bf16 uses a limit far out; `test_sonicmoe_swiglu_limit.py` checks
    the FP8 epilogues' clamp on shared pre-activations."""
    actual = _run_fp8(
        "sonicmoe", 5, "sigmoid_noaux_tc", train_base=train_base, swiglu_limit=4.0
    )
    expected = _run(
        "builtin",
        5,
        "sigmoid_noaux_tc",
        train_base=train_base,
        expert_width=FP8_EXPERT_WIDTH,
        swiglu_limit=4.0,
    )
    for value, reference in zip(actual, expected, strict=True):
        _assert_close(value, reference, **FP8_BOUNDS)
    clamped = _run_fp8(
        "sonicmoe", 5, "sigmoid_noaux_tc", train_base=train_base, swiglu_limit=0.75
    )
    plain = _run_fp8("sonicmoe", 5, "sigmoid_noaux_tc", train_base=train_base)
    for index in (0, 4):  # output and the hidden rows' gradient
        difference = (clamped[index].float() - plain[index].float()).norm()
        assert float(difference / plain[index].float().norm()) > 5e-2, index


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


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("rank", [1, 4, 7, 9])
@pytest.mark.parametrize("train_base", [False, True])
def test_unaligned_lora_rank_matches_builtin(rank, train_base):
    options = {"train_base": train_base, "rank": rank, "grad_dtype": torch.float32}
    actual = _run("sonicmoe", 3, "sigmoid_noaux_tc", **options)
    expected = _run("builtin", 3, "sigmoid_noaux_tc", **options)
    for value, reference in zip(actual, expected, strict=True):
        _assert_close(value, reference)


def test_aligned_lora_factors_need_no_copy():
    a = torch.empty(2, 16, 32)
    b = torch.empty(2, 64, 16)
    aligned_a, aligned_b = sonicmoe_lora._aligned_factors(a, b)
    assert aligned_a is a and aligned_b is b


@pytest.mark.gpu
@fp8_hopper
@pytest.mark.parametrize("rank", [4, 16])
def test_block_fp8_lora_rank_forward_backward(rank):
    from mlops.block_fp8 import BlockFP8Weight

    def run(implementation):
        primals = list(_inputs(5, train_base=False, expert_width=128, rank=rank))
        for index in (3, 4):
            primals[index] = BlockFP8Weight.quantize(primals[index].detach())
        output, aux, counts, probabilities = expert_lora(
            *primals,
            top_k=TOP_K,
            routing_mode="sigmoid_noaux_tc",
            scale=0.5,
            implementation=implementation,
            compute_precision="bf16" if implementation == "builtin" else None,
            **_routing(5, "sigmoid_noaux_tc"),
        )
        torch.manual_seed(5)
        loss = (output.float() * torch.randn_like(output)).sum() + 0.3 * aux
        gradients = torch.autograd.grad(
            loss, [value for value in primals if value.requires_grad]
        )
        assert all(isinstance(primals[i], BlockFP8Weight) for i in (3, 4))
        return output, aux, counts, probabilities, *gradients

    for value, reference in zip(run("sonicmoe"), run("builtin"), strict=True):
        _assert_close(value, reference, **FP8_BOUNDS)
