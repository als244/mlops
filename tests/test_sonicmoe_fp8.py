"""FP8 expert math on the SonicMoE implementations: ``QuackFP8Weight``
containers as expert weights, ``compute_precision`` apart from them, and the
kernels around the GEMMs."""

from __future__ import annotations

import pytest
import torch

import mlops
from mlops.dispatch import resolve_implementation
from mlops.dispatch.context import use_implementation, weight_gradients_at
from mlops.expert_parallel.quack.parameters.fp8 import QuackFP8Weight
from mlops.providers.sonicmoe import fp8 as sonicmoe_fp8
from mlops.providers.sonicmoe import moe as sonicmoe_moe

# Neither width equals the other or half of it; all multiples of 16 (the FP8
# rows' 16-byte pitch) with enough columns for the scales to matter.
ROWS, WIDTH, EXPERTS, EXPERT_WIDTH, TOP_K = 96, 128, 8, 64, 2
SINGLE, COMPOSED = "sonicmoe.moe", "sonicmoe.moe.composed"
MODES = ("softmax_then_topk", "topk_then_softmax")


def _probe(device="cuda"):
    return (
        torch.empty(ROWS, WIDTH, device=device, dtype=torch.bfloat16),
        torch.empty(ROWS, WIDTH, device=device, dtype=torch.bfloat16),
        torch.empty(EXPERTS, WIDTH, device=device),
        torch.empty(
            EXPERTS, 2 * EXPERT_WIDTH, WIDTH, device=device, dtype=torch.bfloat16
        ),
        torch.empty(EXPERTS, WIDTH, EXPERT_WIDTH, device=device, dtype=torch.bfloat16),
    )


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    return bool(
        sonicmoe_moe.supports(
            *_probe(),
            surface="semantic",
            top_k=TOP_K,
            routing_mode="softmax_then_topk",
            compute_precision="fp8_current",
        )
    )


hopper = pytest.mark.skipif(
    not _supported(), reason="needs an SM90 GPU with the quack and sonicmoe extras"
)


def _inputs(seed, *, fp8_weights=False):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.bfloat16, scale=1.0):
        values = torch.randn(*shape, device="cuda", generator=generator) * scale
        return values.to(dtype).requires_grad_()

    w13 = randn(EXPERTS, 2 * EXPERT_WIDTH, WIDTH, scale=WIDTH**-0.5)
    w2 = randn(EXPERTS, WIDTH, EXPERT_WIDTH, scale=EXPERT_WIDTH**-0.5)
    if fp8_weights:
        # The containers are the leaves: built once, as from a checkpoint.
        w13 = QuackFP8Weight.quantize(w13.detach(), interleaved=True).requires_grad_()
        w2 = QuackFP8Weight.quantize(w2.detach()).requires_grad_()
    return (
        randn(ROWS, WIDTH),
        randn(ROWS, WIDTH),
        randn(EXPERTS, WIDTH, dtype=torch.float32, scale=WIDTH**-0.5),
        w13,
        w2,
    )


def _run(
    implementation,
    seed,
    mode,
    *,
    compute_precision=None,
    grad_dtype=None,
    fp8_weights=False,
    swiglu_limit=None,
):
    primals = _inputs(seed, fp8_weights=fp8_weights)
    with use_implementation("moe", implementation), weight_gradients_at(grad_dtype):
        output, aux, counts, probability_sum = mlops.moe(
            *primals,
            top_k=TOP_K,
            routing_mode=mode,
            compute_precision=compute_precision,
            swiglu_limit=swiglu_limit,
        )
    torch.manual_seed(seed)
    upstream = torch.randn_like(output)
    loss = (output.float() * upstream).sum() + 0.3 * aux
    gradients = torch.autograd.grad(loss, primals)
    return (output, aux, counts, probability_sum, *gradients)


NAMES = (
    "output",
    "aux",
    "counts",
    "probability_sum",
    "grad_h2",
    "grad_residual",
    "grad_router",
    "grad_w13",
    "grad_w2",
)
# E4M3 with one scale per row carries about 6% relative error per element,
# more for the small entries of a gradient row. Observed on an H100 at this
# geometry: output 1.7e-2, grad_h2 6.4e-2 (two quantized operands and the
# sum over the row's slots), the weight gradients 2-3e-2.
BOUNDS = {
    "output": 4e-2,
    "grad_h2": 8e-2,
    "grad_router": 8e-2,
    "grad_w13": 6e-2,
    "grad_w2": 6e-2,
}


def _assert_close(name, actual, expected, bound=None):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if not actual.is_floating_point():
        assert torch.equal(actual, expected), name
        return
    actual, expected = actual.detach().float(), expected.detach().float()
    relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-6))
    bound = BOUNDS.get(name, 2e-2) if bound is None else bound
    assert relative <= bound, f"{name}: relative error {relative:.3e}"


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("implementation", [COMPOSED, SINGLE])
@pytest.mark.parametrize("grad_dtype", [None, torch.float32])
def test_fp8_weights_agree_with_the_bf16_form(implementation, mode, grad_dtype):
    actual = _run(implementation, 7, mode, grad_dtype=grad_dtype, fp8_weights=True)
    expected = _run(implementation, 7, mode, grad_dtype=grad_dtype)
    for name, value, reference in zip(NAMES, actual, expected, strict=True):
        _assert_close(name, value, reference)
    # The containers' gradients are plain tensors in nn.Linear order at the
    # logical (or the policy's) dtype, like the bf16 weights' own.
    assert actual[7].dtype is expected[7].dtype and actual[8].dtype is expected[8].dtype


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("implementation", [COMPOSED, SINGLE])
def test_the_limit_reaches_the_fp8_kernels(implementation):
    """FP8 moves pre-activations by a few percent, enough to cross a limit the
    bf16 run did not, so the comparison with bf16 uses a limit far out;
    `test_sonicmoe_swiglu_limit.py` checks the FP8 epilogues' clamp on
    shared pre-activations."""
    actual = _run(
        implementation, 7, "softmax_then_topk", fp8_weights=True, swiglu_limit=3.5
    )
    expected = _run(implementation, 7, "softmax_then_topk", swiglu_limit=3.5)
    for name, value, reference in zip(NAMES, actual, expected, strict=True):
        _assert_close(name, value, reference)
    clamped = _run(
        implementation, 7, "softmax_then_topk", fp8_weights=True, swiglu_limit=0.75
    )
    plain = _run(implementation, 7, "softmax_then_topk", fp8_weights=True)
    for index in (0, 4, 7, 8):  # output and the gradients of h2, w13, w2
        value, reference = (
            clamped[index].detach().float(),
            plain[index].detach().float(),
        )
        assert float((value - reference).norm() / reference.norm()) > 5e-2, NAMES[index]


@pytest.mark.gpu
@hopper
def test_bf16_weights_asked_for_fp8_math_are_quantized_per_call():
    per_call = _run(COMPOSED, 7, "softmax_then_topk", compute_precision="fp8_current")
    stored = _run(COMPOSED, 7, "softmax_then_topk", fp8_weights=True)
    for value, other in zip(per_call, stored, strict=True):
        assert torch.equal(value, other)


@pytest.mark.gpu
@hopper
def test_fp8_weights_asked_for_bf16_math_are_dequantized_per_call():
    reference = _run(COMPOSED, 7, "softmax_then_topk")
    for implementation in (
        COMPOSED,
        "builtin.moe.grouped_gemm_composed",
        "native_torch.moe",
    ):
        actual = _run(
            implementation,
            7,
            "softmax_then_topk",
            compute_precision="bf16",
            fp8_weights=True,
        )
        for name, value, expected in zip(NAMES, actual, reference, strict=True):
            # Only the weights' own quantization error separates the two.
            _assert_close(name, value, expected, bound=5e-2)


@pytest.mark.gpu
@hopper
def test_fp8_is_repeatable_and_the_forms_agree():
    composed = _run(COMPOSED, 11, "softmax_then_topk", fp8_weights=True)
    again = _run(COMPOSED, 11, "softmax_then_topk", fp8_weights=True)
    single = _run(SINGLE, 11, "softmax_then_topk", fp8_weights=True)
    for value, repeat in zip(composed, again, strict=True):
        assert torch.equal(value, repeat)
    for name, value, other in zip(NAMES, composed, single, strict=True):
        # The composed form recomputes the activation from the bf16 ``h``.
        _assert_close(name, value, other)


@pytest.mark.gpu
@hopper
def test_dispatch_follows_the_weights_and_the_precision():
    plain = _probe()
    containers = (
        *plain[:3],
        QuackFP8Weight.quantize(plain[3], interleaved=True),
        QuackFP8Weight.quantize(plain[4]),
    )
    options = {
        "surface": "semantic",
        "top_k": TOP_K,
        "routing_mode": "softmax_then_topk",
    }
    chosen = resolve_implementation("moe", *containers, **options)
    assert chosen.implementation_id == COMPOSED
    chosen = resolve_implementation(
        "moe", *plain, compute_precision="fp8_current", **options
    )
    assert chosen.implementation_id == COMPOSED
    from mlops.providers.builtin import moe as builtin_moe

    verdict = builtin_moe.supports(*containers, **options)
    assert not verdict and "compute_precision='bf16'" in verdict.reason
    assert builtin_moe.supports(*containers, compute_precision="bf16", **options)
    not_interleaved = (*plain[:3], QuackFP8Weight.quantize(plain[3]), containers[4])
    verdict = sonicmoe_moe.supports(*not_interleaved, **options)
    assert not verdict and "interleaved=True" in verdict.reason


@pytest.mark.gpu
@hopper
def test_the_containers_components_are_the_operators_operands():
    weight = torch.randn(3, 8, 32, device="cuda", dtype=torch.bfloat16)
    plain = QuackFP8Weight.quantize(weight)
    data, transposed, scale, transposed_scale = plain.components()
    assert data.shape == (3, 8, 32) and data.dtype is torch.float8_e4m3fn
    assert transposed.shape == (3, 32, 8) and scale.shape == (3, 8)
    assert transposed_scale.shape == (3, 32) and data.is_contiguous()
    row_max = weight.float().abs().amax(-1, keepdim=True)
    restored = data.float() * scale[..., None]
    assert float(((restored - weight.float()).abs() / row_max).max()) < 0.07
    restored_t = (transposed.float() * transposed_scale[..., None]).transpose(1, 2)
    assert float(((restored_t - weight.float()).abs() / row_max).max()) < 0.07
    interleaved = QuackFP8Weight.quantize(weight, interleaved=True)
    # Row 2j of expert e is gate row j, row 2j + 1 its up row j, in both
    # orientations; the logical view is nn.Linear order again.
    expected = torch.stack(data.chunk(2, dim=1), dim=2).reshape(data.shape)
    assert torch.equal(interleaved._data.view(torch.uint8), expected.view(torch.uint8))
    expected_t = torch.stack(transposed.chunk(2, dim=2), dim=3).reshape(
        transposed.shape
    )
    assert torch.equal(
        interleaved._transposed.view(torch.uint8), expected_t.view(torch.uint8)
    )
    assert torch.equal(interleaved.dequantize(), plain.dequantize())
    # Publication from a master keeps the layout and matches quantization.
    published = QuackFP8Weight(
        *[torch.empty_like(v) for v in interleaved.components()],
        dtype=torch.bfloat16,
        interleaved=True,
    )
    published.copy_(weight * 1.5)
    again = QuackFP8Weight.quantize(weight * 1.5, interleaved=True)
    for value, other in zip(published.components(), again.components(), strict=True):
        assert torch.equal(value.view(torch.uint8), other.view(torch.uint8))
    with pytest.raises(ValueError, match="row orders"):
        published.copy_(plain)


@pytest.mark.gpu
@hopper
def test_row_kernels_compile_once_per_width():
    """Token counts, gathers, pitches and SwiGLU limits are runtime arguments:
    no launch dimension or limit compiles another kernel."""
    from mlops.expert_parallel.quack.kernels import quantize_rows, swiglu_rows
    from mlops.expert_parallel.quack.kernels.quantize_transpose import (
        _batched_transpose_quantize,
        quantize_weights_fp8,
    )

    device = torch.cuda.current_device()

    def variants(kernel):
        return len(kernel.device_caches[device][0])

    def _activations(h, limit):
        swiglu_rows.swiglu_rows(h, limit=limit)
        swiglu_rows.swiglu_quantize_rows_fp8(h, limit=limit)

    def _backwards(h, limit=None):
        cotangent = torch.randn(
            h.shape[0], h.shape[1] // 2, device="cuda", dtype=h.dtype
        )
        swiglu_rows.swiglu_rows_backward(cotangent, h, limit=limit)
        score = torch.rand(h.shape[0], device="cuda")
        swiglu_rows.routed_swiglu_rows_backward(cotangent, h, score, limit=limit)
        swiglu_rows.routed_swiglu_rows_backward(
            cotangent, h, score, scaled_activation=True, limit=limit
        )

    kernels = (
        quantize_rows._quantize_rows,
        swiglu_rows._swiglu_rows,
        swiglu_rows._swiglu_rows_backward,
        swiglu_rows._routed_swiglu_rows_backward,
        _batched_transpose_quantize,
    )
    reference = torch.randn(ROWS, WIDTH, device="cuda", dtype=torch.bfloat16)
    quantize_rows.quantize_rows_fp8(reference)
    quantize_rows.quantize_rows_fp8(
        reference, gather=torch.arange(ROWS, device="cuda", dtype=torch.int32)
    )
    for limit in (None, 10.0):  # one variant without a limit, one with
        _activations(reference, limit)
        _backwards(reference, limit)
    quantize_weights_fp8(torch.randn(2, 16, WIDTH, device="cuda", dtype=torch.bfloat16))
    before = tuple(variants(kernel) for kernel in kernels)
    for rows in (1, 5, ROWS + 7, 3 * ROWS):
        value = torch.randn(rows, WIDTH, device="cuda", dtype=torch.bfloat16)
        quantize_rows.quantize_rows_fp8(value)
        gather = torch.randint(0, rows, (rows + 3,), device="cuda", dtype=torch.int64)
        quantize_rows.quantize_rows_fp8(value, gather=gather)
        quantize_rows.quantize_rows_fp8(value[:, : WIDTH // 2])  # another pitch
        for limit in (None, 10.0, 0.75, 7):
            _activations(value, limit)
            _backwards(value, limit)
    for experts, out in ((1, 32), (3, 48), (2, 64)):
        quantize_weights_fp8(
            torch.randn(experts, out, WIDTH, device="cuda", dtype=torch.bfloat16),
            interleave_rows=False,
        )
    after = tuple(variants(kernel) for kernel in kernels)
    assert after == before, (before, after)


def test_the_facade_checks_the_precision_and_bf16_math_needs_asking_for():
    from mlops.providers.builtin import moe as builtin_moe

    probe = _probe("cpu")
    verdict = builtin_moe.supports(
        *probe,
        surface="semantic",
        top_k=TOP_K,
        routing_mode="softmax_then_topk",
        compute_precision="fp8_current",
    )
    assert not verdict and "fp8_current" in verdict.reason
    assert not sonicmoe_moe.supports(
        *probe,
        surface="semantic",
        top_k=TOP_K,
        routing_mode="softmax_then_topk",
        compute_precision="fp8_current",
    )
    assert "9.x" in sonicmoe_fp8.supports(probe[0], probe[3], probe[4]).reason
    with pytest.raises(ValueError, match="unsupported compute precision"):
        mlops.moe(
            *probe,
            top_k=TOP_K,
            routing_mode="softmax_then_topk",
            compute_precision="int4",
        )
    containers = (
        *probe[:3],
        QuackFP8Weight.quantize(probe[3], interleaved=True),
        QuackFP8Weight.quantize(probe[4]),
    )
    verdict = builtin_moe.supports(
        *containers, surface="semantic", top_k=TOP_K, routing_mode="softmax_then_topk"
    )
    assert not verdict and "compute_precision='bf16'" in verdict.reason
    assert builtin_moe.supports(
        *containers,
        surface="semantic",
        top_k=TOP_K,
        routing_mode="softmax_then_topk",
        compute_precision="bf16",
    )


def test_the_container_round_trips_on_any_device():
    weight = torch.randn(3, 8, 32, dtype=torch.bfloat16)
    plain = QuackFP8Weight.quantize(weight)
    interleaved = QuackFP8Weight.quantize(weight, interleaved=True)
    assert plain.shape == interleaved.shape == weight.shape
    assert plain.dtype is torch.bfloat16 and interleaved.interleaved
    assert torch.equal(plain.dequantize(), interleaved.dequantize())
    error = (plain.dequantize().float() - weight.float()).abs()
    assert float((error / weight.float().abs().amax(-1, keepdim=True)).max()) < 0.07
    # Straight-through: the container's gradient is the weight's, and a
    # dequantized view's gradient is the container's.
    leaf = weight.clone().requires_grad_()
    QuackFP8Weight.quantize(
        leaf, interleaved=True
    ).dequantize().float().sum().backward()
    assert torch.equal(leaf.grad, torch.ones_like(leaf))
    container = QuackFP8Weight.quantize(weight).requires_grad_()
    container.dequantize(dtype=torch.float32).sum().backward()
    assert container.grad.dtype is torch.bfloat16 and bool((container.grad == 1).all())
    with pytest.raises(ValueError, match="even"):
        QuackFP8Weight.quantize(torch.randn(2, 7, 32), interleaved=True)
