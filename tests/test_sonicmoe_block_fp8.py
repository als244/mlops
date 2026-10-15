"""Block-scaled FP8 expert weights: `BlockFP8Weight`, the grouped block GEMM
(``kernels/block_fp8.py``) and the SonicMoE forms at ``"fp8_block"``."""

from __future__ import annotations

import pytest
import torch

import mlops
from mlops.block_fp8 import BlockFP8Weight
from mlops.dispatch import resolve_implementation
from mlops.dispatch.context import use_implementation, weight_gradients_at
from mlops.kernels.block_fp8 import dequantize_blocks, quantize_blocks
from mlops.lora.experts import expert_lora
from mlops.providers.sonicmoe import moe as sonicmoe_moe

# Widths that leave partial 128-wide blocks on both axes; multiples of 16.
ROWS, WIDTH, EXPERTS, EXPERT_WIDTH, TOP_K, RANK = 96, 192, 8, 80, 2, 8
SINGLE, COMPOSED = "sonicmoe.moe", "sonicmoe.moe.composed"
MODES = ("softmax_then_topk", "sigmoid_noaux_tc")


def _relative(actual, expected):
    actual, expected = actual.detach().float(), expected.detach().float()
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-6))


# ---- the container, on any device --------------------------------------------


def test_the_container_round_trips_on_the_cpu():
    torch.manual_seed(0)
    weight = torch.randn(3, 200, 136) * torch.linspace(0.1, 4, 136)
    container = BlockFP8Weight.quantize(weight, dtype=torch.bfloat16)
    assert container.shape == (3, 200, 136) and container.dtype is torch.bfloat16
    data, scale = container.components()
    assert data.dtype is torch.float8_e4m3fn and data.shape == (3, 200, 136)
    assert scale.dtype is torch.float32 and scale.shape == (3, 2, 2)
    # E4M3 keeps about 3 bits; a block's largest value maps to 448.
    assert _relative(container.dequantize(dtype=torch.float32), weight) < 4e-2
    assert torch.equal(
        dequantize_blocks(data, scale), container.dequantize(dtype=torch.float32)
    )
    names, metadata = container.__tensor_flatten__()
    again = BlockFP8Weight.__tensor_unflatten__(
        {name: getattr(container, name) for name in names}, metadata, None, None
    )
    assert torch.equal(again._data.view(torch.uint8), data.view(torch.uint8))
    published = BlockFP8Weight.quantize(torch.zeros_like(weight))
    published.copy_(weight)  # a master's values, quantized as at init
    assert torch.equal(published._data.view(torch.uint8), data.view(torch.uint8))
    assert torch.equal(published._scale, scale)
    clone = container.detach().clone()
    assert isinstance(clone, BlockFP8Weight) and torch.equal(clone._scale, scale)


def test_quantize_and_dequantize_are_straight_through():
    weight = torch.randn(2, 128, 64, requires_grad=True)
    container = BlockFP8Weight.quantize(weight)
    leaf = container.detach().requires_grad_()
    (leaf.dequantize() * 3).sum().backward()
    assert torch.equal(leaf.grad, torch.full_like(weight, 3.0))
    (gradient,) = torch.autograd.grad(container.dequantize().sum(), weight)
    assert torch.equal(gradient, torch.ones_like(weight))


def test_an_all_zero_block_stays_zero():
    weight = torch.zeros(1, 256, 128)
    weight[0, 128:] = 1.0
    data, scale = quantize_blocks(weight)
    assert torch.count_nonzero(data[0, :128].float()) == 0
    assert scale[0, 0, 0] > 0 and torch.equal(
        dequantize_blocks(data, scale)[0, 128:], weight[0, 128:]
    )


# ---- the kernels, on SM90 ----------------------------------------------------


def _supported(compute_precision="fp8_block") -> bool:
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


hopper = pytest.mark.skipif(
    not _supported(), reason="needs an SM90 GPU with the quack and sonicmoe extras"
)


def _grouped_reference(a, gather, weight, offsets, *, transposed, interleaved_rows):
    """``a [rows]`` (fp32) through ``weight [E, out, in]`` (fp32) per expert."""
    count = a.shape[0] if gather is None else gather.shape[0]
    n = weight.shape[2] if transposed else weight.shape[1]
    out = torch.zeros(count, n, device=a.device)
    bounds = offsets.tolist()
    for expert in range(weight.shape[0]):
        lo, hi = bounds[expert], bounds[expert + 1]
        rows = a[gather[lo:hi].long()] if gather is not None else a[lo:hi]
        if interleaved_rows:
            rows = torch.cat((rows[:, 0::2], rows[:, 1::2]), dim=1)
        w = weight[expert]
        out[lo:hi] = rows @ (w if transposed else w.T)
    return out


def _interleaved(value):
    half = value.shape[1] // 2
    out = torch.empty_like(value)
    out[:, 0::2], out[:, 1::2] = value[:, :half], value[:, half:]
    return out


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("math", ["fp8", "bf16"])
@pytest.mark.parametrize("gathered", [False, True])
def test_the_grouped_block_gemm_matches_the_dequantized_product(
    transposed, math, gathered
):
    from mlops.expert_parallel.quack.kernels.quantize_rows import quantize_rows_fp8
    from mlops.kernels.block_fp8 import grouped_block_gemm

    torch.manual_seed(1)
    counts = [0, 70, 131, 15]  # an empty expert, and tiles left partial
    offsets = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], device="cuda")
    offsets = offsets.to(torch.int32)
    out, inner = 96, 192  # partial blocks on both axes
    weight = torch.randn(4, out, inner, device="cuda") * 0.1
    data, scale = quantize_blocks(weight)
    exact = dequantize_blocks(data, scale)
    k = out if transposed else inner
    tokens = torch.randn(300 if gathered else sum(counts), k, device="cuda")
    gather = (
        torch.randint(0, 300, (sum(counts),), device="cuda", dtype=torch.int32)
        if gathered
        else None
    )
    rows, row_scale = quantize_rows_fp8(tokens.bfloat16(), gather=gather)
    operand = rows.float() * row_scale[:, None]
    actual = grouped_block_gemm(
        rows,
        data,
        scale,
        offsets,
        row_scale=row_scale,
        transposed=transposed,
        math=math,
    )
    expected = _grouped_reference(
        operand, None, exact, offsets, transposed=transposed, interleaved_rows=False
    )
    assert actual.dtype is torch.bfloat16 and actual.shape == expected.shape
    assert _relative(actual, expected) < 1e-2
    # bf16 rows, gathered in the kernel, multiplied in bf16.
    if math == "bf16":
        plain = tokens.bfloat16()
        actual = grouped_block_gemm(
            plain,
            data,
            scale,
            offsets,
            gather=gather,
            transposed=transposed,
            math="bf16",
        )
        expected = _grouped_reference(
            plain.float(),
            gather,
            exact,
            offsets,
            transposed=transposed,
            interleaved_rows=False,
        )
        assert _relative(actual, expected) < 1e-2


@pytest.mark.gpu
@hopper
def test_the_gate_up_layouts_and_the_addend():
    from mlops.expert_parallel.quack.kernels.quantize_rows import quantize_rows_fp8
    from mlops.kernels.block_fp8 import grouped_block_gemm

    torch.manual_seed(2)
    offsets = torch.tensor([0, 40, 40, 140], device="cuda", dtype=torch.int32)
    w13 = torch.randn(3, 2 * 80, 192, device="cuda") * 0.1  # gate rows, up rows
    data, scale = quantize_blocks(w13)
    exact = dequantize_blocks(data, scale)
    x = torch.randn(140, 192, device="cuda").bfloat16()
    rows, row_scale = quantize_rows_fp8(x)
    addend = torch.randn(140, 160, device="cuda").bfloat16()
    h = grouped_block_gemm(
        rows,
        data,
        scale,
        offsets,
        row_scale=row_scale,
        addend=addend,
        interleave_out=True,
    )
    plain = _grouped_reference(
        rows.float() * row_scale[:, None],
        None,
        exact,
        offsets,
        transposed=False,
        interleaved_rows=False,
    )
    assert _relative(h, _interleaved(plain) + addend.float()) < 1e-2
    # The input gradient reads the interleaved dh against gate-then-up rows.
    dh = torch.randn(140, 160, device="cuda").bfloat16()
    rows, row_scale = quantize_rows_fp8(dh)
    dx = grouped_block_gemm(
        rows,
        data,
        scale,
        offsets,
        row_scale=row_scale,
        transposed=True,
        interleaved_rows=True,
    )
    expected = _grouped_reference(
        rows.float() * row_scale[:, None],
        None,
        exact,
        offsets,
        transposed=True,
        interleaved_rows=True,
    )
    assert _relative(dx, expected) < 1e-2


@pytest.mark.gpu
@hopper
def test_the_block_gemm_compiles_once_per_layout():
    from mlops.expert_parallel.quack.kernels.quantize_rows import quantize_rows_fp8
    from mlops.kernels import block_fp8

    device = torch.cuda.current_device()
    weight = BlockFP8Weight.quantize(torch.randn(4, 96, 192, device="cuda"))

    def run(rows):
        counts = torch.randint(0, rows + 1, (4,)).tolist()
        offsets = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()])
        x = torch.randn(sum(counts), 192, device="cuda").bfloat16()
        data, scales = quantize_rows_fp8(x) if sum(counts) else (x, None)
        if sum(counts):
            block_fp8.grouped_block_gemm(
                data,
                *weight.components(),
                offsets.to(device="cuda", dtype=torch.int32),
                row_scale=scales,
                block_rows=64,
            )

    run(50)
    before = len(block_fp8._grouped_block_gemm.device_caches[device][0])
    for rows in (1, 7, 130, 333):
        run(rows)
    assert len(block_fp8._grouped_block_gemm.device_caches[device][0]) == before


# ---- the operators -----------------------------------------------------------


def _inputs(seed, *, block_weights=False, train_base=True, expert_width=EXPERT_WIDTH):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.bfloat16, scale=1.0, grad=True):
        values = torch.randn(*shape, device="cuda", generator=generator) * scale
        return values.to(dtype).requires_grad_(grad)

    w13 = randn(EXPERTS, 2 * expert_width, WIDTH, scale=WIDTH**-0.5)
    w2 = randn(EXPERTS, WIDTH, expert_width, scale=expert_width**-0.5)
    if block_weights:
        # The containers are the leaves, as from a checkpoint.
        w13 = BlockFP8Weight.quantize(w13.detach()).requires_grad_(train_base)
        w2 = BlockFP8Weight.quantize(w2.detach()).requires_grad_(train_base)
    else:
        # The values the containers hold, for the bf16 form.
        w13 = (
            BlockFP8Weight.quantize(w13.detach())
            .dequantize()
            .requires_grad_(train_base)
        )
        w2 = (
            BlockFP8Weight.quantize(w2.detach()).dequantize().requires_grad_(train_base)
        )
    hidden = randn(ROWS, WIDTH)
    residual = randn(ROWS, WIDTH)
    router = randn(EXPERTS, WIDTH, dtype=torch.float32, scale=WIDTH**-0.5)
    bias = torch.randn(EXPERTS, device="cuda", generator=generator) * 0.1
    return (hidden, residual, router, w13, w2), bias


def _routing(mode, bias):
    if mode == "sigmoid_noaux_tc":
        return {"router_bias": bias, "routed_scaling": 2.5}
    return {}


def _run(
    implementation,
    seed,
    mode,
    *,
    block_weights,
    grad_dtype=None,
    limit=None,
    expert_width=EXPERT_WIDTH,
):
    primals, bias = _inputs(
        seed, block_weights=block_weights, expert_width=expert_width
    )
    with use_implementation("moe", implementation), weight_gradients_at(grad_dtype):
        output, aux, counts, probability_sum = mlops.moe(
            *primals,
            top_k=TOP_K,
            routing_mode=mode,
            swiglu_limit=limit,
            **_routing(mode, bias),
        )
    torch.manual_seed(seed)
    upstream = torch.randn_like(output)
    loss = (output.float() * upstream).sum() + 0.3 * aux
    gradients = torch.autograd.grad(loss, primals)
    return (output, aux, counts, probability_sum, *gradients)


# Rows quantized per row per call against the same weight values: the
# activations' E4M3 rounding, as the two-copy form's bounds allow for.
BOUNDS = (4e-2, 2e-2, 0, 2e-2, 8e-2, 2e-2, 8e-2, 6e-2, 6e-2)


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("implementation", [COMPOSED, SINGLE])
@pytest.mark.parametrize("grad_dtype", [None, torch.float32])
def test_block_weights_agree_with_the_bf16_form(implementation, mode, grad_dtype):
    actual = _run(implementation, 7, mode, block_weights=True, grad_dtype=grad_dtype)
    expected = _run(implementation, 7, mode, block_weights=False, grad_dtype=grad_dtype)
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        if not value.is_floating_point():
            assert torch.equal(value, reference)
        else:
            assert _relative(value, reference) <= BOUNDS[index], index
    # The containers' gradients are plain tensors at the logical dtype.
    assert actual[7].dtype is expected[7].dtype
    assert actual[7].shape == (EXPERTS, 2 * EXPERT_WIDTH, WIDTH)


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("implementation", [COMPOSED, SINGLE])
@pytest.mark.parametrize("limit", [None, 4.0])
def test_the_fused_swiglu_kernels_agree_with_the_bf16_form(implementation, limit):
    """Expert width 128 (gate/up halves of whole 64-row groups): the up
    projection's GEMM with the SwiGLU in its epilogue and, in the single
    form, the down projection's input gradient with the SwiGLU backward in
    its epilogue."""
    options = {"block_weights": True, "limit": limit, "expert_width": 128}
    actual = _run(implementation, 11, "sigmoid_noaux_tc", **options)
    options["block_weights"] = False
    expected = _run(implementation, 11, "sigmoid_noaux_tc", **options)
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        if not value.is_floating_point():
            assert torch.equal(value, reference)
        else:
            assert _relative(value, reference) <= BOUNDS[index], index


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("implementation", [COMPOSED, SINGLE])
def test_the_limit_reaches_the_block_kernels(implementation):
    clamped = _run(
        implementation, 9, "sigmoid_noaux_tc", block_weights=True, limit=0.75
    )
    plain = _run(implementation, 9, "sigmoid_noaux_tc", block_weights=True)
    for index in (0, 4, 7, 8):  # output and the gradients of h2, w13, w2
        assert _relative(clamped[index], plain[index]) > 5e-2, index


@pytest.mark.gpu
@hopper
def test_dispatch_follows_block_weights():
    primals, _ = _inputs(3, block_weights=True)
    options = {"top_k": TOP_K, "routing_mode": "softmax_then_topk"}
    selected = resolve_implementation("moe", *primals, surface="semantic", **options)
    assert selected.provider == "sonicmoe"
    from mlops.providers.builtin import moe as builtin_moe

    assert not builtin_moe.supports(*primals, surface="semantic", **options)
    assert builtin_moe.supports(
        *primals, surface="semantic", compute_precision="bf16", **options
    )
    refusal = sonicmoe_moe.supports(
        *primals, surface="semantic", compute_precision="fp8_current", **options
    )
    assert not refusal and "BlockFP8Weight" in refusal.reason
    # bf16 math on the containers dequantizes them per call: the bf16 form.
    with use_implementation("moe", COMPOSED):
        output, *_ = mlops.moe(*primals, compute_precision="bf16", **options)
    plain, _ = _inputs(3, block_weights=False)
    with use_implementation("moe", COMPOSED):
        expected, *_ = mlops.moe(*plain, **options)
    assert torch.equal(output, expected)


def _lora_inputs(seed, *, block_weights, expert_width=EXPERT_WIDTH):
    (hidden, residual, router, w13, w2), bias = _inputs(
        seed, block_weights=block_weights, train_base=False, expert_width=expert_width
    )
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)

    def factor(*shape, scale):
        values = torch.randn(*shape, device="cuda", generator=generator) * scale
        return values.requires_grad_()

    factors = (
        factor(EXPERTS, RANK, WIDTH, scale=WIDTH**-0.5),
        factor(EXPERTS, 2 * expert_width, RANK, scale=RANK**-0.5),
        factor(EXPERTS, RANK, expert_width, scale=expert_width**-0.5),
        factor(EXPERTS, WIDTH, RANK, scale=RANK**-0.5),
    )
    return (hidden, residual, router, w13, w2, *factors), bias


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("expert_width", [EXPERT_WIDTH, 128])
def test_a_block_base_matches_the_bf16_base_under_lora(mode, expert_width):
    """At expert width 128 the LoRA up projection runs the fused SwiGLU GEMM."""

    def run(implementation, block_weights):
        primals, bias = _lora_inputs(
            5, block_weights=block_weights, expert_width=expert_width
        )
        output, aux, *_ = expert_lora(
            *primals,
            top_k=TOP_K,
            routing_mode=mode,
            scale=0.5,
            implementation=implementation,
            swiglu_limit=4.0,
            **_routing(mode, bias),
        )
        torch.manual_seed(5)
        loss = (output.float() * torch.randn_like(output)).sum() + 0.3 * aux
        trainable = [value for value in primals if value.requires_grad]
        return (output, *torch.autograd.grad(loss, trainable))

    actual = run("sonicmoe", True)
    expected = run("builtin", False)
    for value, reference in zip(actual, expected, strict=True):
        assert _relative(value, reference) < 8e-2
