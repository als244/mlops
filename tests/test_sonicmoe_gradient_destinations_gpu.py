"""Actual destination epilogues and backward calls, with independent products.

These checks qualify accumulation, not a complete model/optimizer trajectory.
The product oracle uses CPU float64 on the exact operands received by the GEMM.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

from mlops.providers.sonicmoe import lora
from mlops.providers.sonicmoe.lora_destinations import (
    _Destinations,
    accumulate_down,
    accumulate_up,
)

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def device():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device supported by the SonicMoE provider")
    if torch.cuda.get_device_capability()[0] not in lora.CAPABILITIES:
        pytest.skip("SonicMoE dispatch supports compute capability 9.x and 10.x")
    # Providers install their shared pipeline before the first Quack launch.
    # Direct product checks must obey the same startup order as training.
    from mlops.quack_runtime import initialize

    initialize()


def random(shape, *, dtype=torch.bfloat16, scale=0.125, generator=None):
    return (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(
        device="cuda", dtype=dtype
    )


def product_oracle(left, right, offsets, *, gather=None, transpose=False, concat=None):
    left, right = left.detach().cpu().double(), right.detach().cpu().double()
    if gather is not None:
        left = left[gather.cpu().long()]
    bounds = offsets.cpu().tolist()
    result = []
    for start, stop in pairwise(bounds):
        matrix = left[start:stop].T @ right[start:stop]
        if transpose:
            matrix = matrix.T
        if concat:
            matrix = torch.cat((matrix[::2], matrix[1::2]), dim=0)
        result.append(matrix)
    return torch.stack(result)


def compare(actual, expected, dtype):
    expected = expected.to(dtype)
    # BF16/FP16 permit one output ulp near a rounding boundary. FP32 bounds
    # allow accumulation order, while detecting incorrect layout/prior/scale.
    tolerance = {
        torch.float32: (2e-5, 2e-6),
        torch.bfloat16: (8e-3, 1e-4),
        torch.float16: (1e-3, 1e-5),
    }[dtype]
    torch.testing.assert_close(
        actual.detach().cpu(), expected, rtol=tolerance[0], atol=tolerance[1]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["up_a", "up_b", "up_weight", "down_b"])
def test_product_reads_prior_in_correct_layout_without_fresh_gradient(dtype, layout):
    generator = torch.Generator().manual_seed(50971)
    experts, width, inner, rank, tokens = 4, 64, 64, 32, 83
    offsets = torch.tensor([0, 23, 23, 40, 69], device="cuda", dtype=torch.int32)
    rows = 69
    transpose = layout in ("up_a", "up_weight")
    canonical = layout == "up_b"
    concat = ("out",) if layout == "up_weight" else None
    gather = (
        None
        if canonical
        else torch.randint(
            tokens, (rows,), generator=generator, dtype=torch.int32
        ).cuda()
    )
    left_width = 2 * inner if canonical else width
    right_width = 2 * inner if layout == "up_weight" else rank
    left = random((rows if canonical else tokens, left_width), generator=generator)
    right = random((rows, right_width), generator=generator)
    shape = (
        (experts, right_width, left_width)
        if transpose
        else (experts, left_width, right_width)
    )
    destination = random(shape, dtype=dtype, generator=generator)
    pointer = destination.data_ptr()
    prior = destination.cpu().double()
    index = 3 if layout in ("up_b", "down_b") else (1 if concat else 2)
    product = product_oracle(
        left,
        right,
        offsets,
        gather=gather,
        transpose=transpose,
        concat=("out",) if canonical else concat,
    )
    for scale in (1.75, -0.25):
        targets = _Destinations(
            [destination],
            [index],
            1,
            [scale],
            canonical,
            True,
            {index: shape},
            dtype,
            destination.device,
            up=layout != "down_b",
            weight_components=[],
        )
        result = targets.product(
            index,
            shape,
            dtype,
            destination.device,
            left,
            right,
            offsets,
            gather=gather,
            concat=concat,
            transpose=transpose,
        )
        torch.cuda.synchronize()
        assert result.numel() == 0
        assert destination.data_ptr() == pointer
        compare(destination, prior + scale * product, dtype)
        # The empty expert must retain its previous contents exactly.
        assert torch.equal(destination[1].cpu(), prior[1].to(dtype))
        prior = destination.cpu().double()


def backward_arguments(direction, dtype, train_base):
    generator = torch.Generator().manual_seed(97213)
    experts, tokens, top_k, width, inner, rank = 4, 65, 2, 64, 64, 32
    rows = tokens * top_k
    ids = (torch.arange(tokens)[:, None] + torch.arange(top_k)) % (experts - 1)
    offsets, gather, scatter, reverse = lora.routing_metadata(ids.int().cuda(), experts)
    rand = lambda shape: random(shape, generator=generator)
    a = rand((experts, rank, width if direction == "up" else inner))
    b = rand((experts, 2 * inner if direction == "up" else width, rank))
    weight = rand(
        (experts, 2 * inner, width) if direction == "up" else (experts, width, inner)
    )
    weight_shape = list(weight.shape) if train_base else [0, 0, 0]
    z = rand((rows, rank))
    if direction == "up":
        args = (
            rand((rows, 2 * inner)),
            rand((tokens, width)),
            gather,
            reverse,
            weight,
            a,
            b,
            z,
            offsets,
            weight_shape,
            list(a.shape),
            list(b.shape),
            top_k,
            dtype,
            [],
        )
    else:
        args = (
            rand((tokens, width)),
            rand((rows, 2 * inner)),
            weight,
            a,
            b,
            z,
            offsets,
            gather,
            scatter,
            random((rows,), dtype=torch.float32, generator=generator),
            weight_shape,
            list(a.shape),
            list(b.shape),
            dtype,
            None,
            [],
        )
    return args, generator


@pytest.mark.parametrize(
    "direction,dtype,selected",
    [
        # Partial destinations cover both directions, dtypes and LoRA factors
        # pairwise. Full destinations retain every direction/dtype combination
        # because the base-weight product has its own kernel layout.
        ("up", torch.float32, (2,)),
        ("down", torch.bfloat16, (2,)),
        ("up", torch.bfloat16, (3,)),
        ("down", torch.float32, (3,)),
        ("up", torch.float32, (1, 2, 3)),
        ("up", torch.bfloat16, (1, 2, 3)),
        ("down", torch.float32, (1, 2, 3)),
        ("down", torch.bfloat16, (1, 2, 3)),
    ],
)
def test_actual_backward_calls_each_product_once_and_preserves_unconsumed_outputs(
    direction,
    dtype,
    selected,
    monkeypatch,
):
    train_base = 1 in selected
    args, generator = backward_arguments(direction, dtype, train_base)
    original = getattr(lora, f"_{direction}_backward")(*args)
    destinations = [
        random(original[i].shape, dtype=dtype, generator=generator) for i in selected
    ]
    pointers = [v.data_ptr() for v in destinations]
    prior = [v.cpu().double() for v in destinations]
    products, allocations, calls, gemm_calls = {}, [], [], []
    real_product, real_empty = lora._parameter_product, lora._empty

    def observed_product(
        targets,
        index,
        shape,
        output_dtype,
        device,
        left,
        right,
        offsets,
        **options,
    ):
        # An unselected output delegates through the fresh helper. Count one
        # logical product here, then independently count actual GEMMs below.
        if targets is not None:
            calls.append(index)
        if targets is not None and index in selected:
            oracle_options = dict(options)
            if direction == "up" and index == 3:
                oracle_options["concat"] = ("out",)
            products[index] = product_oracle(left, right, offsets, **oracle_options)
        return real_product(
            targets,
            index,
            shape,
            output_dtype,
            device,
            left,
            right,
            offsets,
            **options,
        )

    def observed_empty(shape, output_dtype, device):
        allocations.append(tuple(shape))
        return real_empty(shape, output_dtype, device)

    monkeypatch.setattr(lora, "_parameter_product", observed_product)
    monkeypatch.setattr(lora, "_empty", observed_empty)
    from quack import gemm_interface

    def observe_gemm(function):
        def measured(*values, **keywords):
            if keywords.get("cu_seqlens_k") is not None:
                gemm_calls.append(keywords["out"].data_ptr())
            return function(*values, **keywords)

        return measured

    monkeypatch.setattr(gemm_interface, "gemm", observe_gemm(gemm_interface.gemm))
    monkeypatch.setattr(
        gemm_interface, "gemm_add_out", observe_gemm(gemm_interface.gemm_add_out)
    )
    accumulate = accumulate_up if direction == "up" else accumulate_down
    result = accumulate(
        *args,
        destinations,
        list(selected),
        1,
        gradient_scales=[1.75] * len(selected),
        canonical_b=direction == "up" and 3 in selected,
        fuse_narrow_gradient_accumulation=True,
    )
    torch.cuda.synchronize()
    assert calls == ([1, 2, 3] if train_base else [2, 3])
    assert len(gemm_calls) == len(calls)
    assert all(gemm_calls.count(pointer) == 1 for pointer in pointers)
    assert len(allocations) == len(calls) - len(selected)
    assert [v.data_ptr() for v in destinations] == pointers
    for index, before, output in zip(selected, prior, destinations, strict=True):
        assert result[index].numel() == 0
        compare(output, before + 1.75 * products[index], dtype)
        assert torch.equal(output[-1].cpu(), before[-1].to(dtype))
    for index, expected in enumerate(original):
        if index not in selected:
            if expected is None:
                assert result[index].numel() == 0
            else:
                torch.testing.assert_close(result[index], expected, rtol=0, atol=0)
