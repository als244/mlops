"""Destination schemas and geometry; real kernels have separate GPU checks."""

import sys
from types import ModuleType

import pytest
import torch
from torch._functorch.aot_autograd import aot_function
from torch._higher_order_ops.auto_functionalize import can_auto_functionalize
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from mlops.providers.sonicmoe.lora_destinations import (
    _Destinations,
    accumulate_down,
    accumulate_up,
)


def arguments(direction, dtype=torch.float32, *, train_base=False):
    experts, tokens, top_k, width, inner, rank = 4, 5, 2, 16, 8, 8
    rows = tokens * top_k
    tensor = lambda *shape: torch.empty(shape, dtype=torch.bfloat16)
    offsets = torch.empty(experts + 1, dtype=torch.int32)
    gather = torch.empty(rows, dtype=torch.int32)
    scatter = torch.empty(rows, dtype=torch.int32)
    reverse = torch.empty(rows, dtype=torch.int32)
    a = tensor(experts, rank, width if direction == "up" else inner)
    b = tensor(experts, 2 * inner if direction == "up" else width, rank)
    weight = tensor(
        experts, *([2 * inner, width] if direction == "up" else [width, inner])
    )
    weight_shape = list(weight.shape) if train_base else [0, 0, 0]
    z = tensor(rows, rank)
    if direction == "up":
        args = (
            tensor(rows, 2 * inner),
            tensor(tokens, width),
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
            tensor(tokens, width),
            tensor(rows, 2 * inner),
            weight,
            a,
            b,
            z,
            offsets,
            gather,
            scatter,
            torch.empty(rows),
            weight_shape,
            list(a.shape),
            list(b.shape),
            dtype,
            None,
            [],
        )
    destinations = [torch.empty_like(a, dtype=dtype), torch.empty_like(b, dtype=dtype)]
    return args, destinations


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("train_base", [False, True])
def test_fixed_returns_support_fake_capture_and_aot_functionalization(
    direction, dtype, train_base
):
    args, destinations = arguments(direction, dtype, train_base=train_base)
    operation = accumulate_up if direction == "up" else accumulate_down
    overload = getattr(
        torch.ops.mlops, f"lora_sonicmoe_{direction}_bwd_accumulate"
    ).default
    assert can_auto_functionalize(overload)

    def function(*values):
        original, supplied = values[:-2], list(values[-2:])
        outputs = operation(
            *original,
            supplied,
            [2, 3],
            1,
            gradient_scales=[1.0, 1.75],
            canonical_b=direction == "up",
            fuse_narrow_gradient_accumulation=True,
        )
        return outputs, supplied

    graph = make_fx(function, tracing_mode="fake")(*args, *destinations)
    node = next(n for n in graph.graph.nodes if n.target == overload)
    outputs = node.meta["val"]
    assert outputs[2].numel() == outputs[3].numel() == 0
    assert outputs[1].numel() > 0 if train_base else outputs[1].numel() == 0
    assert outputs[0].numel() > 0
    if direction == "down":
        assert outputs[4].shape == args[9].shape
    # Use the compiler's auto-functionalization path. Standalone
    # torch.func.functionalize does not support mutable custom operators.
    positions = [i for i, value in enumerate(args) if isinstance(value, torch.Tensor)]
    compiled_graphs = []

    def tensors_only(*values):
        original = list(args)
        for index, value in zip(positions, values[:-2], strict=True):
            original[index] = value
        return function(*original, *values[-2:])

    def compiler(graph, _example):
        compiled_graphs.append(graph)
        return graph

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode:
        values = tuple(
            mode.from_tensor(value)
            for value in (
                *[args[i] for i in positions],
                *destinations,
            )
        )
        aot_function(tensors_only, fw_compiler=compiler)(*values)
    assert len(compiled_graphs) == 1
    compiled_graphs[0].graph.lint()
    assert any(
        str(node.target) == "auto_functionalized_v2"
        for node in compiled_graphs[0].graph.nodes
    )


def target(destinations=None, indices=None, **changes):
    shape = (4, 8, 16)
    supplied = [torch.empty(shape)] if destinations is None else destinations
    values = {
        "destinations": supplied,
        "output_indices": [2] if indices is None else indices,
        "contract_version": 1,
        "gradient_scales": None,
        "canonical_b": False,
        "fuse_narrow_gradient_accumulation": False,
        "shapes": {2: shape, 3: shape},
        "dtype": torch.float32,
        "device": torch.device("cpu"),
        "up": True,
        "weight_components": [],
    }
    values.update(changes)
    return _Destinations(**values)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"contract_version": 2}, "contract"),
        ({"output_indices": [1]}, "inactive"),
        ({"output_indices": [2, 2]}, "unique"),
        ({"gradient_scales": [float("inf")]}, "finite"),
        ({"gradient_scales": []}, "finite"),
        ({"canonical_b": True}, "requires the B"),
        ({"weight_components": [torch.empty(1)]}, "BF16 base"),
    ],
)
def test_invalid_destination_contracts_fail_before_kernels(changes, message):
    with pytest.raises(ValueError, match=message):
        target(**changes)


def test_narrow_rounding_and_cropped_rank_require_separate_permission():
    with pytest.raises(ValueError, match="fused-rounding"):
        target(dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="aligned gradient shape"):
        target([torch.empty(4, 3, 16)])
    with pytest.raises(ValueError, match="dtype/device"):
        target([torch.empty(4, 8, 16, dtype=torch.float64)])


def test_overlapping_destinations_refuse_but_disjoint_arena_slices_are_valid():
    backing = torch.empty(1024)
    first, second = backing[:512].view(4, 8, 16), backing[512:].view(4, 8, 16)
    target([first, second], [2, 3])
    with pytest.raises(ValueError, match="overlap"):
        target([first, first], [2, 3])


@pytest.mark.parametrize(
    "canonical,transpose", [(False, False), (True, False), (False, True)]
)
def test_product_uses_same_prior_and_output_with_opmath_epilogue(
    monkeypatch,
    canonical,
    transpose,
):
    recorded = []
    interface = ModuleType("quack.gemm_interface")
    interface.gemm_add_out = lambda *args, **kwargs: recorded.append((args, kwargs))
    monkeypatch.setitem(sys.modules, "quack.gemm_interface", interface)
    destination = torch.empty(4, 8, 16)
    index = 3 if canonical else 2
    targets = target(
        [destination], [index], canonical_b=canonical, gradient_scales=[1.75]
    )
    result = targets.product(
        index,
        destination.shape,
        destination.dtype,
        destination.device,
        torch.empty(17, 16 if transpose else 8),
        torch.empty(17, 8 if transpose else 16),
        torch.empty(5, dtype=torch.int32),
        transpose=transpose,
    )
    assert result.numel() == 0 and len(recorded) == 1
    arguments, keywords = recorded[0]
    assert arguments[2] is keywords["out"]
    assert arguments[2].untyped_storage()._cdata == destination.untyped_storage()._cdata
    assert keywords["alpha"] == 1.75 and keywords["beta"] == 1.0
    assert not keywords["add_to_output"]
    assert keywords["concat_layout"] == ("out,C" if canonical else None)
