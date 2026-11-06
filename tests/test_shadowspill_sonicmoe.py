"""Optional engine integration: actual capture, destination paths and refusals."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

pytest.importorskip("shadowspill.pytorch")

from shadowspill.pytorch import AxisFrom, ChunkLocal
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.gradients import (
    accumulate_gradient_outputs,
)
from test_sonicmoe_gradient_destinations import arguments

from mlops.integrations.shadowspill import (
    sonicmoe_chunk_rules,
    sonicmoe_gradient_destination_rules,
)
from mlops.providers.sonicmoe import lora


def capture(direction, dtype, transform="canonical"):
    args, _destinations = arguments(direction, dtype)
    operation = getattr(lora, f"_{direction}_backward")
    positions = [i for i, value in enumerate(args) if isinstance(value, torch.Tensor)]
    tensors = tuple(args[i] for i in positions)

    def function(*values):
        supplied = list(args)
        for index, value in zip(positions, values, strict=True):
            supplied[index] = value
        result = operation(*supplied)
        a, b = result[2:4]
        if transform == "canonical" and direction == "up":
            experts, packed, rank = b.shape
            b = b.view(experts, packed // 2, 2, rank).transpose(1, 2).reshape_as(b)
        elif transform == "cast":
            b = b.to(torch.bfloat16).to(dtype)
        elif transform == "crop":
            b = b[:, :, :-1]
        elif transform == "two_scales":
            b = b * 2.0
        elif transform not in ("identity", "canonical"):
            raise AssertionError(transform)
        return a, b * 1.75

    graph = make_fx(function, tracing_mode="fake")(*tensors)
    return GraphArtifact.capture(
        kind="backward", graph_module=graph, example_inputs=tensors
    )


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("permission", [False, True])
def test_actual_operator_groups_destinations_and_respects_narrow_permission(
    direction, dtype, permission
):
    initialized = torch.cuda.is_initialized()
    original = capture(direction, dtype)
    code = original.graph_module.code
    rules = sonicmoe_gradient_destination_rules()
    transformed = accumulate_gradient_outputs(
        original,
        (0, 1),
        gradient_destination_rules=rules,
        fuse_narrow_gradient_accumulation=permission,
    )
    calls = [
        node
        for node in transformed.graph_module.graph.nodes
        if str(node.target) == rules[direction == "down"].replacement
    ]
    eligible = dtype == torch.float32 or permission
    assert len(calls) == int(eligible)
    if eligible:
        node = calls[0]
        named = {
            argument.name: value
            for argument, value in zip(node.target._schema.arguments, node.args)
        }
        named.update(node.kwargs)
        assert named["output_indices"] == [2, 3]
        assert named["gradient_scales"] == [1.0, 1.75]
        assert named["canonical_b"] == (direction == "up")
        assert named["fuse_narrow_gradient_accumulation"] == permission
        assert named["contract_version"] == 1
        assert len(named["destinations"]) == 2
    assert original.graph_module.code == code
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("transform", ["cast", "crop", "two_scales"])
def test_unsupported_b_path_preserves_a_fusion_and_b_fallback(direction, transform):
    original = capture(direction, torch.float32, transform)
    rules = sonicmoe_gradient_destination_rules()
    transformed = accumulate_gradient_outputs(
        original,
        (0, 1),
        gradient_destination_rules=rules,
        fuse_narrow_gradient_accumulation=True,
    )
    calls = [
        node
        for node in transformed.graph_module.graph.nodes
        if str(node.target) == rules[direction == "down"].replacement
    ]
    assert len(calls) == 1
    node = calls[0]
    index = next(
        i
        for i, argument in enumerate(node.target._schema.arguments)
        if argument.name == "output_indices"
    )
    assert node.args[index] == [2]
    # Its unconsumed B still has the original shape; it is neither silently
    # dropped nor rounded at the destination instead of at the explicit cast.
    assert node.meta["val"][3].numel() > 0
    assert "aten.add_.Tensor" in transformed.operator_targets


def test_chunk_rules_keep_sorted_rows_local_and_match_mutating_outputs():
    rules = {rule.target: rule for rule in sonicmoe_chunk_rules()}
    assert len(rules) == 7
    for direction in ("up", "down"):
        fresh = rules[f"mlops.lora_sonicmoe_{direction}_bwd.default"]
        accumulating = rules[f"mlops.lora_sonicmoe_{direction}_bwd_accumulate.default"]
        assert fresh.output_axes == accumulating.output_axes
        assert fresh.reductions == accumulating.reductions == (1, 2, 3)
    metadata = rules["mlops.sonicmoe_routing_metadata.default"]
    assert metadata.output_axes == (None, ChunkLocal(0), ChunkLocal(0), ChunkLocal(0))
    assert metadata.reductions == (0,)
    assert rules["mlops.lora_sonicmoe_up_bwd.default"].output_axes[0] == AxisFrom(1)


def test_import_needs_no_shadowspill_and_factories_do_not_initialize_cuda():
    source = Path(__file__).parents[1] / "src"
    script = """
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'shadowspill' or fullname.startswith('shadowspill.'):
            raise ModuleNotFoundError('engine intentionally unavailable')
sys.meta_path.insert(0, NoEngine())
import mlops.integrations.shadowspill as integration
import torch
assert not torch.cuda.is_initialized()
assert not any(x.startswith('shadowspill') for x in sys.modules)
for factory in (integration.sonicmoe_chunk_rules, integration.sonicmoe_gradient_destination_rules):
    try:
        factory()
    except ModuleNotFoundError as error:
        assert 'engine intentionally unavailable' in str(error)
    else:
        raise AssertionError('factory unexpectedly resolved missing engine')
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", script, str(source)], check=True, timeout=60)
