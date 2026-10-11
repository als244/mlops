"""Compressed checkpoint declarations, bounded reads and component lifetimes."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from mlops.sequential_moe import SequentialMoE, SequentialMoEConfig
from mlops.sequential_moe.precision.checkpoint import HFWeights
from mlops.sequential_moe.precision.safetensors import SafeTensors


def fixture(directory):
    quant = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "fp4": {
                "format": "nvfp4-pack-quantized",
                "targets": ["re:experts\\.0\\..*"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                    "symmetric": True,
                    "strategy": "tensor_group",
                },
            },
            "fp8": {
                "format": "float-quantized",
                "targets": ["re:experts\\.1\\..*"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "block_structure": [128, 128],
                    "symmetric": True,
                },
            },
        },
    }
    (directory / "config.json").write_text(json.dumps({"quantization_config": quant}))
    state = {
        "experts.0.gate_proj.weight_packed": torch.arange(
            256, dtype=torch.uint8
        ).reshape(16, 16),
        "experts.0.gate_proj.weight_scale": torch.ones(16, 2).to(torch.float8_e4m3fn),
        "experts.0.gate_proj.weight_global_scale": torch.tensor([4.0]),
        "experts.1.gate_proj.weight": torch.ones(16, 32).to(torch.float8_e4m3fn),
        "experts.1.gate_proj.weight_scale": torch.tensor([[0.25]]),
        "head.weight": torch.arange(512).reshape(16, 32).to(torch.bfloat16),
    }
    save_file(state, str(directory / "a.safetensors"))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(state, "a.safetensors")})
    )
    return state


def test_mixed_formats_are_resolved_per_projection_without_payload_reads(tmp_path):
    fixture(tmp_path)
    with HFWeights(tmp_path) as source:
        for prefix, expected in (
            ("experts.0.gate_proj", "nvfp4"),
            ("experts.1.gate_proj", "fp8"),
            ("head", "bf16"),
        ):
            values, spec = source.projection(prefix, (16, 32), meta=True)
            assert spec.format == expected
            assert all(t.is_meta for t in values)
        assert not source.source._handles
        values, _ = source.projection("experts.0.gate_proj", (16, 32))
        torch.testing.assert_close(values[-1], torch.tensor([0.25]))


def test_mapped_values_survive_page_release_and_reader_close(tmp_path):
    expected = fixture(tmp_path)
    before = (tmp_path / "a.safetensors").read_bytes()
    with SafeTensors(tmp_path) as source:
        views = []
        for name, value in expected.items():
            actual = source.tensor(name)
            source.release_pages(name)
            torch.testing.assert_close(
                actual.view(torch.uint8), value.view(torch.uint8)
            )
            views.append((actual, value))
        # Independently compare with the official safetensors reader.
        with safe_open(tmp_path / "a.safetensors", framework="pt") as official:
            torch.testing.assert_close(
                source.tensor("head.weight"), official.get_tensor("head.weight")
            )
    for actual, value in views:
        torch.testing.assert_close(actual.view(torch.uint8), value.view(torch.uint8))
    assert (tmp_path / "a.safetensors").read_bytes() == before
    with pytest.raises(RuntimeError, match="closed"):
        source.tensor("head.weight")


def test_bad_shard_path_is_rejected_before_open(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../escape.safetensors"}})
    )
    with pytest.raises(ValueError, match="leaves the checkpoint directory"):
        SafeTensors(tmp_path)


@pytest.mark.parametrize(
    "storage", ["bf16", "fp16", "fp32", "fp8", "nvfp4", "mxfp4", "int4"]
)
def test_meta_experts_allocate_no_weight_payload(storage):
    model = SequentialMoE(
        SequentialMoEConfig(128, 128, 2, 1, weight_storage=storage), device="meta"
    )
    assert all(value.is_meta for value in model.parameters())
    for expert in model.experts:
        assert all(value.is_meta for value in expert.base_tensors())


def test_huggingface_snapshot_symlinks_are_supported(tmp_path):
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "revision"
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    expected = torch.arange(8, dtype=torch.bfloat16)
    save_file({"weight": expected}, str(blobs / "object"))
    (snapshot / "model.safetensors").symlink_to(blobs / "object")
    with SafeTensors(snapshot) as source:
        assert source.tensor("weight", meta=True).is_meta
        torch.testing.assert_close(source.tensor("weight"), expected)
