"""Portable module geometry never creates a communicator or GPU payload."""

import pytest
import torch

from mlops.expert_parallel import LoRAConfig, QuackMoE, QuackMoEConfig, QuackMoELoRA
from mlops.expert_parallel.quack.config import _configuration, config_signature
from mlops.expert_parallel.quack.registry import (
    _RUNTIMES,
    _register_runtime,
    resource_handle,
)


@pytest.mark.parametrize("precision", ["bf16", "fp8_current"])
@pytest.mark.parametrize("gradient", [torch.bfloat16, torch.float32])
def test_portable_expert_geometry_is_resource_independent(
    monkeypatch, precision, gradient
):
    def forbidden(*args, **kwargs):
        raise AssertionError("portable construction touched CUDA or distributed")

    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.distributed, "new_group", forbidden)
    c = QuackMoEConfig(
        ep_size=2,
        num_experts=8,
        top_k=2,
        model_dim=256,
        expert_hidden_dim=128,
        tokens_per_rank=32,
        compute_precision=precision,
        weight_grad_dtype=gradient,
    )
    before = dict(_RUNTIMES)
    a = QuackMoE.from_resource(c, "model.layer.0", device="meta")
    b = QuackMoE.from_resource(
        _configuration(config_signature(c)), "model.layer.0", device="meta"
    )
    assert a._handle == b._handle == resource_handle("model.layer.0") < 0
    assert a._handle != resource_handle("model.layer.1")
    assert a.gate_up_weight.shape == (4, 256, 256)
    assert a.down_weight.shape == (4, 256, 128)
    assert a.gate_up_weight.dtype == gradient
    assert all(p.is_meta for p in a.parameters())
    assert _RUNTIMES == before
    a.to(device="meta")
    a.close()
    b.close()
    assert _RUNTIMES == before


def test_stable_resource_binding_rejects_duplicate_open():
    runtime = object()
    handle = _register_runtime(runtime, name="test.resources.duplicate")
    try:
        assert _RUNTIMES[handle] is runtime
        with pytest.raises(ValueError, match="already open"):
            _register_runtime(object(), name="test.resources.duplicate")
    finally:
        _RUNTIMES.pop(handle)


@pytest.mark.parametrize("precision", ["bf16", "fp8_current"])
@pytest.mark.parametrize("gradient", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_portable_lora_owns_factors_and_preserves_its_runtime_contract(
    monkeypatch,
    precision,
    gradient,
    device,
):
    from mlops.expert_parallel.quack.resources import _settings

    def forbidden(*args, **kwargs):
        raise AssertionError("portable LoRA touched CUDA or distributed")

    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.distributed, "new_group", forbidden)
    config = QuackMoEConfig(
        ep_size=2,
        num_experts=8,
        top_k=2,
        model_dim=256,
        expert_hidden_dim=128,
        tokens_per_rank=32,
        compute_precision=precision,
        weight_grad_dtype=torch.float32,
        share_expert_banks=True,
        num_shared_experts=1,
    )
    lora = LoRAConfig(rank=16, alpha=8, gradient_dtype=gradient)
    before = dict(_RUNTIMES)
    a = QuackMoELoRA.from_resource(config, "test.lora.0", lora=lora, device=device)
    b = QuackMoELoRA.from_resource(config, "test.lora.1", lora=lora, device=device)
    assert a._handle == resource_handle("test.lora.0") < 0
    assert a._handle != b._handle
    assert _settings(a.resource_spec) == (config, lora)
    assert _settings(config_signature(config)) == (config, None)
    assert [tuple(p.shape) for p in a.lora_parameters()] == [
        (4, 16, 256),
        (4, 256, 16),
        (4, 16, 128),
        (4, 256, 16),
    ]
    factors = set(map(id, a.lora_parameters()))
    for p in a.parameters():
        assert p.device.type == device
        assert p.requires_grad == (id(p) in factors)
    for p in a.lora_parameters():
        assert p.dtype == gradient
        assert getattr(p, "_data", p).dtype == torch.bfloat16
    if device == "cpu":
        for name in ("lora_gate_up_a", "lora_down_a"):
            assert torch.count_nonzero(
                getattr(getattr(a, name), "_data", getattr(a, name))
            )
        for name in ("lora_gate_up_b", "lora_down_b"):
            p, other = getattr(a, name), getattr(b, name)
            data, independent = getattr(p, "_data", p), getattr(other, "_data", other)
            assert torch.count_nonzero(data) == 0
            assert (
                data.untyped_storage().data_ptr()
                != independent.untyped_storage().data_ptr()
            )
    a.close()
    b.close()
    assert _RUNTIMES == before
