"""Portable module geometry never creates a communicator or GPU payload."""

import pytest
import torch

from mlops.expert_parallel import QuackMoE, QuackMoEConfig
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
