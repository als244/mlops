"""Single-rank contracts for the optional distributed AdamW runtime."""

from __future__ import annotations

import torch
import torch.distributed as dist
import pytest

from mlops.optim import AdamW


def _parameters():
    torch.manual_seed(17001)
    return [
        torch.nn.Parameter(
            torch.randn(257, device="cuda", dtype=torch.bfloat16)
        ),
        torch.nn.Parameter(
            torch.randn(113, device="cuda", dtype=torch.bfloat16)
        ),
    ]


def _exercise_world_one(
    opt_state_strategy: str,
    replica_group,
) -> None:
    reference = _parameters()
    actual = [torch.nn.Parameter(value.detach().clone()) for value in reference]
    common = {
        "lr": 3e-4,
        "betas": (0.9, 0.95),
        "weight_decay": 0.03,
        "gradient_dtype": torch.bfloat16,
        "reduction_dtype": torch.float32,
        "state_dtype": torch.float32,
        "master_parameter_dtype": torch.float32,
    }
    reference_optimizer = AdamW(reference, **common)
    actual_optimizer = AdamW(
        actual,
        **common,
        replica_group=replica_group,
        opt_state_strategy=opt_state_strategy,
        bucket_bytes=256,
    )
    assert (
        actual_optimizer.distributed_spec.opt_state_strategy
        == opt_state_strategy
    )

    for step in range(4):
        torch.manual_seed(17100 + step)
        for expected, value in zip(reference, actual, strict=True):
            gradient = torch.randn_like(expected)
            expected.grad = gradient
            value.grad = gradient.clone()
        reference_optimizer.step()
        actual_optimizer.step()

    actual_optimizer.synchronize()
    assert all(
        torch.equal(expected, value)
        for expected, value in zip(reference, actual, strict=True)
    )
    manifest = actual_optimizer.execution_manifest()
    assert manifest["spec"]["opt_state_strategy"] == opt_state_strategy
    assert manifest["spec"]["world_size"] == 1
    # The bucket cap bounds each full collective buffer. Mixed-dtype packing
    # and sharded output add bounded companion buffers, all reused serially.
    assert manifest["workspace_bytes"] <= 3 * 256
    assert all(
        bucket["workspace_bytes"] <= 3 * 256
        for bucket in manifest["buckets"]
    )
    assert len(manifest["buckets"]) > 2
    assert manifest["buckets"]

    checkpoint = actual_optimizer.state_dict()
    actual_optimizer.load_state_dict(checkpoint)
    legacy = {
        **checkpoint,
        "mlops_distributed": {
            **checkpoint["mlops_distributed"],
            "spec": {
                **checkpoint["mlops_distributed"]["spec"],
                "schema_version": "mlops.distributed_adamw.v1",
            },
        },
    }
    with pytest.raises(ValueError, match="schema_version mismatch"):
        actual_optimizer.load_state_dict(legacy)

    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    for _ in range(10):
        actual_optimizer.step()
    actual_optimizer.synchronize()
    torch.cuda.synchronize()
    # Retiring ProcessGroup/allocator bookkeeping is allowed to release memory;
    # the steady-state contract is that repeated steps cannot grow live tensor
    # allocation.
    assert torch.cuda.memory_allocated() <= allocated_before


@torch.no_grad()
def test_distributed_strategies_preserve_world_one_semantics_and_cleanup(tmp_path):
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        pytest.skip("requires CUDA and NCCL")
    assert not dist.is_initialized()
    initialized = False
    try:
        dist.init_process_group(
            "nccl",
            init_method=f"file://{tmp_path / 'nccl_store'}",
            rank=0,
            world_size=1,
        )
        initialized = True
        for opt_state_strategy in ("replicated", "sharded"):
            _exercise_world_one(
                opt_state_strategy,
                dist.group.WORLD,
            )
    finally:
        if initialized:
            dist.destroy_process_group()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@torch.no_grad()
def test_gloo_uses_portable_host_completion(tmp_path):
    if not torch.cuda.is_available() or not dist.is_gloo_available():
        pytest.skip("requires CUDA and Gloo")
    assert not dist.is_initialized()
    initialized = False
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{tmp_path / 'gloo_store'}",
            rank=0,
            world_size=1,
        )
        initialized = True
        for opt_state_strategy in ("replicated", "sharded"):
            _exercise_world_one(opt_state_strategy, dist.group.WORLD)
        optimizer = AdamW(
            _parameters(),
            replica_group=dist.group.WORLD,
        )
        assert optimizer.distributed_spec.collective_backend == "gloo"
        assert optimizer.distributed_spec.completion_mode == "host_wait"
    finally:
        if initialized:
            dist.destroy_process_group()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def test_distributed_constructor_requires_initialized_replica_group():
    if not torch.cuda.is_available() or dist.is_initialized():
        pytest.skip("requires CUDA without an initialized default group")
    parameter = torch.nn.Parameter(
        torch.randn(16, device="cuda", dtype=torch.bfloat16)
    )
    try:
        AdamW([parameter], replica_group=object())
    except RuntimeError as error:
        assert "initialized torch.distributed" in str(error)
    else:  # pragma: no cover - defensive contract check
        raise AssertionError("expected an uninitialized-group error")
