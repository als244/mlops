"""Two-rank numerical canary for ``mlops.optim.AdamW``.

Launch this file with one torchrun worker per host. It is intentionally
separate from ordinary pytest because it requires a real multi-host NCCL
group.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch
import torch.distributed as dist

from mlops.optim import AdamW


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.detach().float().flatten()
    expected = expected.detach().float().flatten()
    difference = actual - expected
    return {
        "cosine": float(
            torch.nn.functional.cosine_similarity(actual, expected, dim=0)
        ),
        "relative_l2": float(
            difference.norm() / expected.norm().clamp_min(1e-30)
        ),
        "sign_agreement": float(
            (torch.signbit(actual) == torch.signbit(expected)).float().mean()
        ),
        "max_abs": float(difference.abs().max()),
    }


def _assert_metrics(metrics: dict[str, float]) -> None:
    assert metrics["cosine"] >= 0.999
    assert metrics["relative_l2"] <= 0.001
    assert metrics["sign_agreement"] >= 0.999


def _parameters(seed: int) -> list[torch.nn.Parameter]:
    torch.manual_seed(seed)
    return [
        torch.nn.Parameter(
            torch.randn(4099, device="cuda", dtype=torch.bfloat16)
        ),
        torch.nn.Parameter(
            torch.randn(2053, device="cuda", dtype=torch.float32)
        ),
    ]


def _full_reference_state(
    optimizer: AdamW,
    parameters: list[torch.nn.Parameter],
    name: str,
) -> torch.Tensor:
    return torch.cat(
        [
            (
                optimizer.state[parameter].get("master_parameter", parameter)
                if name == "master_parameter"
                else optimizer.state[parameter][name]
            )
            .float()
            .reshape(-1)
            for parameter in parameters
        ]
    )


def _distributed_state(
    optimizer: AdamW,
    name: str,
) -> list[torch.Tensor]:
    values = []
    for bucket in optimizer._distributed.buckets:
        if name == "master_parameter":
            value = (
                bucket.master_parameter
                if bucket.master_parameter is not None
                else bucket.update_parameter
            )
        else:
            value = getattr(bucket, name)
        if optimizer.opt_state_strategy == "replicated":
            values.append(value[: bucket.element_count].float())
            continue
        gathered = torch.empty(
            bucket.padded_element_count,
            device=value.device,
            dtype=value.dtype,
        )
        dist.all_gather_single(gathered, value, group=optimizer.replica_group)
        values.append(gathered[: bucket.element_count].float())
    return values


def _run_case(
    *,
    opt_state_strategy: str,
    reduction: str,
    steps: int,
    rank: int,
) -> dict[str, object]:
    reference = _parameters(8100)
    actual = [torch.nn.Parameter(value.detach().clone()) for value in reference]
    frozen = torch.nn.Parameter(
        torch.randn(127, device="cuda", dtype=torch.bfloat16),
        requires_grad=False,
    )
    frozen_before = frozen.detach().clone()
    for parameter in reference:
        parameter.grad_dtype = torch.float32
    for parameter in actual:
        parameter.grad_dtype = torch.bfloat16
    reference_optimizer = AdamW(
        [
            {
                "params": reference[:1],
                "state_dtype": torch.float32,
                "master_parameter_dtype": torch.float32,
            },
            {
                "params": reference[1:],
                "state_dtype": torch.bfloat16,
                "master_parameter_dtype": "parameter",
            },
        ],
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.03,
        gradient_dtype=torch.float32,
    )
    actual_optimizer = AdamW(
        [
            {
                "params": actual[:1],
                "state_dtype": torch.float32,
                "master_parameter_dtype": torch.float32,
            },
            {
                "params": [*actual[1:], frozen],
                "state_dtype": torch.bfloat16,
                "master_parameter_dtype": "parameter",
            },
        ],
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.03,
        gradient_dtype=torch.bfloat16,
        reduction_dtype=torch.float32,
        replica_group=dist.group.WORLD,
        opt_state_strategy=opt_state_strategy,
        gradient_reduction=reduction,
        bucket_bytes=1 << 14,
    )

    started = time.perf_counter()
    for step in range(steps):
        for index, (expected, value) in enumerate(
            zip(reference, actual, strict=True)
        ):
            torch.manual_seed(9000 + 101 * step + 17 * rank + index)
            local_gradient = torch.randn(
                value.shape,
                device=value.device,
                dtype=torch.bfloat16,
            )
            reduced_gradient = local_gradient.float()
            dist.all_reduce(reduced_gradient, group=dist.group.WORLD)
            if reduction == "mean":
                reduced_gradient /= dist.get_world_size()
            expected.grad = reduced_gradient
            value.grad = local_gradient
        reference_optimizer.step()
        actual_optimizer.step()
    actual_optimizer.synchronize()
    torch.cuda.synchronize()
    duration = time.perf_counter() - started

    parameter_metrics = [
        _metrics(value, expected)
        for value, expected in zip(actual, reference, strict=True)
    ]
    for result in parameter_metrics:
        _assert_metrics(result)
    assert torch.equal(frozen, frozen_before)
    assert frozen not in actual_optimizer.state

    moment_metrics = {}
    for name in ("exp_avg", "exp_avg_sq", "master_parameter"):
        expected = _full_reference_state(reference_optimizer, reference, name)
        actual_buckets = _distributed_state(actual_optimizer, name)
        actual_state = torch.cat(actual_buckets)
        result = _metrics(actual_state, expected)
        _assert_metrics(result)
        moment_metrics[name] = result

    checkpoint = copy.deepcopy(actual_optimizer.state_dict())
    saved_means = [
        bucket.exp_avg.clone() for bucket in actual_optimizer._distributed.buckets
    ]
    for bucket in actual_optimizer._distributed.buckets:
        bucket.exp_avg.zero_()
    actual_optimizer.load_state_dict(checkpoint)
    assert all(
        torch.equal(bucket.exp_avg, saved)
        for bucket, saved in zip(
            actual_optimizer._distributed.buckets,
            saved_means,
            strict=True,
        )
    )

    return {
        "opt_state_strategy": opt_state_strategy,
        "reduction": reduction,
        "steps": steps,
        "duration_s": duration,
        "parameter_metrics": parameter_metrics,
        "state_metrics": moment_metrics,
        "execution_manifest": actual_optimizer.execution_manifest(),
    }


def _run_local_slice_canary(rank: int) -> dict[str, object]:
    shared_reference = _parameters(8300)[:1]
    shared_actual = [
        torch.nn.Parameter(shared_reference[0].detach().clone())
    ]
    local_parameter = torch.nn.Parameter(
        torch.full(
            (31 + rank,),
            float(rank + 1),
            device="cuda",
            dtype=torch.bfloat16,
        )
    )
    shared_reference[0].grad_dtype = torch.float32
    reference_optimizer = AdamW(
        shared_reference,
        lr=1e-3,
        gradient_dtype=torch.float32,
        state_dtype=torch.float32,
        master_parameter_dtype=torch.float32,
    )
    shared_optimizer = AdamW(
        shared_actual,
        lr=1e-3,
        gradient_dtype=torch.bfloat16,
        reduction_dtype=torch.float32,
        state_dtype=torch.float32,
        master_parameter_dtype=torch.float32,
        replica_group=dist.group.WORLD,
    )
    # Use an update larger than one BF16 ULP so the local-slice assertion is
    # about optimizer ownership rather than storage rounding.
    local_optimizer = AdamW([local_parameter], lr=0.1)
    local_before = local_parameter.detach().clone()
    torch.manual_seed(8400 + rank)
    local_gradient = torch.randn_like(shared_actual[0])
    reduced_gradient = local_gradient.float()
    dist.all_reduce(reduced_gradient)
    reduced_gradient /= dist.get_world_size()
    shared_reference[0].grad = reduced_gradient
    shared_actual[0].grad = local_gradient
    local_parameter.grad = torch.full_like(local_parameter, float(rank + 1))
    reference_optimizer.step()
    shared_optimizer.step()
    local_optimizer.step()
    shared_optimizer.synchronize()
    result = _metrics(shared_actual[0], shared_reference[0])
    _assert_metrics(result)
    assert not torch.equal(local_parameter, local_before)
    return {
        "shared_metrics": result,
        "local_parameter_elements": local_parameter.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if arguments.backend == "nccl":
        dist.init_process_group(
            "nccl",
            device_id=torch.device("cuda", local_rank),
        )
    else:
        dist.init_process_group("gloo")
    rank = dist.get_rank()
    try:
        cases = [
            _run_case(
                opt_state_strategy=opt_state_strategy,
                reduction=reduction,
                steps=arguments.steps,
                rank=rank,
            )
            for opt_state_strategy in ("replicated", "sharded")
            for reduction in ("mean", "sum")
        ]
        local_slice = _run_local_slice_canary(rank)
        result = {
            "world_size": dist.get_world_size(),
            "backend": arguments.backend,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": (
                torch.cuda.nccl.version()
                if arguments.backend == "nccl"
                else None
            ),
            "devices": [None] * dist.get_world_size(),
            "cases": cases,
            "local_slice": local_slice,
        }
        dist.all_gather_object(
            result["devices"],
            torch.cuda.get_device_name(),
        )
        if rank == 0:
            rendered = json.dumps(result, indent=2, sort_keys=True)
            if arguments.output:
                with open(arguments.output, "w", encoding="utf-8") as handle:
                    handle.write(rendered + "\n")
            print(rendered, flush=True)
    finally:
        dist.barrier()
        dist.destroy_process_group()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
