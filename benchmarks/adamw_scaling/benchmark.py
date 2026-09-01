#!/usr/bin/env python3
"""Measure one two-rank ``mlops.optim.AdamW`` scaling cell.

The launcher deliberately starts a fresh process group for every cell.  This
worker therefore handles exactly one backend, state strategy, and parameter
count.  Rank zero writes the aggregate JSON result.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import socket
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from mlops.optim import AdamW


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument(
        "--opt-state-strategy",
        choices=("sharded", "replicated"),
        required=True,
    )
    parser.add_argument("--parameter-count", type=int, required=True)
    parser.add_argument(
        "--parameter-layout",
        choices=("bucketed", "single"),
        default="bucketed",
    )
    parser.add_argument("--bucket-bytes", type=int, default=64 << 20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a nonempty sample."""
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _device_metadata() -> dict[str, Any]:
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "device_index": device,
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "device_total_bytes": total_bytes,
        "device_free_bytes_before_setup": free_bytes,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nccl_version": list(torch.cuda.nccl.version()),
    }


def _parameter_chunk_elements(bucket_bytes: int) -> int:
    element_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
    return max(1, bucket_bytes // element_bytes)


@torch.no_grad()
def _make_parameters(
    parameter_count: int,
    bucket_bytes: int,
    rank: int,
    parameter_layout: str,
) -> list[torch.nn.Parameter]:
    """Construct exact-size, bucket-shaped parameters and nonzero gradients."""
    chunk_elements = (
        parameter_count
        if parameter_layout == "single"
        else _parameter_chunk_elements(bucket_bytes)
    )
    parameters: list[torch.nn.Parameter] = []
    remaining = parameter_count
    chunk_index = 0
    while remaining:
        count = min(remaining, chunk_elements)
        parameter = torch.nn.Parameter(
            torch.empty(count, device="cuda", dtype=torch.bfloat16)
        )
        # Setup is outside the timed region.  Nonzero values exercise the same
        # optimizer arithmetic and memory traffic as a training update.
        parameter.uniform_(-0.02, 0.02, generator=torch.Generator(device="cuda").manual_seed(8100 + chunk_index))
        gradient = torch.empty_like(parameter)
        gradient.normal_(
            mean=0.0,
            std=0.01,
            generator=torch.Generator(device="cuda").manual_seed(
                9100 + 101 * rank + chunk_index
            ),
        )
        parameter.grad = gradient
        parameters.append(parameter)
        remaining -= count
        chunk_index += 1
    return parameters


def _gather_objects(value: Any, group: Any) -> list[Any]:
    gathered: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, value, group=group)
    return gathered


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _skip_result(
    *,
    arguments: argparse.Namespace,
    rank_reports: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": "mlops.adamw_scaling.v2",
        "status": "skipped",
        "reason": reason,
        "backend": arguments.backend,
        "opt_state_strategy": arguments.opt_state_strategy,
        "parameter_count": arguments.parameter_count,
        "parameter_layout": arguments.parameter_layout,
        "parameter_bytes": arguments.parameter_count * 2,
        "bucket_bytes": arguments.bucket_bytes,
        "rank_reports": rank_reports,
    }


def _run(arguments: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if arguments.backend == "nccl":
        dist.init_process_group("nccl", device_id=device)
        control_group = dist.new_group(backend="gloo")
    else:
        dist.init_process_group("gloo")
        control_group = dist.group.WORLD

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    parameters: list[torch.nn.Parameter] = []
    optimizer: AdamW | None = None
    try:
        metadata = _device_metadata()
        allocation_error = None
        try:
            parameters = _make_parameters(
                arguments.parameter_count,
                arguments.bucket_bytes,
                rank,
                arguments.parameter_layout,
            )
        except (torch.OutOfMemoryError, RuntimeError) as error:
            allocation_error = f"{type(error).__name__}: {error}"
            parameters.clear()
            gc.collect()
            torch.cuda.empty_cache()

        setup_reports = _gather_objects(
            {"rank": rank, "metadata": metadata, "allocation_error": allocation_error},
            control_group,
        )
        if any(report["allocation_error"] for report in setup_reports):
            if rank == 0:
                return _skip_result(
                    arguments=arguments,
                    rank_reports=setup_reports,
                    reason="parameter or gradient allocation failed on at least one rank",
                )
            return None

        optimizer = AdamW(
            parameters,
            lr=3e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            gradient_dtype=torch.bfloat16,
            reduction_dtype=torch.bfloat16,
            state_dtype=torch.bfloat16,
            master_parameter_dtype=torch.bfloat16,
            replica_group=dist.group.WORLD,
            opt_state_strategy=arguments.opt_state_strategy,
            gradient_reduction="mean",
            bucket_bytes=arguments.bucket_bytes,
        )

        # Materialization, allocator growth, and Triton/NCCL warmup are not
        # scheduler-visible steady-state optimizer work.
        for _ in range(arguments.warmup_steps):
            dist.barrier(group=control_group)
            optimizer.step()
            optimizer.synchronize()
            dist.barrier(group=control_group)

        torch.cuda.synchronize(device)
        dist.barrier(group=control_group)
        allocated_after_warmup = torch.cuda.memory_allocated(device)
        reserved_after_warmup = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)

        samples_us: list[float] = []
        for _ in range(arguments.repetitions):
            # Rank alignment is deliberately outside CUDA-event timing.
            dist.barrier(group=control_group)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(device))
            optimizer.step()
            end.record(torch.cuda.current_stream(device))
            # AdamW.step joins its internal compute and communication work to
            # the caller stream.  Synchronizing only this event therefore
            # measures the whole scheduler-visible step without a device-wide
            # synchronization in the timed interval.
            end.synchronize()
            samples_us.append(float(start.elapsed_time(end)) * 1_000.0)
            optimizer.synchronize()
            dist.barrier(group=control_group)

        local_report = {
            "rank": rank,
            "metadata": metadata,
            "samples_us": samples_us,
            "median_us": statistics.median(samples_us),
            "p95_us": _percentile(samples_us, 0.95),
            "allocated_after_warmup_bytes": allocated_after_warmup,
            "reserved_after_warmup_bytes": reserved_after_warmup,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        rank_reports = _gather_objects(local_report, control_group)
        if rank != 0:
            return None

        distributed_samples_us = [
            max(report["samples_us"][index] for report in rank_reports)
            for index in range(arguments.repetitions)
        ]
        median_us = statistics.median(distributed_samples_us)
        p95_us = _percentile(distributed_samples_us, 0.95)
        median_s = median_us / 1_000_000.0
        parameter_bytes = arguments.parameter_count * 2
        collective_payload_multiplier = (
            2 if arguments.opt_state_strategy == "sharded" else 1
        )
        manifest = optimizer.execution_manifest()
        return {
            "schema_version": "mlops.adamw_scaling.v2",
            "status": "passed",
            "backend": arguments.backend,
            "opt_state_strategy": arguments.opt_state_strategy,
            "world_size": world_size,
            "parameter_count": arguments.parameter_count,
            "parameter_layout": arguments.parameter_layout,
            "parameter_bytes": parameter_bytes,
            "parameter_gib": parameter_bytes / (1 << 30),
            "parameter_tensor_count": len(parameters),
            "bucket_bytes": arguments.bucket_bytes,
            "bucket_count": len(manifest["buckets"]),
            "warmup_steps": arguments.warmup_steps,
            "repetitions": arguments.repetitions,
            "distributed_samples_us": distributed_samples_us,
            "median_us": median_us,
            "p95_us": p95_us,
            "parameters_per_second": arguments.parameter_count / median_s,
            "model_payload_gb_per_second": parameter_bytes / median_s / 1e9,
            "achieved_link_gbit_per_second": (
                parameter_bytes * 8 / median_s / 1e9
            ),
            "collective_payload_bytes_per_step": (
                collective_payload_multiplier * parameter_bytes
            ),
            "effective_collective_payload_gb_per_second": (
                collective_payload_multiplier * parameter_bytes / median_s / 1e9
            ),
            "execution_manifest_totals": {
                "parameter_bytes": sum(
                    bucket["parameter_bytes"] for bucket in manifest["buckets"]
                ),
                "optimizer_state_bytes_per_rank": sum(
                    bucket["optimizer_state_bytes"]
                    for bucket in manifest["buckets"]
                ),
                "parameter_shard_bytes_per_rank": sum(
                    bucket["parameter_shard_bytes"]
                    for bucket in manifest["buckets"]
                ),
                "workspace_bytes_per_rank": manifest["workspace_bytes"],
            },
            "implementation_id": manifest["spec"]["local_implementation_id"],
            "rank_reports": rank_reports,
        }
    finally:
        if optimizer is not None:
            try:
                optimizer.synchronize()
            except Exception:
                pass
        parameters.clear()
        optimizer = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def main() -> None:
    arguments = _arguments()
    started = time.perf_counter()
    result = _run(arguments)
    if result is not None:
        result["cell_wall_time_s"] = time.perf_counter() - started
        _write_result(arguments.output, result)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
