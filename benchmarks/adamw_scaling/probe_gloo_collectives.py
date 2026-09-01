#!/usr/bin/env python3
"""Probe one Gloo CUDA collective and its asynchronous stream bridge."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        choices=("all_reduce", "reduce_scatter", "all_gather"),
        required=True,
    )
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument(
        "--tensor-device",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--completion",
        choices=("host_wait", "stream_bridge"),
        default="stream_bridge",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _launch(
    operation: str,
    *,
    elements: int,
    rank: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, torch.Tensor]:
    if operation == "all_reduce":
        output = torch.full(
            (elements,),
            float(rank + 1),
            device=device,
            dtype=dtype,
        )
        return dist.all_reduce(output, async_op=True), output
    if operation == "reduce_scatter":
        source = torch.full(
            (elements * world_size,),
            float(rank + 1),
            device=device,
            dtype=dtype,
        )
        output = torch.empty(elements, device=device, dtype=dtype)
        work = dist.reduce_scatter_single(output, source, async_op=True)
        return work, output
    source = torch.full(
        (elements,),
        float(rank + 1),
        device=device,
        dtype=dtype,
    )
    output = torch.empty(
        elements * world_size,
        device=device,
        dtype=dtype,
    )
    return dist.all_gather_single(output, source, async_op=True), output


def main() -> None:
    arguments = _arguments()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(
        "cuda", local_rank
    ) if arguments.tensor_device == "cuda" else torch.device("cpu")
    dtype = getattr(torch, arguments.dtype)
    try:
        dist.barrier()
        if device.type == "cpu":
            work, output = _launch(
                arguments.operation,
                elements=arguments.elements,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=dtype,
            )
            work.wait()
            stream_bridge = "not applicable to CPU tensors"
        elif arguments.completion == "host_wait":
            # This branch must never call block_current_stream: it isolates
            # collective support from asynchronous CUDA completion support.
            work, output = _launch(
                arguments.operation,
                elements=arguments.elements,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=dtype,
            )
            work.wait()
            torch.cuda.synchronize(device)
            stream_bridge = "not tested; pure host wait"
        else:
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                work, output = _launch(
                    arguments.operation,
                    elements=arguments.elements,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    dtype=dtype,
                )
                try:
                    work.block_current_stream()
                    stream_bridge = "supported"
                    completion = torch.cuda.Event()
                    completion.record(stream)
                except Exception as error:
                    stream_bridge = f"{type(error).__name__}: {error}"
                    completion = None

            # If Gloo cannot bridge its Work into the CUDA stream, a host wait
            # can still establish whether the collective itself is implemented.
            if completion is None:
                work.wait()
            else:
                completion.synchronize()
            torch.cuda.synchronize(device)
        report = {
            "rank": rank,
            "hostname": socket.gethostname(),
            "device": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
            "dtype": str(dtype),
            "completion": arguments.completion,
            "operation": arguments.operation,
            "work_type": str(type(work)),
            "block_current_stream": stream_bridge,
            "collective_completed": True,
            "output_mean": float(output.float().mean().cpu()),
        }
        reports: list[Any] = [None] * world_size
        dist.all_gather_object(reports, report)
        if rank == 0:
            result = {
                "schema_version": "mlops.gloo_collective_probe.v1",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "world_size": world_size,
                "elements": arguments.elements,
                "operation": arguments.operation,
                "completion": arguments.completion,
                "rank_reports": reports,
            }
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(result, sort_keys=True))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
