#!/usr/bin/env python3
"""Launch failure-isolated AdamW scaling cells on two SSH-connected hosts."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", default="tubingen")
    parser.add_argument("--master-addr", default="192.168.50.23")
    parser.add_argument("--master-port", type=int, default=29640)
    parser.add_argument("--local-nic", default="enp114s0f1np1")
    parser.add_argument("--remote-nic", default="enp4s0f0np0")
    parser.add_argument(
        "--local-repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--remote-repo",
        default="/home/shein/Documents/mlops",
    )
    parser.add_argument(
        "--local-conda",
        default="/home/shein/miniconda3/bin/conda",
    )
    parser.add_argument(
        "--remote-conda",
        default="/home/shein/miniconda3/bin/conda",
    )
    parser.add_argument("--conda-env", default="dataflow")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("nccl", "gloo"),
        default=("nccl",),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("sharded", "replicated"),
        default=("sharded", "replicated"),
    )
    parser.add_argument("--min-exponent", type=int, default=20)
    parser.add_argument("--max-exponent", type=int, default=31)
    parser.add_argument("--bucket-bytes", type=int, default=64 << 20)
    parser.add_argument(
        "--parameter-layout",
        choices=("bucketed", "single"),
        default="bucketed",
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--cell-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--sync-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rsync the local package to the remote repo before launching",
    )
    return parser.parse_args()


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _torchrun_command(
    *,
    repo: str,
    conda_executable: str,
    conda_env: str,
    node_rank: int,
    master_addr: str,
    master_port: int,
    backend: str,
    strategy: str,
    parameter_count: int,
    parameter_layout: str,
    bucket_bytes: int,
    warmup_steps: int,
    repetitions: int,
    output: str,
) -> list[str]:
    return [
        conda_executable,
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "torchrun",
        "--nnodes=2",
        "--nproc-per-node=1",
        f"--node-rank={node_rank}",
        f"--master-addr={master_addr}",
        f"--master-port={master_port}",
        f"{repo}/benchmarks/adamw_scaling/benchmark.py",
        "--backend",
        backend,
        "--opt-state-strategy",
        strategy,
        "--parameter-count",
        str(parameter_count),
        "--parameter-layout",
        parameter_layout,
        "--bucket-bytes",
        str(bucket_bytes),
        "--warmup-steps",
        str(warmup_steps),
        "--repetitions",
        str(repetitions),
        "--output",
        output,
    ]


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_cell(
    arguments: argparse.Namespace,
    *,
    backend: str,
    strategy: str,
    exponent: int,
    port: int,
) -> dict[str, Any]:
    parameter_count = 1 << exponent
    cell_name = f"{backend}_{strategy}_2p{exponent}"
    cell_dir = arguments.output_dir / "cells" / cell_name
    cell_dir.mkdir(parents=True, exist_ok=True)
    result_path = cell_dir / "result.json"
    local_log = (cell_dir / "chicago.log").open("w")
    remote_log = (cell_dir / "tubingen.log").open("w")
    common_environment = {
        "NCCL_DEBUG": "WARN",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    local_command = _torchrun_command(
        repo=str(arguments.local_repo),
        conda_executable=arguments.local_conda,
        conda_env=arguments.conda_env,
        node_rank=0,
        master_addr=arguments.master_addr,
        master_port=port,
        backend=backend,
        strategy=strategy,
        parameter_count=parameter_count,
        parameter_layout=arguments.parameter_layout,
        bucket_bytes=arguments.bucket_bytes,
        warmup_steps=arguments.warmup_steps,
        repetitions=arguments.repetitions,
        output=str(result_path),
    )
    remote_result = f"/tmp/mlops-{cell_name}.json"
    remote_command = _torchrun_command(
        repo=arguments.remote_repo,
        conda_executable=arguments.remote_conda,
        conda_env=arguments.conda_env,
        node_rank=1,
        master_addr=arguments.master_addr,
        master_port=port,
        backend=backend,
        strategy=strategy,
        parameter_count=parameter_count,
        parameter_layout=arguments.parameter_layout,
        bucket_bytes=arguments.bucket_bytes,
        warmup_steps=arguments.warmup_steps,
        repetitions=arguments.repetitions,
        output=remote_result,
    )
    local_env = os.environ.copy()
    local_env.update(common_environment)
    local_env.update(
        {
            "NCCL_SOCKET_IFNAME": arguments.local_nic,
            "GLOO_SOCKET_IFNAME": arguments.local_nic,
        }
    )
    remote_env = {
        **common_environment,
        "NCCL_SOCKET_IFNAME": arguments.remote_nic,
        "GLOO_SOCKET_IFNAME": arguments.remote_nic,
    }
    remote_shell = (
        f"cd {shlex.quote(arguments.remote_repo)} && "
        + " ".join(
            f"{key}={shlex.quote(value)}" for key, value in remote_env.items()
        )
        + " "
        + _shell_join(remote_command)
    )
    started = time.perf_counter()
    remote_process = subprocess.Popen(
        ["ssh", arguments.remote_host, remote_shell],
        stdout=remote_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    local_process = subprocess.Popen(
        local_command,
        cwd=arguments.local_repo,
        env=local_env,
        stdout=local_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        deadline = started + arguments.cell_timeout_s
        while local_process.poll() is None or remote_process.poll() is None:
            if time.perf_counter() >= deadline:
                timed_out = True
                break
            if (
                local_process.poll() not in (None, 0)
                or remote_process.poll() not in (None, 0)
            ):
                break
            time.sleep(0.25)
    finally:
        _terminate(local_process)
        _terminate(remote_process)
        local_log.close()
        remote_log.close()

    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = {
            "schema_version": "mlops.adamw_scaling.v2",
            "status": "timeout" if timed_out else "failed",
            "reason": "cell produced no rank-zero result; inspect the per-host logs",
            "backend": backend,
            "opt_state_strategy": strategy,
            "parameter_count": parameter_count,
            "parameter_bytes": parameter_count * 2,
            "bucket_bytes": arguments.bucket_bytes,
        }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["launcher"] = {
        "cell_name": cell_name,
        "port": port,
        "wall_time_s": time.perf_counter() - started,
        "local_returncode": local_process.returncode,
        "remote_returncode": remote_process.returncode,
        "timed_out": timed_out,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    passed = [result for result in results if result["status"] == "passed"]
    summary = {
        "schema_version": "mlops.adamw_scaling_sweep.v2",
        "created_unix_s": time.time(),
        "results": results,
        "counts": {
            status: sum(result["status"] == status for result in results)
            for status in sorted({result["status"] for result in results})
        },
        "passed_median_us": (
            statistics.median(result["median_us"] for result in passed)
            if passed
            else None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# AdamW scaling results",
        "",
        "| Backend | Strategy | Parameters | Buckets | Median (ms) | P95 (ms) | Mparam/s | Achieved link rate (Gb/s) | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        if result["status"] == "passed":
            lines.append(
                "| {backend} | {strategy} | {count:,} | {buckets} | {median:.3f} | {p95:.3f} | {rate:.2f} | {collective:.3f} | passed |".format(
                    backend=result["backend"],
                    strategy=result["opt_state_strategy"],
                    count=result["parameter_count"],
                    buckets=result["bucket_count"],
                    median=result["median_us"] / 1_000.0,
                    p95=result["p95_us"] / 1_000.0,
                    rate=result["parameters_per_second"] / 1e6,
                    collective=(
                        result.get("achieved_link_gbit_per_second")
                        or result["parameter_bytes"]
                        * 8
                        / (result["median_us"] / 1_000_000.0)
                        / 1e9
                    ),
                )
            )
        else:
            lines.append(
                f"| {result['backend']} | {result['opt_state_strategy']} | "
                f"{result['parameter_count']:,} | — | — | — | — | — | "
                f"{result['status']}: {result.get('reason', '')} |"
            )
    lines.extend(
        [
            "",
            "Times are the slower rank for each aligned repetition. Barriers, setup,",
            "materialization, JIT compilation, and handle retirement are outside the",
            "CUDA-event interval. Achieved link rate uses one model payload per rank",
            "for this two-rank topology; it is a comparable end-to-end rate, not a",
            "claim about the provider's internal collective algorithm.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    arguments = _arguments()
    if arguments.max_exponent < arguments.min_exponent:
        raise ValueError("--max-exponent must be at least --min-exponent")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    if arguments.sync_source:
        subprocess.run(
            [
                "rsync",
                "-a",
                "--exclude=results/",
                f"{arguments.local_repo}/",
                f"{arguments.remote_host}:{arguments.remote_repo}/",
            ],
            check=True,
        )
        subprocess.run(
            [
                "ssh",
                arguments.remote_host,
                f"{shlex.quote(arguments.remote_conda)} run -n "
                f"{shlex.quote(arguments.conda_env)} python -m pip install "
                f"--no-deps -e {shlex.quote(arguments.remote_repo)}",
            ],
            check=True,
        )

    results: list[dict[str, Any]] = []
    cell_index = 0
    for backend in arguments.backends:
        for strategy in arguments.strategies:
            for exponent in range(arguments.min_exponent, arguments.max_exponent + 1):
                cell = _run_cell(
                    arguments,
                    backend=backend,
                    strategy=strategy,
                    exponent=exponent,
                    port=arguments.master_port + cell_index,
                )
                results.append(cell)
                _write_summary(arguments.output_dir, results)
                print(
                    f"[{cell_index + 1}] {backend}/{strategy}/2^{exponent}: "
                    f"{cell['status']}"
                )
                cell_index += 1


if __name__ == "__main__":
    main()
