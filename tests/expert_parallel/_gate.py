"""The opt-in GPU matrix, prerequisite checks, and bounded torchrun processes."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

GPU = Path(__file__).parent / "gpu"


def worker_command(worker, world_size, output):
    return [
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "--max-restarts=0",
        "--tee",
        "3",
        "--log-dir",
        str(output / "workers"),
        str(GPU / f"worker_{worker}.py"),
    ]


@dataclass(frozen=True)
class Case:
    backend: str
    lora: bool
    precision: str
    recompute: bool
    chunks: int = 1

    @property
    def name(self):
        model = self.backend + ("-lora" if self.lora else "")
        mode = "recompute" if self.recompute else "save"
        return f"{model}-{self.precision}-{mode}-chunks{self.chunks}"

    def command(self, world_size, output):
        worker = "lora" if self.lora else self.backend
        command = worker_command(worker, world_size, output) + [
            "--precision",
            self.precision,
        ]
        if self.lora:
            command += [
                "--backend",
                self.backend,
                "--execution",
                "both",
                "--outdir",
                str(output),
            ]
        elif self.backend == "te":
            command += [
                "--compiled",
                "--num-shared-experts",
                "1",
                "--output",
                str(output / "result.json"),
            ]
        else:
            command += ["--compiled", "--outdir", str(output)]
        if self.recompute:
            command += ["--recompute"]
        if self.chunks > 1:
            command += ["--num-chunks", str(self.chunks), "--num-buffers", "2"]
            if self.precision == "fp8_current":
                command += ["--activation-transport", "fp8"]
        return command


def cases(backend="both"):
    return [
        Case(selected, lora, precision, recompute, chunks)
        for selected in ("quack", "te")
        if backend in ("both", selected)
        for chunks in ((1, 4) if selected == "quack" else (1,))
        for lora in (False, True)
        for precision in (
            ("bf16", "fp8_current", "fp8_block")
            if selected == "te"
            else ("bf16", "fp8_current")
        )
        for recompute in (False, True)
    ]


def check_environment(backend, world_size):
    """Called only when the user explicitly enables the distributed GPU gate."""
    import torch

    requirements = ["moonep", "cutlass"]
    if backend in ("quack", "both"):
        requirements += ["quack", "sonicmoe"]
    if backend in ("te", "both"):
        # TE locates its extension inside its package during import; it need
        # not be independently discoverable as a top-level Python module.
        requirements += ["transformer_engine"]
    missing = [name for name in requirements if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"Missing EP dependencies: {', '.join(missing)}. Run "
            f"scripts/setup_expert_parallel.sh --backend {backend} --python {sys.executable}"
        )
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count < world_size:
        raise RuntimeError(
            f"EP gate requested {world_size} GPUs, but only {count} are visible. "
            "Use --ep-world-size=1 for singleton validation."
        )
    unsupported = [
        i for i in range(world_size) if torch.cuda.get_device_capability(i) != (9, 0)
    ]
    if unsupported:
        raise RuntimeError(
            f"EP checks require H100/SM90 GPUs; incompatible devices: {unsupported}"
        )
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("EP checks require PyTorch with NCCL support")


def run(command, output, timeout):
    """Record each case independently; stop the entire worker group on timeout."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {
        "command": command,
        "started_utc": datetime.now(UTC).isoformat(),
        "status": "RUNNING",
    }
    report = output / "status.json"
    report.write_text(json.dumps(record, indent=2) + "\n")
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    print(
        f"START {output.name} {record['started_utc']} log={output / 'console.log'}",
        flush=True,
    )
    failure = None
    process = None
    with (output / "console.log").open("w") as log:
        try:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            returncode = process.wait(timeout=timeout)
            if returncode:
                failure = f"worker exited with status {returncode}"
        except BaseException as error:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            failure = f"{type(error).__name__}: {error}"
            if not isinstance(error, (subprocess.TimeoutExpired, OSError)):
                raise
        finally:
            record.update(
                status="FAIL" if failure else "PASS",
                finished_utc=datetime.now(UTC).isoformat(),
                elapsed_seconds=time.monotonic() - started,
                returncode=process.returncode if process is not None else None,
            )
            if failure:
                record["error"] = failure
            report.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"{record['status']} {output.name} {record['elapsed_seconds']:.2f}s", flush=True
    )
    if failure:
        tail = "\n".join((output / "console.log").read_text().splitlines()[-60:])
        raise RuntimeError(f"{failure}\nFull log: {output / 'console.log'}\n{tail}")
    return record
