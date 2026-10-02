"""The GPU harness must collect safely and preserve actionable failure evidence."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from _gate import cases, run


def test_gpu_matrix_covers_supported_precisions_and_chunk_paths():
    matrix = cases()
    assert len(matrix) == len({case.name for case in matrix}) == 28
    assert {case.precision for case in cases("te")} == {
        "bf16",
        "fp8_current",
        "fp8_block",
    }
    assert {case.chunks for case in cases("quack")} == {1, 4}
    for case in matrix:
        command = case.command(2, Path("/tmp/results") / case.name)
        worker = Path(command[command.index("--log-dir") + 2])
        assert worker.is_file()
        assert (
            importlib.machinery.PathFinder.find_spec("quack", [str(worker.parent)])
            is None
        )
        assert ("--recompute" in command) == case.recompute
        if case.lora:
            assert command[command.index("--backend") + 1] == case.backend


@pytest.mark.parametrize("status", [0, 7])
def test_subprocess_status_and_log_survive_failure(tmp_path, status, capsys):
    output = tmp_path / "case"
    command = [
        sys.executable,
        "-c",
        f"print('worker evidence', flush=True); raise SystemExit({status})",
    ]
    if status:
        with pytest.raises(RuntimeError, match="worker evidence"):
            run(command, output, 10)
    else:
        run(command, output, 10)
    record = json.loads((output / "status.json").read_text())
    assert record["status"] == ("FAIL" if status else "PASS")
    assert record["returncode"] == status
    assert "worker evidence" in (output / "console.log").read_text()
    # Do not replay expected failure messages into a live GPU gate's stdout.
    capsys.readouterr()


def test_timeout_is_bounded_and_recorded(tmp_path, capsys):
    with pytest.raises(RuntimeError, match="TimeoutExpired"):
        run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path / "timeout",
            0.2,
        )
    record = json.loads((tmp_path / "timeout/status.json").read_text())
    assert record["status"] == "FAIL"
    assert record["elapsed_seconds"] < 10
    capsys.readouterr()


def test_rank_visibility_preserves_scheduler_mapping(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "ep_bootstrap", Path(__file__).parent / "gpu/_bootstrap.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    module.select_rank_device()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-second"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(RuntimeError, match="needs a device"):
        module.select_rank_device()
