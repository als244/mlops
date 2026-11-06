"""Private immutable Quack object inventory, independent of ShadowSpill storage.

Only selected, target-qualified capture owns exportable objects. Ordinary eager
caches cannot establish provenance. A reader validates the complete manifest
and every blob before loading anything. This component does not cover other
provider compilers, tuning choices, or device-memory admission.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import re
import sys
import sysconfig
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

from . import initialize
from ._cache_scope import CallKey, QuackCalls

_SCHEMA = "mlops.quack.objects.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@lru_cache(maxsize=8)
def _runtime_digest(path: str, stat_key: tuple) -> str:
    # The runtime is tens of MiB. Revalidate its inode/size/timestamps, but do
    # not hash identical library bytes for every unique task's manifest.
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _cpu_identity() -> dict[str, str]:
    # CuTe's default host compilation can use the native CPU. Be conservative
    # until an explicit portable host target is qualified; OS/ISA compatibility
    # is distinct from the GPU target and Python ABI.
    fields = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if not line:
            break
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"model name", "flags", "Features", "CPU part"}:
            fields[key.strip()] = value.strip()
    return fields


def _environment(device: Any = None) -> dict[str, Any]:
    initialize()
    import cutlass
    import torch
    import tvm_ffi
    from cutlass.cutlass_dsl import CuTeDSL
    from quack.cute_dsl_utils import get_device_capacity

    jit = importlib.import_module("quack.cache.jit")
    env = CuTeDSL._get_dsl().envar
    target = env.arch
    match = re.fullmatch(r"sm_(\d+)([af]?)", target)
    if match is None or env.dryrun:
        raise ValueError("unsupported CuTe target or dry run")
    runtime = Path(env.shared_libs or "")
    if runtime.name != "libcute_dsl_runtime.so" or not runtime.is_file():
        raise ValueError("unsupported CuTe host runtime libraries")
    runtime_stat = runtime.stat()
    runtime_key = tuple(
        getattr(runtime_stat, name)
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    )
    # Arbitrary compiler options can override GPU/host targets and ABI. Add
    # explicit reviewed options later instead of assigning them guessed meaning.
    if env.compiler_opt:
        raise ValueError("prepared Quack objects do not accept CUTE_DSL_COMPILER_OPT")
    if device is not None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("prepared Quack execution requires a CUDA device")
        major, minor = torch.cuda.get_device_capability(device)
        if int(match[1]) != major * 10 + minor:
            raise ValueError("CuTe binary target differs from the execution device")
    fingerprint = jit._compute_source_fingerprint.__wrapped__()
    if jit._compute_source_fingerprint() != fingerprint:
        raise ValueError("Quack cached source identity predates source registration")
    return {
        "schema": _SCHEMA,
        # Recompute rather than adopt an old memoized fingerprint after someone
        # registered another source directory. No file hashing runs at task entry.
        "source": fingerprint,
        "target": target,
        "dispatch": list(get_device_capacity(device)),
        "python": sys.implementation.cache_tag,
        "soabi": sysconfig.get_config_var("SOABI"),
        "system": platform.system(),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "cpu": _cpu_identity(),
        "cutlass": cutlass.__version__,
        "tvm_ffi": tvm_ffi.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "host_runtime": _runtime_digest(str(runtime), runtime_key),
        "options": {
            name: getattr(env, name)
            for name in (
                "enable_assertions",
                "enable_pyir",
                "enable_tvm_ffi",
                "lineinfo",
                "debug",
            )
        },
    }


class QuackObjects:
    """Capture/export selected objects and load validated task-owned bindings.

    ``cache_root`` holds reusable build intermediates, never runtime spill.
    ``device`` is required by a GPU executor; omit it only for CPU cross-compile
    and load controls that never launch a kernel. Export returns ordinary JSON
    metadata and SHA-256 addressed blobs, with no absolute paths or pickle.
    """

    def __init__(self, calls: QuackCalls, cache_root: Path, *, device: Any = None):
        self.calls = calls
        self.device = device
        self.environment = _environment(device)
        stamp = _digest(json.dumps(self.environment, sort_keys=True).encode())
        self.directory = Path(cache_root) / _SCHEMA / stamp
        self.functions = {function.cache.function for function in calls.functions}

    def _validate_environment(self):
        if _environment(self.device) != self.environment:
            raise ValueError(
                "Quack source/target/ABI changed during object preparation"
            )

    @contextmanager
    def capture(self) -> Iterator[dict[CallKey, Any]]:
        self._validate_environment()
        with self.calls.capture(cache_directory=self.directory) as selected:
            yield selected

    def export(self, selected: Mapping[CallKey, Any]) -> tuple[dict, dict[str, bytes]]:
        self._validate_environment()
        records, blobs = [], {}
        for key in sorted(selected, key=lambda key: (key.function, key.specialization)):
            if key.function not in self.functions or not _DIGEST.fullmatch(
                key.specialization
            ):
                raise ValueError("unknown prepared Quack function or specialization")
            path = (
                self.directory
                / self.environment["source"]
                / (key.specialization + ".o")
            )
            # Missing export is an error, never a compile fallback.
            data = path.read_bytes()
            if not data:
                raise ValueError("empty prepared Quack object")
            sha = _digest(data)
            blobs[sha] = data
            records.append(
                {
                    "function": key.function,
                    "specialization": key.specialization,
                    "sha256": sha,
                    "bytes": len(data),
                }
            )
        return {"environment": deepcopy(self.environment), "calls": records}, blobs

    def load(
        self, manifest: dict, read_blob: Callable[[str], bytes]
    ) -> LoadedQuackObjects:
        self._validate_environment()
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"environment", "calls"}
            or manifest["environment"] != self.environment
        ):
            raise ValueError("incompatible Quack source/target/ABI manifest")
        records, blobs = {}, {}
        if not isinstance(manifest["calls"], list):
            raise TypeError("invalid prepared Quack call manifest")
        for item in manifest["calls"]:
            if not isinstance(item, dict) or set(item) != {
                "function",
                "specialization",
                "sha256",
                "bytes",
            }:
                raise ValueError("invalid prepared Quack call record")
            if (
                not isinstance(item["function"], str)
                or item["function"] not in self.functions
                or not isinstance(item["specialization"], str)
                or not _DIGEST.fullmatch(item["specialization"])
                or not isinstance(item["sha256"], str)
                or not _DIGEST.fullmatch(item["sha256"])
                or type(item["bytes"]) is not int
                or item["bytes"] <= 0
            ):
                raise ValueError("invalid prepared Quack call identity")
            key = CallKey(item["function"], item["specialization"])
            if key in records:
                raise ValueError("duplicate prepared Quack call")
            sha = item["sha256"]
            if sha not in blobs:
                blobs[sha] = read_blob(sha)
            data = blobs[sha]
            if (
                not isinstance(data, bytes)
                or len(data) != item["bytes"]
                or _digest(data) != sha
            ):
                raise ValueError("corrupt prepared Quack object")
            records[key] = sha
        # All records and bytes are checked before loading the first executable.
        return LoadedQuackObjects(self.calls, records, blobs)


class LoadedQuackObjects:
    def __init__(self, calls, records, blobs):
        from cutlass import cute

        self.directory = tempfile.TemporaryDirectory(prefix="mlops-quack-objects-")
        self.modules = {}
        self.binding = None
        self.active = 0
        self.lock = RLock()
        try:
            for sha, data in blobs.items():
                path = Path(self.directory.name) / (sha + ".o")
                path.write_bytes(data)
                self.modules[sha] = cute.runtime.load_module(
                    str(path), enable_tvm_ffi=True
                )
            self.binding = calls.bind(
                {key: self.modules[sha]["func"] for key, sha in records.items()}
            )
        except BaseException:
            self.close()
            raise

    @contextmanager
    def activate(self):
        with self.lock:
            if self.binding is None:
                raise RuntimeError("prepared Quack objects are closed")
            self.active += 1
        try:
            with self.binding.activate():
                yield
        finally:
            with self.lock:
                self.active -= 1

    def close(self):
        """Release ownership after the caller has quiesced its execution streams."""
        with self.lock:
            if self.active:
                raise RuntimeError("cannot close active prepared Quack objects")
            self.binding = None
            self.modules.clear()
            self.directory.cleanup()
