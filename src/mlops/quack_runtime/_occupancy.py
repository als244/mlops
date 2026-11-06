"""Private prepared CuTe occupancy helper, preserving all live driver queries.

Only HardwareInfo's empty-kernel compilation is replaced. The original method
still queries the current hardware, dynamic shared memory, cluster size and
optional stream. No occupancy result is serialized or supplied by this module.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from pathlib import Path
from threading import RLock

from ._quack_objects import _digest
from ._quack_objects import _environment as _quack_environment

_SOURCE = "5a45ae55d2092095eb9d02a8b78d4eb831285ba2815d3139d77c2823a7cfb864"
_SCHEMA = "mlops.cute.occupancy.v1"
_active: ContextVar[LoadedOccupancy | None] = ContextVar(
    "mlops_occupancy", default=None
)
_lock = RLock()
_installed = False


def _environment(device=None):
    record = _quack_environment(device)
    # This is a device-only CUBIN, not Quack's object containing native host
    # code. Do not require a CPU builder and GPU worker to have the same ISA.
    return {
        name: record[name]
        for name in ("source", "target", "cutlass", "host_runtime", "options")
    }


def _hardware_info():
    from cutlass.utils.hardware_info import HardwareInfo

    path = Path(inspect.getfile(HardwareInfo))
    if hashlib.sha256(path.read_bytes()).hexdigest() != _SOURCE:
        raise ValueError("unsupported CuTe occupancy helper contract")
    return HardwareInfo


def _install():
    global _installed
    with _lock:
        if _installed:
            return
        hardware = _hardware_info()
        original = hardware._get_device_function

        @wraps(original)
        def function(instance):
            binding = _active.get()
            return (
                original(instance) if binding is None else binding.kernel_for(instance)
            )

        hardware._get_device_function = function
        _installed = True


class OccupancyHelper:
    """Prepare one empty-kernel CUBIN and bind it to an explicit live CUDA context.

    The optional device can be omitted only for CPU cross-compilation/export.
    Loading requires an indexed device or the caller's current CUDA device, and
    its context must be current. A saved scalar is never an alternative to a
    runtime occupancy query, including for streams from green contexts.
    """

    def __init__(self, cache_root: Path, *, device=None):
        _hardware_info()
        self.device = device
        self.environment = _environment(device)
        stamp = _digest(json.dumps(self.environment, sort_keys=True).encode())
        self.directory = Path(cache_root) / _SCHEMA / stamp

    def _validate(self):
        _hardware_info()
        if _environment(self.device) != self.environment:
            raise ValueError(
                "CuTe occupancy source/target/ABI changed during preparation"
            )

    def _bytes(self, manifest, read_blob):
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"schema", "source", "environment", "sha256", "bytes"}
            or manifest["schema"] != _SCHEMA
            or manifest["source"] != _SOURCE
            or manifest["environment"] != self.environment
            or not isinstance(manifest["sha256"], str)
            or len(manifest["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in manifest["sha256"])
            or type(manifest["bytes"]) is not int
            or manifest["bytes"] <= 0
        ):
            raise ValueError("incompatible CuTe occupancy artifact")
        data = read_blob(manifest["sha256"])
        if (
            not isinstance(data, bytes)
            or len(data) != manifest["bytes"]
            or _digest(data) != manifest["sha256"]
        ):
            raise ValueError("corrupt CuTe occupancy artifact")
        return data

    def prepare(self):
        """Export a qualified binary, reusing only this identity's build cache."""
        from cutlass import cute
        from quack.cache.jit import FileLock

        self._validate()
        self.directory.mkdir(parents=True, exist_ok=True)
        metadata = self.directory / "helper.json"
        binary = self.directory / "helper.cubin"
        with FileLock(self.directory / "helper.lock", exclusive=True, timeout=60):
            if metadata.exists():
                manifest = json.loads(metadata.read_text())
                data = self._bytes(manifest, lambda _: binary.read_bytes())
            else:
                hardware = _hardware_info()
                # The trace accesses only the class's empty kernel. Bypassing
                # __init__ here avoids CUDA device/context queries on CPU builders.
                owner = hardware.__new__(hardware)
                with tempfile.TemporaryDirectory(
                    prefix="mlops-occupancy-build-"
                ) as directory:
                    compiled = cute.compile(
                        owner._host_function,
                        options=f"--dump-dir={directory} --keep-cubin",
                    )
                    data = compiled.artifacts.CUBIN
                if not isinstance(data, bytes) or not data:
                    raise ValueError("CuTe did not export the occupancy CUBIN")
                manifest = {
                    "schema": _SCHEMA,
                    "source": _SOURCE,
                    "environment": deepcopy(self.environment),
                    "sha256": _digest(data),
                    "bytes": len(data),
                }
                for target, content in (
                    (binary, data),
                    (metadata, (json.dumps(manifest, sort_keys=True) + "\n").encode()),
                ):
                    temporary = target.with_suffix(target.suffix + ".tmp")
                    temporary.write_bytes(content)
                    os.replace(temporary, target)
        return manifest, {manifest["sha256"]: data}

    def load(self, manifest, read_blob):
        """Validate before loading; the caller owns a current matching context."""
        import torch

        self._validate()
        data = self._bytes(manifest, read_blob)
        if self.device is None:
            raise ValueError("loading occupancy requires an explicit execution device")
        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError("loading occupancy requires a CUDA device")
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        _install()
        return LoadedOccupancy(data, index)


class LoadedOccupancy:
    def __init__(self, data, device):
        from cuda.bindings import driver

        self.driver = driver
        self.hardware = _hardware_info()(device)
        self.context = int(self.hardware.context)
        self.device = device
        self.library = None
        self.kernel = None
        self.active = 0
        self.lock = RLock()
        if not self.context or int(self._check(driver.cuCtxGetDevice())) != device:
            raise ValueError("occupancy execution device/context is not current")
        try:
            self.library = self._check(
                driver.cuLibraryLoadData(data, None, None, 0, None, None, 0)
            )
            if self._check(driver.cuLibraryGetKernelCount(self.library)) != 1:
                raise ValueError("occupancy artifact must contain exactly one kernel")
            kernels = self._check(driver.cuLibraryEnumerateKernels(1, self.library))
            self.kernel = self._check(driver.cuKernelGetFunction(kernels[0]))
        except BaseException:
            self.close()
            raise

    def _check(self, result):
        return self.hardware._checkCudaErrors(result)

    def kernel_for(self, hardware):
        if self.library is None:
            raise RuntimeError("prepared occupancy helper is closed")
        if (
            hardware.device_id != self.device
            or int(hardware.context) != self.context
            or int(self._check(self.driver.cuCtxGetCurrent())) != self.context
        ):
            raise ValueError(
                "prepared occupancy helper belongs to another device/context"
            )
        return self.kernel

    @contextmanager
    def activate(self):
        with self.lock:
            if self.library is None:
                raise RuntimeError("prepared occupancy helper is closed")
            self.active += 1
        token = _active.set(self)
        try:
            yield
        finally:
            _active.reset(token)
            with self.lock:
                self.active -= 1

    def close(self):
        """Unload the owned query-only library after active queries have returned."""
        with self.lock:
            if self.active:
                raise RuntimeError("cannot close an active occupancy helper")
            if self.library is not None:
                self._check(self.driver.cuLibraryUnload(self.library))
                self.library = self.kernel = None
