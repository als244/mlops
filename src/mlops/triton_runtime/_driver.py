"""Prepare Triton's one process-wide native CUDA launcher helper on a CPU.

The Python extension is a process-lifetime dependency, not a task CUDA module.
Install it before initializing Triton's driver. It owns no GPU allocations;
per-kernel CUDA modules belong to TritonModulePool and are loaded before admission.
"""

from __future__ import annotations

import fcntl
import importlib
import os
import platform
import sys
import sysconfig
import tempfile
from copy import deepcopy
from functools import wraps
from pathlib import Path
from threading import RLock

from ._objects import _PINS, digest, json_bytes

_SCHEMA = "mlops.triton.cuda_driver.v1"
_BUILD = "26c6ddd4dba3534ce54790f12d2b75729f51e467c48b2f740dd62b7c75a85b39"
_DRIVER = "835f068cf517d237dc62be91275261e260c2bae985a9242fd28866303c1e5a60"
_lock = RLock()
_installed = None


def _environment():
    from triton import knobs

    native = importlib.import_module("triton.backends.nvidia.driver")
    build = importlib.import_module("triton.runtime.build")
    source = Path(native.dirname) / "driver.c"
    if (
        digest(source.read_bytes()) != _DRIVER
        or digest(Path(build.__file__).read_bytes()) != _BUILD
        or digest(Path(native.__file__).read_bytes())
        != _PINS["triton.backends.nvidia.driver"]
    ):
        raise ValueError("unsupported Triton native-helper source")
    if knobs.build.impl is not None or knobs.build.backend_dirs:
        raise ValueError(
            "prepared Triton native helper refuses custom builders/includes"
        )
    return {
        "schema": _SCHEMA,
        "source": _DRIVER,
        "build": _BUILD,
        "headers": {
            path.name: digest(path.read_bytes())
            for path in sorted((Path(native.dirname) / "include").glob("*.h"))
        },
        "python": sys.implementation.cache_tag,
        "soabi": sysconfig.get_config_var("SOABI"),
        "suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "system": platform.system(),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
    }


class NativeDriver:
    def __init__(self, cache_root):
        self.environment = _environment()
        self.directory = (
            Path(cache_root) / _SCHEMA / digest(json_bytes(self.environment))
        )

    def _check(self):
        if _environment() != self.environment:
            raise ValueError("Triton native-helper source/ABI changed")

    def prepare(self):
        """Compile/reuse a qualified host extension without initializing CUDA."""
        import json

        from triton.runtime.build import _build

        self._check()
        native = importlib.import_module("triton.backends.nvidia.driver")
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / ("cuda_utils" + self.environment["suffix"])
        record = self.directory / "manifest.json"
        with (self.directory / "lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if record.exists():
                manifest = json.loads(record.read_text())
                data = destination.read_bytes()
                self.validate(manifest, lambda _: data)
                return manifest, {manifest["sha256"]: data}
            with tempfile.TemporaryDirectory(dir=self.directory) as temporary:
                source = Path(temporary) / "cuda_utils.c"
                source.write_bytes((Path(native.dirname) / "driver.c").read_bytes())
                library = _build(
                    "cuda_utils",
                    str(source),
                    temporary,
                    native.library_dirs(),
                    native.include_dirs,
                    native.libraries,
                    [],
                )
                data = Path(library).read_bytes()
                os.replace(library, destination)
            manifest = {
                "environment": deepcopy(self.environment),
                "sha256": digest(data),
                "bytes": len(data),
            }
            # Publish metadata last; a partial interrupted build is never reused.
            temporary_record = self.directory / "manifest.pending"
            temporary_record.write_bytes(json_bytes(manifest))
            os.replace(temporary_record, record)
            return manifest, {manifest["sha256"]: data}

    def validate(self, manifest, read_blob):
        from ._objects import _DIGEST

        self._check()
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"environment", "sha256", "bytes"}
            or manifest["environment"] != self.environment
            or not isinstance(manifest["sha256"], str)
            or not _DIGEST.fullmatch(manifest["sha256"])
            or type(manifest["bytes"]) is not int
            or manifest["bytes"] <= 0
        ):
            raise ValueError("incompatible Triton native-helper manifest")
        data = read_blob(manifest["sha256"])
        if (
            not isinstance(data, bytes)
            or len(data) != manifest["bytes"]
            or digest(data) != manifest["sha256"]
        ):
            raise ValueError("corrupt Triton native helper")
        return data

    def install(self, manifest, read_blob):
        """Install one validated process-lifetime helper; incompatible peers refuse.

        Repeated matching installs share the same extension and its Python types.
        Python extensions cannot be safely unloaded per task. Already initialized
        eager drivers are accepted only if they use these exact binary bytes.
        No global driver is replaced and no module is loaded before validation.
        """
        global _installed
        from triton.runtime.build import _load_module_from_path
        from triton.runtime.driver import driver

        data = self.validate(manifest, read_blob)
        with _lock:
            if _installed is not None:
                if _installed[0] != manifest:
                    raise ValueError(
                        "another prepared Triton native helper is installed"
                    )
                return _installed[1]
            module = None
            for current in (driver._default, driver._active):
                if current is None:
                    continue
                candidate = getattr(
                    getattr(current, "utils", None), "load_binary", None
                )
                candidate = getattr(candidate, "__self__", None)
                path = getattr(candidate, "__file__", None)
                if (
                    path is None
                    or digest(Path(path).read_bytes()) != manifest["sha256"]
                ):
                    raise ValueError(
                        "initialize prepared helper before Triton's eager driver"
                    )
                if module is not None and module is not candidate:
                    raise ValueError(
                        "multiple native helper module instances are active"
                    )
                module = candidate
            directory = None
            if module is None:
                directory = tempfile.TemporaryDirectory(prefix="mlops-triton-driver-")
                try:
                    path = Path(directory.name) / (
                        "cuda_utils" + self.environment["suffix"]
                    )
                    path.write_bytes(data)
                    module = _load_module_from_path("cuda_utils", str(path))
                except BaseException:
                    directory.cleanup()
                    raise
            native = importlib.import_module("triton.backends.nvidia.driver")
            original = native.compile_module_from_src

            @wraps(original)
            def prepared_helper(*args, **kwargs):
                # Pinned CudaUtils.__init__ uses this exact call. Unexpected
                # requests must not silently invoke a native compiler.
                if (
                    args
                    or set(kwargs)
                    != {"src", "name", "library_dirs", "include_dirs", "libraries"}
                    or kwargs["name"] != "cuda_utils"
                    or digest(kwargs["src"].encode()) != _DRIVER
                    or kwargs["include_dirs"] != native.include_dirs
                    or kwargs["libraries"] != native.libraries
                ):
                    raise ValueError("unknown prepared Triton native-helper request")
                return module

            native.compile_module_from_src = prepared_helper
            # Retain the extension, backing file, and hook for process lifetime.
            _installed = (deepcopy(manifest), module, directory)
            return module
