"""Validated external Triton binaries with executor-owned CUDA modules.

This protocol stores only CUBIN, launch metadata and a typed AST signature; it
does not call JITFunction.preload(), compile(), or regenerate IR on restoration.
Kernel code and launchers are loaded before the caller performs memory admission.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock

from ._scope import CallKey, TritonCalls

_SCHEMA = "mlops.triton.objects.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PINS = {
    "triton.compiler.compiler": "021a45ca590052ab6ca6d850eb0a102abf542165836ad92bfb3bb573218fc5d3",
    "triton.backends.nvidia.driver": "992ba21e12ee170a904f3a1caca24a78ce3dd66b661087faea5973b995480f63",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _encode(value):
    import triton.language as tl

    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (tuple, list):
        return {type(value).__name__: [_encode(item) for item in value]}
    if type(value) is dict:
        return {"dict": [[_encode(k), _encode(v)] for k, v in value.items()]}
    if type(value) is tl.dtype:
        return {"dtype": str(value)}
    if type(value) is tl.constexpr:
        return {"constexpr": _encode(value.value)}
    raise TypeError(f"unsupported prepared Triton constant: {type(value)}")


def _decode(value):
    import triton.language as tl

    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is not dict or len(value) != 1:
        raise ValueError("invalid typed Triton value")
    tag, items = next(iter(value.items()))
    if tag in ("tuple", "list") and isinstance(items, list):
        result = [_decode(item) for item in items]
        return tuple(result) if tag == "tuple" else result
    if tag == "dict" and isinstance(items, list):
        result = {}
        for pair in items:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("invalid typed Triton mapping")
            key, val = map(_decode, pair)
            if key in result:
                raise ValueError("duplicate typed Triton mapping key")
            result[key] = val
        return result
    if tag == "dtype" and isinstance(items, str) and tl.dtype.is_dtype(items):
        return tl.dtype(items)
    if tag == "constexpr":
        return tl.constexpr(_decode(items))
    raise ValueError("unsupported typed Triton value")


def _environment(calls, target):
    import triton
    from triton import knobs
    from triton._C.libtriton import get_cache_invalidating_env_vars
    from triton.compiler import make_backend
    from triton.runtime.cache import triton_key

    for name, expected in _PINS.items():
        if (
            digest(Path(importlib.import_module(name).__file__).read_bytes())
            != expected
        ):
            raise ValueError("unsupported Triton binary-loader contract")
    if target.backend != "cuda" or target.warp_size != 32:
        raise ValueError("prepared Triton objects currently require a CUDA target")
    if (
        knobs.compilation.override
        or knobs.compilation.instrumentation_mode
        or knobs.runtime.add_stages_inspection_hook is not None
    ):
        raise ValueError("prepared Triton objects do not support compiler overrides")
    backend = make_backend(target)
    return {
        "schema": _SCHEMA,
        "triton": triton.__version__,
        "compiler": digest(triton_key().encode()),
        "backend": backend.hash(),
        "target": target.__dict__,
        "environment": get_cache_invalidating_env_vars(),
        "functions": {
            name: {
                "source": fn.cache_key,
                "specialization": _encode(
                    (
                        fn.do_not_specialize,
                        fn.do_not_specialize_on_alignment,
                        fn.debug,
                        fn.noinline,
                    )
                ),
            }
            for name, fn in sorted(calls.functions.items())
        },
    }


class TritonObjects:
    """Export selected calls and validate manifests without needing a GPU.

    Target is an explicit Triton GPUTarget. Ordinary eager caches are acceptable
    only when their actual AST/options/environment reproduce the compiler hash.
    Custom linked libraries, instrumentation and IR overrides refuse in v1.
    """

    def __init__(self, calls: TritonCalls, *, target):
        from triton.compiler import make_backend

        self.calls, self.target = calls, target
        self.backend = make_backend(target)
        self.environment = _environment(calls, target)

    def _check(self):
        if _environment(self.calls, self.target) != self.environment:
            raise ValueError("Triton source/target/environment changed")

    def _options(self, metadata):
        # Relocate the bundled libdevice by content, never open an artifact's
        # absolute path. Arbitrary external libraries need a separate contract.
        values = dict(metadata)
        values["extern_libs"] = None
        for key, value in values.items():
            if isinstance(value, list):
                values[key] = tuple(value)
        options = self.backend.parse_options(values)
        if (
            options.arch != f"sm{self.target.arch}"
            or options.ir_override
            or options.instrumentation_mode
        ):
            raise ValueError("incompatible Triton options/target")
        return options

    def _compile_hash(self, src, metadata):
        from triton.runtime.cache import get_cache_key

        key = get_cache_key(
            src, self.backend, self._options(metadata), self.environment["environment"]
        )
        return digest(key.encode())

    def export(self, selected):
        from triton.compiler import ASTSource, CompiledKernel

        self._check()
        records, blobs = [], {}
        for key, kernel in sorted(selected.items()):
            fn = self.calls.functions.get(key.function)
            if (
                fn is None
                or not _DIGEST.fullmatch(key.specialization)
                or type(kernel) is not CompiledKernel
                or type(kernel.src) is not ASTSource
                or kernel.src.fn is not fn
            ):
                raise ValueError("unknown prepared Triton call or AST source")
            metadata = dict(kernel.metadata._asdict())
            metadata["target"] = metadata["target"].__dict__
            if metadata["target"] != self.target.__dict__:
                raise ValueError("captured Triton kernel has a different target")
            expected_libs = self.backend.parse_options({}).extern_libs
            if tuple(map(tuple, metadata["extern_libs"])) != expected_libs:
                raise ValueError("prepared Triton objects require bundled libdevice")
            if self._compile_hash(kernel.src, metadata) != kernel.hash:
                raise ValueError("stale captured Triton source/options/environment")
            metadata["extern_libs"] = [["libdevice", "bundled"]]
            data = kernel.asm["cubin"]
            if not isinstance(data, bytes) or not data:
                raise ValueError("missing Triton CUBIN")
            sha = digest(data)
            blobs[sha] = data
            records.append(
                {
                    "function": key.function,
                    "specialization": key.specialization,
                    "source": _encode(
                        (kernel.src.signature, kernel.src.constants, kernel.src.attrs)
                    ),
                    "metadata": json.loads(json_bytes(metadata)),
                    "sha256": sha,
                    "bytes": len(data),
                }
            )
        return {"environment": deepcopy(self.environment), "calls": records}, blobs

    def validate(self, manifest, read_blob):
        """Check every record and byte before a caller loads any executable."""
        from triton.compiler import ASTSource

        self._check()
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"environment", "calls"}
            or manifest["environment"] != self.environment
        ):
            raise ValueError("incompatible Triton source/target/environment manifest")
        if not isinstance(manifest["calls"], list):
            raise TypeError("invalid Triton call list")
        records, blobs = {}, {}
        for row in manifest["calls"]:
            if not isinstance(row, dict) or set(row) != {
                "function",
                "specialization",
                "source",
                "metadata",
                "sha256",
                "bytes",
            }:
                raise ValueError("invalid Triton call record")
            if (
                not isinstance(row["function"], str)
                or row["function"] not in self.calls.functions
                or any(
                    not isinstance(row[k], str) or not _DIGEST.fullmatch(row[k])
                    for k in ("specialization", "sha256")
                )
                or type(row["bytes"]) is not int
                or row["bytes"] <= 0
            ):
                raise ValueError("invalid Triton call identity")
            key = CallKey(row["function"], row["specialization"])
            if key in records:
                raise ValueError("duplicate Triton call")
            metadata = row["metadata"]
            if (
                not isinstance(metadata, dict)
                or metadata.get("target") != self.target.__dict__
                or metadata.get("extern_libs") != [["libdevice", "bundled"]]
            ):
                raise ValueError("invalid Triton launch metadata")
            signature, constants, attrs = _decode(row["source"])
            src = ASTSource(
                self.calls.functions[key.function], signature, constants, attrs
            )
            if metadata.get("name") != src.name:
                raise ValueError("Triton kernel entry name differs from AST source")
            for field in (
                "shared",
                "global_scratch_size",
                "global_scratch_align",
                "profile_scratch_size",
                "profile_scratch_align",
                "num_warps",
                "num_ctas",
            ):
                if type(metadata.get(field)) is not int or metadata[field] < 0:
                    raise ValueError("invalid Triton launch allocation metadata")
            for field in (
                "global_scratch_align",
                "profile_scratch_align",
                "num_warps",
                "num_ctas",
            ):
                if metadata[field] == 0 or metadata[field] & (metadata[field] - 1):
                    raise ValueError("invalid Triton launch alignment/group metadata")
            if any(
                type(metadata.get(field)) is not bool
                for field in ("launch_cooperative_grid", "launch_pdl")
            ):
                raise ValueError("invalid Triton launcher flags")
            # Tensor descriptors need their own encoder/qualification. None of
            # the registered BF16 Sonic kernels use them.
            if metadata.get("tensordesc_meta") not in (None, []):
                raise ValueError("prepared Triton tensor descriptors are unsupported")
            if self._compile_hash(src, metadata) != metadata.get("hash"):
                raise ValueError("Triton source/options hash mismatch")
            sha = row["sha256"]
            if sha not in blobs:
                blobs[sha] = read_blob(sha)
            data = blobs[sha]
            if (
                not isinstance(data, bytes)
                or len(data) != row["bytes"]
                or digest(data) != sha
            ):
                raise ValueError("corrupt Triton CUBIN")
            records[key] = (deepcopy(row), src)
        return records, blobs


class TritonModulePool:
    """An executor's shared, eagerly loaded modules for one CUDA context.

    Exact records share host launchers across task manifests; identical CUBIN and
    launch metadata share CUDA modules even with different AST specializations.
    The caller must
    keep the CUDA context current through activation and quiesce streams before
    close. Bindings become invalid when the pool closes. No module loading is
    deferred into the measured task, and no ordinary eager cache gains ownership.
    """

    def __init__(self, objects: TritonObjects, *, device: int):
        from cuda.bindings import driver as cuda
        from triton.runtime.driver import driver

        self.cuda = cuda
        self.objects, self.device = objects, device
        self.modules, self.binaries, self.active = {}, {}, 0
        self.lock = RLock()
        self.closed = False
        if type(device) is not int or driver.active.get_current_device() != device:
            raise ValueError("Triton module pool requires its explicit current device")
        if driver.active.get_current_target() != objects.target:
            raise ValueError("Triton module pool target differs from current device")
        self.context = self._result(cuda.cuCtxGetCurrent())
        if not int(self.context) or int(self._result(cuda.cuCtxGetDevice())) != device:
            raise ValueError("Triton module pool requires a current CUDA context")
        self.directory = tempfile.TemporaryDirectory(prefix="mlops-triton-objects-")

    @staticmethod
    def _result(result):
        status, *values = result
        if int(status):
            raise RuntimeError(f"CUDA module operation failed: {status}")
        return values[0] if values else None

    def _check(self):
        if self.closed:
            raise RuntimeError("prepared Triton module pool is closed")
        if self._result(self.cuda.cuCtxGetCurrent()) != self.context:
            raise ValueError("prepared Triton pool used in another CUDA context")

    def _unload(self, kernel):
        if kernel.module is not None:
            self._result(self.cuda.cuModuleUnload(self.cuda.CUmodule(kernel.module)))
            kernel.module = kernel.function = kernel._run = None

    def bind(self, manifest, read_blob):
        from triton.compiler import CompiledKernel
        from triton.runtime.driver import driver

        records, blobs = self.objects.validate(manifest, read_blob)
        with self.lock:
            self._check()
            loaded, acquired, acquired_binaries = {}, [], []
            try:
                for key, (row, src) in records.items():
                    identity = digest(json_bytes(row))
                    if identity not in self.modules:
                        directory = Path(self.directory.name) / identity
                        directory.mkdir(exist_ok=True)
                        meta, cubin = (
                            directory / "kernel.json",
                            directory / "kernel.cubin",
                        )
                        meta.write_bytes(json_bytes(row["metadata"]))
                        cubin.write_bytes(blobs[row["sha256"]])
                        kernel = CompiledKernel(
                            src,
                            {meta.name: str(meta), cubin.name: str(cubin)},
                            row["metadata"]["hash"],
                        )
                        # Compilation hashes can differ solely because one
                        # pointer has stronger alignment. Preserve each AST's
                        # launcher/signature, but share identical device code.
                        launch_metadata = {
                            k: v for k, v in row["metadata"].items() if k != "hash"
                        }
                        binary_key = digest(
                            json_bytes((row["sha256"], launch_metadata))
                        )
                        if binary_key in self.binaries:
                            owner = self.binaries[binary_key]
                            kernel._run = driver.active.launcher_cls(
                                src, kernel.metadata
                            )
                            for field in (
                                "module",
                                "function",
                                "n_regs",
                                "n_spills",
                                "n_max_threads",
                            ):
                                setattr(kernel, field, getattr(owner, field))
                        else:
                            try:
                                kernel._init_handles()
                            except BaseException:
                                self._unload(kernel)
                                raise
                            self.binaries[binary_key] = kernel
                            acquired_binaries.append(binary_key)
                        self.modules[identity] = kernel
                        acquired.append(identity)
                    loaded[key] = self.modules[identity]
                binding = self.objects.calls.bind(loaded, device=self.device)
                return _LoadedCalls(self, binding)
            except BaseException:
                for identity in reversed(acquired):
                    self.modules.pop(identity)
                for identity in reversed(acquired_binaries):
                    self._unload(self.binaries.pop(identity))
                raise

    def close(self):
        with self.lock:
            if self.closed:
                return
            self._check()
            if self.active:
                raise RuntimeError("cannot close active Triton module pool")
            for key in list(self.binaries):
                self._unload(self.binaries[key])
                del self.binaries[key]
            for kernel in self.modules.values():
                kernel.module = kernel.function = kernel._run = None
            self.modules.clear()
            self.directory.cleanup()
            self.closed = True


class _LoadedCalls:
    def __init__(self, pool, binding):
        self.pool, self.binding = pool, binding

    @contextmanager
    def activate(self):
        with self.pool.lock:
            self.pool._check()
            self.pool.active += 1
        try:
            with self.binding.activate():
                yield
        finally:
            with self.pool.lock:
                self.pool.active -= 1
