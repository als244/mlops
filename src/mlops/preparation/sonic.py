"""BF16 Sonic/Quack task preparation for ShadowSpill's generic lifecycle.

This adapter owns selected Quack objects, external Triton CUBINs, the native
Triton launcher helper, and CuTe's live occupancy-query helper. Tuning choices
remain ShadowSpill's independent captured contract. It does not cover FP8,
expert parallelism, unrelated attention providers, or change gradient math.
"""

import hashlib
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import RLock

import torch


def _implementation():
    root = Path(__file__).resolve().parents[1]
    sources = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ("preparation", "quack_runtime", "triton_runtime")
        for path in sorted((root / directory).glob("*.py"))
    }
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


class _Common:
    def __init__(self, device, cache_root):
        from triton.backends.compiler import GPUTarget

        from mlops.quack_runtime._benchmark_scope import install
        from mlops.quack_runtime._occupancy import OccupancyHelper
        from mlops.quack_runtime._quack_objects import QuackObjects
        from mlops.quack_runtime._sonic_bindings import sonic_quack_calls
        from mlops.triton_runtime._driver import NativeDriver
        from mlops.triton_runtime._objects import TritonObjects
        from mlops.triton_runtime._sonic import sonic_triton_calls

        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None or torch.version.hip:
            raise ValueError(
                "prepared Sonic requires an explicit NVIDIA execution device"
            )
        self.source = _implementation()
        # Register every kernel source before freezing Quack's source fingerprint.
        self.triton_calls = sonic_triton_calls()
        self.quack_calls = sonic_quack_calls()
        install()
        self.quack = QuackObjects(self.quack_calls, cache_root, device=self.device)
        major, minor = torch.cuda.get_device_capability(self.device)
        self.triton = TritonObjects(
            self.triton_calls, target=GPUTarget("cuda", major * 10 + minor, 32)
        )
        self.native = NativeDriver(cache_root)
        self.occupancy = OccupancyHelper(cache_root, device=self.device)
        self.stack = ExitStack()
        self.closed = False

    def close(self):
        if not self.closed:
            self.closed = True
            self.stack.close()


class _Builder(_Common):
    def __init__(self, device, cache_root):
        super().__init__(device, cache_root)
        try:
            self.native_manifest, self.native_blobs = self.native.prepare()
            # Install before any compilation initializes Triton's lazy driver.
            self.native.install(self.native_manifest, self.native_blobs.__getitem__)
            self.occupancy_manifest, self.occupancy_blobs = self.occupancy.prepare()
            torch.empty(0, device=self.device)
            self.helper = self.occupancy.load(
                self.occupancy_manifest, self.occupancy_blobs.__getitem__
            )
            self.stack.callback(self.helper.close)
        except BaseException:
            self.close()
            raise

    @contextmanager
    def capture(self):
        if self.closed:
            raise RuntimeError("prepared Sonic builder is closed")
        with (
            self.helper.activate(),
            self.quack.capture() as quack,
            self.triton_calls.capture() as triton,
        ):
            yield quack, triton

    def export(self, captured):
        quack, triton = captured
        if not quack and not triton:
            return {"version": 1, "source": self.source, "empty": True}, {}
        q_manifest, q_blobs = self.quack.export(quack)
        t_manifest, t_blobs = self.triton.export(triton)
        return {
            "version": 1,
            "source": self.source,
            "quack": q_manifest,
            "triton": t_manifest,
            "native": self.native_manifest,
            "occupancy": self.occupancy_manifest,
        }, {**q_blobs, **t_blobs, **self.native_blobs, **self.occupancy_blobs}


class _Loader(_Common):
    def __init__(self, device, cache_root):
        super().__init__(device, cache_root)
        self.quack_pool = self.triton_pool = self.helper = None
        self.shared = None
        self.active = 0
        self.lock = RLock()

    def bind(self, manifest, read_blob):
        from mlops.quack_runtime._quack_objects import QuackModulePool
        from mlops.triton_runtime._objects import TritonModulePool

        with self.lock:
            if self.closed or self.active:
                raise RuntimeError("prepared Sonic loader is closed or executing")
            if manifest.get("version") != 1 or manifest.get("source") != self.source:
                raise ValueError("prepared Sonic implementation differs")
            if manifest.get("empty") is True:
                if set(manifest) != {"version", "source", "empty"}:
                    raise ValueError("invalid empty Sonic preparation")
                return _Binding(self, ())
            if set(manifest) != {
                "version",
                "source",
                "quack",
                "triton",
                "native",
                "occupancy",
            }:
                raise ValueError("invalid Sonic preparation manifest")
            # Validate every component before loading any component of this task.
            self.quack.validate(manifest["quack"], read_blob)
            self.triton.validate(manifest["triton"], read_blob)
            self.native.validate(manifest["native"], read_blob)
            self.occupancy._bytes(manifest["occupancy"], read_blob)
            shared = (manifest["native"], manifest["occupancy"])
            if self.shared is not None and self.shared != shared:
                raise ValueError(
                    "Sonic tasks disagree on shared helper implementations"
                )
            if self.shared is None:
                self.native.install(manifest["native"], read_blob)
                self.helper = self.occupancy.load(manifest["occupancy"], read_blob)
                self.stack.callback(self.helper.close)
                self.quack_pool = QuackModulePool(self.quack)
                self.stack.callback(self.quack_pool.close)
                self.triton_pool = TritonModulePool(
                    self.triton, device=self.device.index
                )
                self.stack.callback(self.triton_pool.close)
                self.shared = shared
            quack = self.quack_pool.bind(manifest["quack"], read_blob)
            triton = self.triton_pool.bind(manifest["triton"], read_blob)
            return _Binding(self, (self.helper, quack, triton))

    def close(self):
        with self.lock:
            if self.active:
                raise RuntimeError("cannot close an active prepared Sonic loader")
            super().close()


class _Binding:
    def __init__(self, owner, bindings):
        self.owner, self.bindings = owner, bindings

    @contextmanager
    def activate(self):
        with self.owner.lock:
            if self.owner.closed:
                raise RuntimeError("prepared Sonic loader is closed")
            self.owner.active += 1
        try:
            with ExitStack() as scope:
                for binding in self.bindings:
                    scope.enter_context(binding.activate())
                yield
        finally:
            with self.owner.lock:
                self.owner.active -= 1


def factory(*, phase, device, options):
    """Importable lifecycle factory; use before an ordinary Triton driver warms.

    `cache_root` is optional and affects build intermediates only. Loading reads
    archived bytes; no cache lookup is a replacement for the task manifest.
    """
    if set(options) - {"cache_root"}:
        raise ValueError("unknown prepared Sonic option")
    cache_root = Path(options.get("cache_root", Path.home() / ".cache/mlops/prepared"))
    if phase not in {"build", "load"}:
        raise ValueError("unknown prepared Sonic lifecycle phase")
    with torch.cuda.device(device):
        return (_Builder if phase == "build" else _Loader)(device, cache_root)
