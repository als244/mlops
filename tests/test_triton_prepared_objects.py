"""Real CPU cross-compilation, immutable export, and pre-load validation."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.runtime.jit import MockTensor

from mlops.triton_runtime._objects import (
    TritonModulePool,
    TritonObjects,
    _decode,
    _encode,
)
from mlops.triton_runtime._scope import TritonCalls


@triton.jit
def _scaled_copy(x, out, n, BLOCK: tl.constexpr, SCALE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + offsets, offsets < n, 0)
    tl.store(out + offsets, value * SCALE, offsets < n)


@pytest.fixture
def exported(monkeypatch):
    from triton.runtime.driver import driver

    target = GPUTarget("cuda", 90, 32)
    monkeypatch.setattr(
        driver,
        "_active",
        SimpleNamespace(
            get_current_target=lambda: target,
            get_current_device=lambda: 0,
            get_current_stream=lambda device: 0,
        ),
    )
    fn = triton.JITFunction(_scaled_copy.fn)
    calls = TritonCalls((fn,))
    objects = TritonObjects(calls, target=target)
    with calls.capture() as selected:
        for block in (32, 64):
            fn.run(
                MockTensor(torch.float32),
                MockTensor(torch.float32),
                65,
                BLOCK=block,
                SCALE=2.0,
                grid=(1,),
                warmup=True,
            )
    manifest, blobs = objects.export(selected)
    return objects, manifest, blobs, selected


def test_real_cross_compiled_binaries_roundtrip_without_gpu(exported, monkeypatch):
    objects, manifest, blobs, selected = exported
    initialized = torch.cuda.is_initialized()
    assert len(selected) == len(manifest["calls"]) == 2
    assert all(data.startswith(b"\x7fELF") for data in blobs.values())

    def forbid(*args, **kwargs):
        raise AssertionError("unexpected compilation")

    monkeypatch.setattr(triton, "compile", forbid)
    monkeypatch.setattr(triton.compiler, "compile", forbid)
    monkeypatch.setattr(triton.JITFunction, "_do_compile", forbid)
    records, loaded = objects.validate(
        json.loads(json.dumps(manifest)), blobs.__getitem__
    )
    assert loaded == blobs and records.keys() == selected.keys()
    for key, (_, src) in records.items():
        assert src.hash() == selected[key].src.hash()
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize(
    "change",
    [
        "target",
        "source",
        "duplicate",
        "hash",
        "blob",
        "path",
        "metadata",
        "allocation",
        "entry",
    ],
)
def test_refuse_mismatch_or_corruption_before_loading(exported, change):
    objects, original, blobs, _ = exported
    manifest = deepcopy(original)
    if change == "target":
        manifest["environment"]["target"]["arch"] = 120
    elif change == "source":
        manifest["calls"][-1]["source"] = _encode(({}, {}, {}))
    elif change == "duplicate":
        manifest["calls"].append(deepcopy(manifest["calls"][0]))
    elif change == "hash":
        manifest["calls"][-1]["metadata"]["hash"] = "f" * 64
    elif change == "blob":
        blobs = dict.fromkeys(blobs, b"broken")
    elif change == "path":
        manifest["calls"][-1]["sha256"] = "../library"
    elif change == "allocation":
        del manifest["calls"][-1]["metadata"]["global_scratch_size"]
    elif change == "entry":
        manifest["calls"][-1]["metadata"]["name"] = "absent_function"
    else:
        manifest["calls"][-1]["metadata"]["extern_libs"] = [["libdevice", "/untrusted"]]
    with pytest.raises(ValueError):
        objects.validate(manifest, blobs.__getitem__)


def test_reject_eager_binary_with_stale_options(exported):
    objects, _, _, selected = exported
    kernel = next(iter(selected.values()))
    kernel.hash = "f" * 64
    with pytest.raises(ValueError, match="stale captured"):
        objects.export(selected)


def test_typed_values_preserve_collections_and_do_not_import_objects():
    value = {(0,): [tl.float32, tl.constexpr(3), None, True], (1,): (2, 0.25)}
    decoded = _decode(json.loads(json.dumps(_encode(value))))
    assert decoded == value
    for invalid in ({"callable": "os.system"}, {"dtype": "bogus"}, {"tuple": "bad"}):
        with pytest.raises(ValueError):
            _decode(invalid)
    with pytest.raises(TypeError):
        _encode(object())


@pytest.fixture
def module_driver(monkeypatch, exported):
    from cuda.bindings import driver as cuda
    from triton.compiler import CompiledKernel
    from triton.runtime.driver import driver

    state = SimpleNamespace(context=42, loads=[], unloads=[], fail=False)
    monkeypatch.setattr(cuda, "cuCtxGetCurrent", lambda: (0, state.context))
    monkeypatch.setattr(cuda, "cuCtxGetDevice", lambda: (0, 0))
    monkeypatch.setattr(
        cuda, "cuModuleUnload", lambda module: (state.unloads.append(int(module)) or 0,)
    )
    driver.active.launcher_cls = lambda src, metadata: object()

    def load(kernel):
        module = 100 + len(state.loads)
        state.loads.append(module)
        kernel.module, kernel.function = module, module + 1000
        kernel._run = object()
        kernel.n_regs, kernel.n_spills, kernel.n_max_threads = 32, 0, 1024
        if state.fail:
            raise RuntimeError("simulated admission failure after loading")

    monkeypatch.setattr(CompiledKernel, "_init_handles", load)
    return state


def test_module_owner_sharing_context_and_closed_bindings(exported, module_driver):
    objects, manifest, blobs, _ = exported
    pool = TritonModulePool(objects, device=0)
    first = pool.bind(manifest, blobs.__getitem__)
    load_count = len(module_driver.loads)
    second = pool.bind(manifest, blobs.__getitem__)
    assert len(module_driver.loads) == load_count
    assert first.binding.scope.prepared == second.binding.scope.prepared
    with (
        first.activate(),
        second.activate(),
        pytest.raises(RuntimeError, match="active"),
    ):
        pool.close()
    module_driver.context = 99
    with pytest.raises(ValueError, match="another CUDA context"), first.activate():
        pass
    module_driver.context = 42
    pool.close()
    pool.close()
    assert module_driver.unloads == module_driver.loads
    with pytest.raises(RuntimeError, match="closed"), second.activate():
        pass


def test_partial_load_failure_releases_new_module(exported, module_driver):
    objects, manifest, blobs, _ = exported
    pool = TritonModulePool(objects, device=0)
    module_driver.fail = True
    with pytest.raises(RuntimeError, match="simulated admission failure"):
        pool.bind(manifest, blobs.__getitem__)
    assert module_driver.loads == module_driver.unloads == [100]
    assert not pool.modules and not pool.binaries
    module_driver.fail = False
    pool.bind(manifest, blobs.__getitem__)
    pool.close()
    assert module_driver.loads == module_driver.unloads


def test_all_bytes_validated_before_first_module_load(exported, module_driver):
    objects, manifest, blobs, _ = exported
    pool = TritonModulePool(objects, device=0)
    broken = dict(blobs)
    broken[manifest["calls"][-1]["sha256"]] = b"broken"
    with pytest.raises(ValueError, match="corrupt"):
        pool.bind(manifest, broken.__getitem__)
    assert not module_driver.loads
    pool.close()
