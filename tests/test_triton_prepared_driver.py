"""Actual CPU native build/cache reuse and pre-import artifact validation."""

from copy import deepcopy

import pytest
import torch

from mlops.triton_runtime._driver import NativeDriver


def test_native_cpu_build_and_cache_reuse_without_compiling(tmp_path, monkeypatch):
    from triton.runtime import build

    initialized = torch.cuda.is_initialized()
    helper = NativeDriver(tmp_path)
    manifest, blobs = helper.prepare()
    assert helper.validate(manifest, blobs.__getitem__).startswith(b"\x7fELF")

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected native compilation")

    monkeypatch.setattr(build, "_build", forbidden)
    assert helper.prepare() == (manifest, blobs)
    assert torch.cuda.is_initialized() == initialized
    wrong = deepcopy(manifest)
    wrong["environment"]["soabi"] = "different"
    with pytest.raises(ValueError, match="incompatible"):
        helper.install(wrong, blobs.__getitem__)
    with pytest.raises(ValueError, match="corrupt"):
        helper.install(manifest, lambda _: b"corrupt")
    # Also refuse a corrupt local build-cache object, rather than silently
    # rebuilding an artifact whose claimed identity has stopped being true.
    path = next(helper.directory.glob("*.so"))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        helper.prepare()
