"""Manifest validation must complete before any executable object is loaded."""

import copy
import hashlib
from types import SimpleNamespace

import pytest

from mlops.quack_runtime import _quack_objects as artifacts


@pytest.fixture
def inventory(monkeypatch, tmp_path):
    environment = {"schema": "test", "source": "source", "target": "sm_90a"}
    monkeypatch.setattr(
        artifacts, "_environment", lambda device: copy.deepcopy(environment)
    )
    name = "test.compile"
    calls = SimpleNamespace(
        functions=[SimpleNamespace(cache=SimpleNamespace(function=name))]
    )
    owner = artifacts.QuackObjects(calls, tmp_path)
    data = b"immutable object"
    sha = hashlib.sha256(data).hexdigest()
    manifest = {
        "environment": copy.deepcopy(environment),
        "calls": [
            {
                "function": name,
                "specialization": str(i) * 64,
                "sha256": sha,
                "bytes": len(data),
            }
            for i in (1, 2)
        ],
    }
    loads = []
    monkeypatch.setattr(
        artifacts, "LoadedQuackObjects", lambda *args: loads.append(args)
    )
    return owner, manifest, {sha: data}, loads, environment


def test_shared_blob_is_read_once_and_dispatch_identities_remain_distinct(inventory):
    owner, manifest, blobs, loads, _ = inventory
    reads = []

    def read(sha):
        reads.append(sha)
        return blobs[sha]

    owner.load(manifest, read)
    assert len(reads) == len(loads) == 1
    assert len(loads[0][1]) == 2 and loads[0][2] == blobs


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "unknown-function",
        "path-specialization",
        "path-blob",
        "wrong-size",
        "extra-field",
        "duplicate-call",
        "corrupt-last-blob",
        "missing-last-blob",
    ],
)
def test_refusals_happen_before_any_load(inventory, change):
    owner, manifest, blobs, loads, _ = inventory
    if change == "source":
        manifest["environment"]["source"] = "changed"
    elif change == "unknown-function":
        manifest["calls"][-1]["function"] = "unknown.compile"
    elif change == "path-specialization":
        manifest["calls"][-1]["specialization"] = "../escape"
    elif change == "path-blob":
        manifest["calls"][-1]["sha256"] = "../escape"
    elif change == "wrong-size":
        manifest["calls"][-1]["bytes"] += 1
    elif change == "extra-field":
        manifest["calls"][-1]["extra"] = "unexpected"
    elif change == "duplicate-call":
        manifest["calls"].append(dict(manifest["calls"][0]))
    else:
        sha = "a" * 64
        manifest["calls"][-1]["sha256"] = sha
        if change == "corrupt-last-blob":
            blobs[sha] = b"corruption"
    with pytest.raises((ValueError, KeyError)):
        owner.load(manifest, blobs.__getitem__)
    assert not loads


def test_changed_preparation_environment_refuses_export_and_load(inventory):
    owner, manifest, blobs, loads, environment = inventory
    environment["target"] = "sm_120a"
    with pytest.raises(ValueError, match="changed during"):
        owner.export({})
    with pytest.raises(ValueError, match="changed during"):
        owner.load(manifest, blobs.__getitem__)
    assert not loads
