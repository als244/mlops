"""Provider-owned device selection, saved binding and optional dependencies."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from mlops.preparation import available, sonic
from mlops.providers.sonicmoe import moe


@pytest.mark.parametrize("major,installed", [(9, True), (9, False), (12, True)])
def test_build_records_selected_provider_and_load_does_not_redispatch(
    monkeypatch, major, installed
):
    events = []
    binding = object()

    def factory(**kwargs):
        events.append(kwargs["phase"])
        return SimpleNamespace(
            capture=lambda: nullcontext("captured once"),
            export=lambda captured: ({"captured": captured}, {}),
            bind=lambda manifest, read: binding,
            close=lambda: events.append("close"),
        )

    monkeypatch.setattr(sonic, "factory", factory)
    monkeypatch.setattr(moe, "available", lambda: installed)
    monkeypatch.setattr(
        available.torch.cuda, "get_device_capability", lambda _: (major, 0)
    )
    builder = available.factory(phase="build", device="cuda:0", options={})
    with builder.capture() as state:
        manifest, blobs = builder.export(state)
    builder.close()
    selected = major == 9 and installed
    assert manifest["provider"] == ("sonic" if selected else None)
    assert not blobs

    def forbidden(*args):
        raise AssertionError("saved loading redispatched")

    monkeypatch.setattr(moe, "available", forbidden)
    monkeypatch.setattr(available.torch.cuda, "get_device_capability", forbidden)
    loader = available.factory(phase="load", device="cuda:0", options={})
    restored = loader.bind(manifest, forbidden)
    if selected:
        assert restored is binding
    else:
        with restored.activate():
            pass
    loader.close()
    assert events == (["build", "close", "load", "close"] if selected else [])


def test_saved_provider_failure_cannot_fall_back_to_empty(monkeypatch):
    def missing(**kwargs):
        raise ModuleNotFoundError("recorded provider is missing")

    monkeypatch.setattr(sonic, "factory", missing)
    loader = available.factory(phase="load", device="cuda:0", options={})
    with pytest.raises(ModuleNotFoundError, match="recorded provider"):
        loader.bind({"version": 1, "provider": "sonic", "manifest": {}}, lambda _: b"")
    with pytest.raises(ValueError, match="unsupported saved"):
        loader.bind(
            {"version": 1, "provider": "unknown", "manifest": {}}, lambda _: b""
        )
