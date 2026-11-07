"""Cold sparse-forward bindings preserve measured LSE conventions and dispatch."""

import pytest

from mlops.providers.builtin import sparse_latent_hopper as provider
from mlops.providers.builtin import sparse_latent_preparation as preparation


@pytest.fixture
def installed(monkeypatch):
    function = object()
    monkeypatch.setattr(preparation, "_installed_flashmla", lambda: function)
    monkeypatch.setattr(
        preparation, "_identity", lambda *args: {"implementation": "pinned"}
    )
    return function


def forbidden():
    raise AssertionError("cold loading or execution performed GPU detection")


@pytest.mark.parametrize("factor", [None, 1.0, provider._LOG2E])
def test_cold_bind_preserves_backend_and_factor_without_eager_detection(
    monkeypatch, installed, factor
):
    entry = None if factor is None else (installed, factor)
    monkeypatch.setattr(provider, "_detect_flashmla", lambda: entry)
    builder = preparation.factory(phase="build", device="cuda:0", options={})
    with builder.capture() as captured:
        assert provider._flashmla() == entry
    manifest, blobs = builder.export(captured)
    builder.close()
    assert not blobs
    monkeypatch.setattr(provider, "_detect_flashmla", forbidden)
    loader = preparation.factory(phase="load", device="cuda:0", options={})
    binding = loader.bind(manifest, forbidden)
    with binding.activate():
        assert provider._flashmla() == entry
        with binding.activate():
            assert provider._flashmla() == entry
        assert provider._flashmla() == entry
    with pytest.raises(AssertionError, match="GPU detection"):
        provider._flashmla()
    loader.close()
    with pytest.raises(RuntimeError, match="closed"), binding.activate():
        pass


def test_empty_task_needs_no_backend_and_refuses_an_uncaptured_call(monkeypatch):
    monkeypatch.setattr(preparation, "_installed_flashmla", forbidden)
    monkeypatch.setattr(provider, "_detect_flashmla", forbidden)
    builder = preparation.factory(phase="build", device="cuda:0", options={})
    with builder.capture() as captured:
        pass
    manifest, _ = builder.export(captured)
    loader = preparation.factory(phase="load", device="cuda:0", options={})
    with (
        loader.bind(manifest, forbidden).activate(),
        pytest.raises(ValueError, match="not prepared"),
    ):
        provider._flashmla()


def test_changed_implementation_or_unknown_lse_convention_refuses(
    monkeypatch, installed
):
    monkeypatch.setattr(provider, "_detect_flashmla", lambda: (installed, 1.0))
    builder = preparation.factory(phase="build", device="cuda:0", options={})
    with builder.capture() as captured:
        provider._flashmla()
    manifest, _ = builder.export(captured)
    monkeypatch.setattr(
        preparation, "_identity", lambda *args: {"implementation": "different"}
    )
    loader = preparation.factory(phase="load", device="cuda:0", options={})
    with pytest.raises(ValueError, match="implementation differs"):
        loader.bind(manifest, forbidden)
    manifest["selection"]["lse_to_log2"] = 2.0
    with pytest.raises(ValueError, match="LSE convention"):
        loader.bind(manifest, forbidden)
