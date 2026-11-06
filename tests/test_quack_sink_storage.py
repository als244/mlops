"""Cold and cached epilogue calls must preserve reduction-buffer storage."""

from types import SimpleNamespace

import pytest
import torch


def test_cached_reduction_keeps_cold_backing_and_view(monkeypatch):
    pytest.importorskip("quack")
    from mlops.quack_runtime import initialize

    initialize()
    from quack.epilogue.frontend import EpiMod, gemm_epilogue
    from quack.epilogue.ops import ColVecReduce
    from quack.gemm_runtime import autotune

    @gemm_epilogue(reduces={"total": ColVecReduce("total")})
    def reduction(acc):
        return {"D": acc, "total": acc}

    calls = []
    tunings = []
    globals_ = EpiMod.__call__.__globals__
    monkeypatch.setitem(globals_, "get_device_capacity", lambda device: (9, 0))
    monkeypatch.setattr(
        autotune, "sink_arg_shapes", lambda mod, m, n, **kwargs: {"total": (m, 8)}
    )

    def launch(plan, a, b, out, c, operands, **kwargs):
        partials = operands["total"]
        out.copy_(a @ b)
        partials.copy_(torch.stack([half.sum(-1) for half in out.chunk(2, -1)], -1))
        calls.append(
            (
                partials.untyped_storage().nbytes(),
                partials.shape,
                partials.stride(),
                partials.storage_offset(),
            )
        )

    def tuned(mod, a, b, out, c, *, epi_args, **kwargs):
        tunings.append(True)
        # The candidate sweep owns eight partial columns; its selected kernel
        # uses only two. Exercise the real frontend cache around a CPU launcher.
        partials = epi_args["total"][:, :2]
        plan = SimpleNamespace()
        launch(plan, a, b, out, c, {"total": partials})
        return SimpleNamespace(
            sinks={"total": partials},
            plan=plan,
            config=SimpleNamespace(swap_ab=False, is_dynamic_persistent=False),
        )

    monkeypatch.setitem(globals_, "run_gemm_epi_plan", launch)
    monkeypatch.setattr(autotune, "tuned_mod_gemm", tuned)
    a = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 8
    b = torch.arange(64, dtype=torch.float32).reshape(4, 16) / 16
    expected = (a @ b).sum(-1)
    for _ in range(3):
        result = reduction(a, b, dynamic_scheduler=False)
        torch.testing.assert_close(result["total"], expected)
    assert len(tunings) == 1  # Keep the ordinary fast path after the cold call.
    assert calls == [(256, (8, 2), (8, 1), 0)] * 3


@pytest.mark.parametrize("kind", ["offset_transpose", "contiguous", "empty"])
def test_recipe_restores_storage_and_layout_without_retaining_tensor(kind):
    from mlops.quack_runtime.sink_storage import _allocate, _recipe

    backing = torch.empty(7, 11)
    value = {
        "offset_transpose": backing[2:6, 3:8].T,
        "contiguous": backing,
        "empty": backing[:0],
    }[kind]
    recipe = _recipe(value)
    restored = _allocate(recipe, "cpu")
    assert restored.shape == value.shape
    assert restored.stride() == value.stride()
    assert restored.storage_offset() == value.storage_offset()
    assert restored.untyped_storage().nbytes() == backing.untyped_storage().nbytes()
    assert all(not isinstance(item, torch.Tensor) for item in recipe)
