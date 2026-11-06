"""Actual autotuner control flow: one final mutation and only the winner captured."""

import importlib
from contextlib import ExitStack
from types import SimpleNamespace

import pytest
import torch

from mlops.quack_runtime import _benchmark_scope
from mlops.quack_runtime import _cache_scope as quack
from mlops.triton_runtime import _scope as triton


@pytest.mark.parametrize("backend", ["quack", "triton"])
def test_benchmark_hooks_restore_inputs_and_capture_only_final_winner(
    monkeypatch, backend
):
    _benchmark_scope.install()
    costs = iter(([3.0, 3.0, 3.0], [1.0, 1.0, 1.0]))
    calls = []
    selected = []

    def kernel(dst, alpha, **kwargs):
        calls.append(alpha)
        if quack._active.get() is not None or triton._active.get() is not None:
            selected.append(alpha)
        dst.add_(alpha)

    def bench(function, **kwargs):
        for _ in range(3):
            function()
        return next(costs)

    if backend == "quack":
        module = importlib.import_module("quack.autotuner")
        pool = importlib.import_module("quack.cache.async_compile")
        monkeypatch.setattr(module, "_gpu_warmup", lambda: None)
        monkeypatch.setattr(pool, "pool_scope", pool.suppress_pool)
        tuner = module.Autotuner(
            kernel,
            [],
            [module.AutotuneConfig(alpha=n) for n in (1, 3)],
            restore_value=["dst"],
            do_bench=bench,
        )
        tuner.cache_results = False
        invoke = tuner
    else:
        import triton as compiler
        from triton.runtime.autotuner import Autotuner

        tuner = Autotuner(
            SimpleNamespace(fn=kernel, run=kernel),
            ["dst"],
            [compiler.Config({"alpha": n}) for n in (1, 3)],
            [],
            reset_to_zero=[],
            restore_value=["dst"],
            do_bench=bench,
        )
        invoke = tuner.run
    destination = torch.tensor(7.0)
    with ExitStack() as cleanup:
        for variable, scope in (
            (quack._active, quack._Scope(frozenset(), recorded={})),
            (triton._active, triton._Scope(frozenset(), recorded={})),
        ):
            cleanup.callback(variable.reset, variable.set(scope))
        invoke(destination)
        assert destination.item() == 10
        assert calls == [1, 1, 1, 3, 3, 3, 3]
        assert selected == [3]
        invoke(destination)  # Warm selected config, still captured exactly once.
        assert destination.item() == 13 and selected == [3, 3]
    assert quack._active.get() is triton._active.get() is None


def test_replay_benchmark_refuses_without_suspending_bound_scope():
    scope = triton._Scope(frozenset(), prepared={})
    token = triton._active.set(scope)
    try:
        with pytest.raises(RuntimeError, match="cannot benchmark"):
            _benchmark_scope._bench(lambda: pytest.fail("benchmark entered"))
        assert triton._active.get() is scope
    finally:
        triton._active.reset(token)
