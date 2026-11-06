"""Task isolation and fail-closed dispatch through real Quack cache wrappers."""

import importlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from mlops.quack_runtime._cache_scope import QuackCalls


@pytest.fixture
def compiler(monkeypatch):
    cache = pytest.importorskip("quack.cache")
    monkeypatch.setattr(cache, "CACHE_ENABLED", False)
    compiled = []

    @cache.jit_cache
    def build(shape, *, mode=1):
        value = object()
        compiled.append((shape, mode, value))
        return value

    return build, compiled


def test_capture_includes_warm_aliases_and_new_calls(compiler):
    function, compiled = compiler
    alias = function
    warm = alias((7, 4), mode=2)
    with QuackCalls((function,)).capture() as calls:
        assert alias((7, 4), mode=2) is warm
        cold = alias((3, 4))
        assert alias((3, 4)) is cold
    assert len(calls) == 2 and set(calls.values()) == {warm, cold}
    assert len(compiled) == 2
    assert function.cache_info().currsize == 2
    function.cache_clear()
    assert function.cache_info().currsize == 0


def test_replay_refuses_missing_even_if_global_cache_has_it(compiler):
    function, compiled = compiler
    bindings = QuackCalls((function,))
    with bindings.capture() as calls:
        original = function((7, 4))
    function((3, 4))
    replacement = object()
    prepared = {key: replacement for key in calls}
    with bindings.use(prepared):
        assert function((7, 4)) is replacement
        with pytest.raises(ValueError, match="missing prepared Quack specialization"):
            function((3, 4))
    assert function((7, 4)) is original
    assert len(compiled) == 2


def test_nested_scopes_threads_and_failure_restore_eager_behavior(compiler):
    function, compiled = compiler
    bindings = QuackCalls((function,))
    with bindings.capture() as calls:
        original = function((7, 4))
    left, right = object(), object()
    with ThreadPoolExecutor(max_workers=1) as pool:
        # Start the worker before the task-local bindings exist.
        assert pool.submit(function, (7, 4)).result() is original
        with bindings.use({key: left for key in calls}):
            assert function((7, 4)) is left
            assert pool.submit(function, (7, 4)).result() is original
            with (
                pytest.raises(RuntimeError, match="inner failure"),
                bindings.use({key: right for key in calls}),
            ):
                assert function((7, 4)) is right
                raise RuntimeError("inner failure")
            assert function((7, 4)) is left
    assert function((7, 4)) is original
    assert len(compiled) == 1


def test_refuse_unknown_decorator_and_unchecked_function(monkeypatch, compiler):
    function, _ = compiler
    module = importlib.import_module("quack.cache.jit")
    monkeypatch.setattr(module, "jit_cache", lambda f: f)
    with pytest.raises(ValueError, match="unsupported Quack binary cache contract"):
        QuackCalls((function,))
    monkeypatch.undo()
    with pytest.raises(ValueError, match="checked jit_cache functions"):
        QuackCalls((lambda: None,))


def test_scopes_select_function_objects_and_refuse_ambiguous_identities(monkeypatch):
    cache = pytest.importorskip("quack.cache")
    monkeypatch.setattr(cache, "CACHE_ENABLED", False)

    def factory():
        @cache.jit_cache
        def build(shape):
            return object()

        return build

    left, right = factory(), factory()
    binding = QuackCalls((left,))
    with binding.capture() as calls:
        original_left = left((7, 4))
    with QuackCalls((right,)).capture():
        original_right = right((7, 4))
    replacement = object()
    with binding.use({key: replacement for key in calls}):
        assert left((7, 4)) is replacement
        assert right((7, 4)) is original_right
    assert left((7, 4)) is original_left
    with pytest.raises(ValueError, match="distinct Quack functions"):
        QuackCalls((left, right))


def test_outer_dispatch_caches_cannot_bypass_task_bindings(compiler):
    function, compiled = compiler
    owner = SimpleNamespace(plans={})
    eager_cache = owner.plans
    builds = []

    def launch(shape):
        value = owner.plans.get(shape)
        if value is None:
            builds.append(shape)
            value = function(shape)
            owner.plans[shape] = value
        return value

    original = launch((7, 4))
    bindings = QuackCalls((function,), host_caches=((owner, "plans"),))
    with bindings.capture() as calls:
        assert launch((7, 4)) is original
    assert len(calls) == 1 and len(builds) == 2
    replacement = object()
    first = bindings.bind({key: replacement for key in calls})
    missing = bindings.bind({})
    for _ in range(2):
        with first.activate():
            assert launch((7, 4)) is replacement
            assert len(owner.plans) == 1
    assert len(builds) == 3  # second activation reuses only this binding's plan
    with (
        missing.activate(),
        pytest.raises(ValueError, match="missing prepared Quack specialization"),
    ):
        launch((7, 4))
    assert launch((7, 4)) is original
    assert eager_cache[(7, 4)] is original and len(compiled) == 1
