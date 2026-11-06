"""Exercise the real JITFunction.run/binder with a CPU compiler stand-in."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import triton
import triton.language as tl

from mlops.triton_runtime._scope import TritonCalls


@triton.jit
def _kernel(x, N: tl.constexpr):
    pass


@pytest.fixture
def function(monkeypatch):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import compiler
    from triton.runtime.driver import driver

    target = GPUTarget("cuda", 90, 32)
    current = SimpleNamespace(
        get_current_target=lambda: target,
        get_current_device=lambda: 0,
        get_current_stream=lambda device: 0,
    )
    monkeypatch.setattr(driver, "_active", current)
    built = []

    def compile(src, **kwargs):
        value = SimpleNamespace(src=src, **kwargs)
        built.append(value)
        return value

    monkeypatch.setattr(compiler, "compile", compile)
    monkeypatch.setattr(triton.compiler, "compile", compile)
    return triton.JITFunction(_kernel.fn), built, current


def launch(function, n=32):
    return function.run(3, n, grid=(1,), warmup=True)


def test_capture_includes_warm_alias_and_cold_calls(function):
    fn, built, _ = function
    alias = fn
    old = launch(alias)
    with TritonCalls((fn,)).capture() as selected:
        assert launch(alias) is old
        new = launch(alias, 17)
    assert len(selected) == len(built) == 2
    assert list(selected.values()) == [old, new]


def test_task_replay_refuses_warm_other_specialization_and_device(function):
    fn, built, current = function
    calls = TritonCalls((fn,))
    with calls.capture() as selected:
        old = launch(fn)
    launch(fn, 17)
    prepared = object()
    binding = calls.bind({key: prepared for key in selected}, device=0)
    for _ in range(2):
        with binding.activate():
            assert launch(fn) is prepared
            with pytest.raises(ValueError, match="missing prepared Triton"):
                launch(fn, 17)
    with (
        calls.bind({}, device=0).activate(),
        pytest.raises(ValueError, match="missing prepared Triton"),
    ):
        launch(fn)
    with binding.activate():
        current.get_current_device = lambda: 1
        with pytest.raises(ValueError, match="another device"):
            launch(fn)
    current.get_current_device = lambda: 0
    assert launch(fn) is old and len(built) == 2


def test_nested_threads_and_failure_preserve_original_dispatch(function):
    fn, built, _ = function
    calls = TritonCalls((fn,))
    with calls.capture() as selected:
        original = launch(fn)
    left, right = object(), object()
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(launch, fn).result() is original
        with calls.bind(dict.fromkeys(selected, left), device=0).activate():
            assert launch(fn) is left
            assert pool.submit(launch, fn).result() is original
            with (
                pytest.raises(RuntimeError, match="failure"),
                calls.bind(dict.fromkeys(selected, right), device=0).activate(),
            ):
                assert launch(fn) is right
                raise RuntimeError("failure")
            assert launch(fn) is left
    assert launch(fn) is original and len(built) == 1


def test_reject_ambiguous_function_and_preserve_unregistered_alias(function):
    fn, _, _ = function
    other = triton.JITFunction(fn.fn)
    calls = TritonCalls((fn,))
    with calls.capture() as selected:
        launch(fn)
    eager_other = launch(other)
    with calls.bind(dict.fromkeys(selected, object()), device=0).activate():
        assert launch(other) is eager_other
    with pytest.raises(ValueError, match="distinct Triton"):
        TritonCalls((fn, other))
    with pytest.raises(TypeError, match="plain JITFunctions"):
        TritonCalls((lambda: None,))
