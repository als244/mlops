"""Quack's tuning sweeps inside a bounded allocator (``quack_runtime.sweeps``).

A sweep copies the call's tensors only as far as L2 rotation needs; when one
set is already large it times the call's own tensors and restores them after
each candidate, so an in-place GEMM's output survives tuning; and a sweep in
which every candidate failed is neither cached nor read back.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from mlops.quack_runtime import initialize, sweeps

INF = float("inf")


@pytest.fixture(autouse=True)
def _patched():
    pytest.importorskip("quack")
    initialize()


def _l2(monkeypatch, size):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(L2_cache_size=size),
    )


def _bytes(count):
    return torch.empty(count, dtype=torch.uint8, device="meta")


def test_a_round_takes_the_sets_three_l2s_need_and_no_floor(monkeypatch):
    mib = 1 << 20
    _l2(monkeypatch, 50 * mib)
    count = sweeps.rotation_count
    assert count((_bytes(mib),), {}) == sweeps.MAX_SETS
    assert count((_bytes(10 * mib),), {"out": _bytes(20 * mib)}) == 5
    assert count((_bytes(100 * mib),), {}) == 2
    # One set already moves three L2s: the call's own tensors, no copy.
    assert count((_bytes(150 * mib),), {}) == 1
    assert count((_bytes(4096 * mib),), {"c": _bytes(4096 * mib)}) == 1
    assert count((1, "x"), {}) == 1


def test_one_set_is_the_calls_own_tensors_restored_after_each_candidate():
    generator = torch.Generator().manual_seed(0)
    out = torch.randn(6, 8, generator=generator)
    base = torch.randn(4, 16, generator=generator)
    view = base.t()  # strided
    tail = base[1:3]  # offset into the same storage
    bias = torch.randn(8, generator=generator).expand(6, 8)  # stride 0, read only
    args = (out, view, 3)
    kwargs = {"c": out, "tail": tail, "bias": bias}
    before = [tensor.clone() for tensor in (out, base, bias)]

    arg_sets, kwarg_sets = sweeps.rotation_sets(args, kwargs, 1)
    assert isinstance(arg_sets, sweeps.RotationSets)
    assert arg_sets[0][0] is out and kwarg_sets[0]["tail"] is tail

    def scribble(fn, sets, kwarg_sets, extra_kwargs, **_):
        args, kwargs = sets[0], kwarg_sets[0]
        args[0].add_(100.0)  # an in-place GEMM accumulating into its output
        kwargs["c"].mul_(-3.0)
        args[1].fill_(7.0)
        kwargs["tail"].zero_()
        return [1.0]

    bench = sweeps.restoring(scribble)
    for _ in range(3):
        assert bench(None, arg_sets, kwarg_sets, extra_kwargs={}) == [1.0]
        for tensor, expected in zip((out, base, bias), before, strict=True):
            assert torch.equal(tensor, expected)


def test_a_failing_candidate_still_restores():
    out = torch.ones(4)
    arg_sets, kwarg_sets = sweeps.rotation_sets((out,), {}, 1)

    def failing(fn, sets, kwarg_sets, extra_kwargs, **_):
        sets[0][0].fill_(-1.0)
        raise RuntimeError("smem overflow")

    with pytest.raises(RuntimeError, match="smem overflow"):
        sweeps.restoring(failing)(None, arg_sets, kwarg_sets, extra_kwargs={})
    assert torch.equal(out, torch.ones(4))


def test_more_sets_are_copies():
    out = torch.zeros(4, 4)
    arg_sets, kwarg_sets = sweeps.rotation_sets((out,), {"c": out}, 3)
    assert not isinstance(arg_sets, sweeps.RotationSets)
    assert len(arg_sets) == len(kwarg_sets) == 3
    pointers = {out.data_ptr()}
    for args, kwargs in zip(arg_sets, kwarg_sets, strict=True):
        assert torch.equal(args[0], out)
        pointers.update((args[0].data_ptr(), kwargs["c"].data_ptr()))
    assert len(pointers) == 7


class _Config:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"config: {self.name}"


def _tuner(name="mod_dgated_epi_test"):
    def function():
        pass

    function.__name__ = name
    return SimpleNamespace(fn=function, cache={}, configs_timings=None, bench_time=None)


def _cached(directory):
    return sorted(directory.rglob("*.autotune.json"))


def test_a_failed_sweep_is_not_cached_and_a_cached_failure_is_tuned_again(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("QUACK_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("QUACK_FORCE_CACHE_UPDATE", raising=False)
    configs = [_Config("a"), _Config("b"), _Config("c")]
    key = (("shape", 262144, 2048), torch.bfloat16)
    tuner = _tuner()
    sweeps_run = []

    def sweep(timings):
        def bench():
            sweeps_run.append(timings)
            tuner.configs_timings = dict(zip(configs, timings, strict=True))
            tuner.cache[key] = min(tuner.configs_timings, key=tuner.configs_timings.get)

        return bench

    failed = [[INF] * 3] * 3
    sweeps.check_disk_cache(tuner, key, configs, sweep(failed))
    assert sweeps_run == [failed] and _cached(tmp_path) == []

    # Nothing cached, so the next process sweeps again; this one succeeds.
    tuned = [[2.0] * 3, [1.0] * 3, [INF] * 3]
    sweeps.check_disk_cache(tuner, key, configs, sweep(tuned))
    (path,) = _cached(tmp_path)
    tuner.cache.clear()
    sweeps.check_disk_cache(tuner, key, configs, sweep(None))
    assert len(sweeps_run) == 2 and tuner.cache[key] is configs[1]

    # An all-failed result already on disk is a miss, not a choice.
    path.write_text(
        json.dumps(
            {"key": str(key), "configs_timings": [(str(c), [INF] * 3) for c in configs]}
        )
    )
    tuner.cache.clear()
    sweeps.check_disk_cache(tuner, key, configs, sweep(tuned))
    assert len(sweeps_run) == 3 and tuner.cache[key] is configs[1]
    timings = json.loads(path.read_text())["configs_timings"]
    assert [timing[0] for _, timing in timings] == [2.0, 1.0, INF]


def test_the_runtime_installs_the_sweeps():
    import quack.autotuner as tuner

    assert tuner._pick_l2_rotate_count is sweeps.rotation_count
    assert tuner._clone_l2_rotate_inputs is sweeps.rotation_sets
    assert tuner.Autotuner.check_disk_cache.__wrapped__ is sweeps.check_disk_cache
    assert sweeps.apply() is False


def _hopper():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 9


hopper = pytest.mark.skipif(not _hopper(), reason="needs an SM90 GPU and quack")


def _tune_in_place(rows, width, monkeypatch, tmp_path):
    """``gemm_add_inplace`` tuned from a cold cache: ``(out, expected, set bytes,
    peak bytes above the operands)``."""
    from quack.gemm_interface import gemm_add_inplace

    monkeypatch.setenv("QUACK_CACHE_DIR", str(tmp_path))
    generator = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(rows, width, device="cuda", generator=generator).bfloat16()
    b = (
        torch.randn(width, width, device="cuda", generator=generator) / width**0.5
    ).bfloat16()
    out = torch.randn(rows, width, device="cuda", generator=generator).bfloat16()
    expected = out.float() + a.float() @ b.float()
    set_bytes = sum(t.numel() * t.element_size() for t in (a, b, out))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    gemm_add_inplace(a, b, out)
    torch.cuda.synchronize()
    return out, expected, set_bytes, torch.cuda.max_memory_allocated() - base


@pytest.mark.gpu
@hopper
def test_a_large_in_place_gemm_tunes_on_its_own_output_and_keeps_it(
    monkeypatch, tmp_path
):
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    width = 4096
    # Rows enough that one set (a, b, out) moves more than three L2s.
    rows = -(-(sweeps.L2_ROUNDS * l2 // (4 * width) + width) // 256) * 256 + 128
    out, expected, set_bytes, peak = _tune_in_place(rows, width, monkeypatch, tmp_path)
    assert set_bytes >= sweeps.L2_ROUNDS * l2
    error = (out.float() - expected).norm() / expected.norm()
    assert error < 1e-2, error
    # No copy set (stock Quack makes at least four): the sweep allocates only
    # its thermal warm-up, two 4096 x 4096 bf16 matrices, under one set.
    assert peak < set_bytes // 2, (peak, set_bytes)
    from quack.gemm_interface import gemm_tuned

    if gemm_tuned.cache_results:  # the tuner was made after initialize()
        (path,) = _cached(tmp_path)
        timings = json.loads(path.read_text())["configs_timings"]
        assert any(timing[0] != INF for _, timing in timings)


@pytest.mark.gpu
@hopper
def test_a_small_in_place_gemm_tunes_on_copies_and_keeps_its_output(
    monkeypatch, tmp_path
):
    out, expected, _, _ = _tune_in_place(384, 256, monkeypatch, tmp_path)
    error = (out.float() - expected).norm() / expected.norm()
    assert error < 1e-2, error
