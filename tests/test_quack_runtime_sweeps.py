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


def _replaced(monkeypatch):
    """An allocator that replaces PyTorch's, as a runtime installing its own
    pool as the process allocator does."""
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "pluggable")


def test_a_replacement_allocator_holds_no_copies(monkeypatch):
    mib = 1 << 20
    _l2(monkeypatch, 50 * mib)
    assert sweeps.rotation_count((_bytes(mib),), {}) == sweeps.MAX_SETS
    _replaced(monkeypatch)
    assert sweeps.rotation_count((_bytes(mib),), {}) == 1


def test_a_replacement_allocator_times_candidates_without_a_graph(monkeypatch):
    """A capture would hand the replacement its allocations mid-capture, and
    its own threads' event polls would break it: the calls run eagerly."""
    _replaced(monkeypatch)
    launches = []

    class Event:
        # Elapsed time is the calls launched between two records, 1 ms each.
        def __init__(self, enable_timing=False):
            self.at = None

        def record(self):
            self.at = len(launches)

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return float(other.at - self.at)

    def refuse(*args, **kwargs):
        raise AssertionError("a replacement allocator must not capture a CUDA graph")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", refuse)
    monkeypatch.setattr(torch.cuda, "graph", refuse)
    timing = sweeps.benchmark_rotation(
        lambda *args, **kwargs: launches.append(args),
        [(1,)],
        [{}],
        {},
        warmup_target_ms=2,
        timed_target_ms=3,
        samples=3,
    )
    assert timing == pytest.approx(1.0) and len(launches) > 4


def test_a_replacement_allocators_sweep_stays_in_its_process(monkeypatch, tmp_path):
    monkeypatch.setenv("QUACK_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("QUACK_FORCE_CACHE_UPDATE", raising=False)
    configs = [_Config("a"), _Config("b")]
    key = (("shape", 4096, 512), torch.bfloat16)
    tuner = _tuner()

    def bench():
        tuner.configs_timings = dict(zip(configs, [[2.0] * 3, [1.0] * 3], strict=True))
        tuner.cache[key] = min(tuner.configs_timings, key=tuner.configs_timings.get)

    _replaced(monkeypatch)
    sweeps.check_disk_cache(tuner, key, configs, bench)
    # The winner serves this process; the disk keeps graph-timed sweeps only.
    assert tuner.cache[key] is configs[1] and _cached(tmp_path) == []
    monkeypatch.undo()
    monkeypatch.setenv("QUACK_CACHE_DIR", str(tmp_path))
    tuner.cache.clear()
    sweeps.check_disk_cache(tuner, key, configs, bench)
    assert len(_cached(tmp_path)) == 1


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


def test_concurrent_cache_misses_publish_one_choice(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from time import sleep

    monkeypatch.setenv("QUACK_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("QUACK_FORCE_CACHE_UPDATE", raising=False)
    barrier = Barrier(4)
    sweeps_run = []

    def caller(index):
        local = _tuner()
        configs = [_Config("a"), _Config("b")]
        key = ("concurrent", 256, 64)

        def bench():
            sweeps_run.append(index)
            # All callers reach the same miss while the first tunes.
            sleep(0.1)
            local.configs_timings = {
                config: [1.0 if i == index % 2 else 2.0]
                for i, config in enumerate(configs)
            }
            local.cache[key] = min(local.configs_timings, key=local.configs_timings.get)

        barrier.wait(timeout=10)
        sweeps.check_disk_cache(local, key, configs, bench)
        return str(local.cache[key])

    with ThreadPoolExecutor(max_workers=4) as pool:
        choices = list(pool.map(caller, range(4)))
    assert len(sweeps_run) == 1
    assert len(set(choices)) == 1


def test_adaptive_timer_validates_budgets_and_supports_long_targets(monkeypatch):
    assert sweeps._positive_setting(None, "MLOPS_QUACK_TUNE_TIME_MS", 2000) == 2000
    monkeypatch.setenv("MLOPS_QUACK_TUNE_TIME_MS", "4000")
    assert sweeps._positive_setting(None, "MLOPS_QUACK_TUNE_TIME_MS", 2000) == 4000
    for value in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="positive"):
            sweeps._positive_setting(value, "budget", 2000)
    with pytest.raises(ValueError, match="positive integer"):
        sweeps._positive_setting(1.5, "samples", 3, integer=True)


def test_capture_size_is_bounded_and_preserves_complete_rotations():
    # Fast kernels need extra replays, not an unbounded captured graph.
    assert sweeps.calls_for_budget(.001, 50, 4) == 256
    # A 76 ms kernel needs one execution per graph, without a 50-call floor.
    assert sweeps.calls_for_budget(76, 50, 1) == 1
    for sets in (1, 3, 4, 16):
        calls = sweeps.calls_for_budget(.01, 50, sets)
        assert calls % sets == 0 and sets <= calls <= 256


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_adaptive_timer_runs_requested_budgets_and_keeps_matmul_output():
    from time import perf_counter

    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    expected = a @ b
    torch.cuda.synchronize()
    started = perf_counter()
    timings = sweeps.benchmark_rotation(
        torch.mm, [(a, b)], [{"out": out}], {},
        warmup_target_ms=50, timed_target_ms=100, samples=3,
        quantiles=(.5, .2, .8),
    )
    elapsed = perf_counter() - started
    # Measured CUDA budgets alone require at least 150 ms, even when the
    # bounded graph is too short to cover the budget in one replay.
    assert elapsed >= .15
    assert 0 < timings[1] <= timings[0] <= timings[2]
    torch.testing.assert_close(out, expected)


def test_overwrite_gemm_makes_no_backups_but_aliases_are_restored():
    from quack.gemm_interface import gemm_tuned

    from mlops.quack_runtime.preservation import tensors_to_preserve
    a, b, out = torch.randn(4, 4), torch.randn(4, 4), torch.zeros(4, 4)
    assert tensors_to_preserve(gemm_tuned.fn, (a, b, out), {}) == []
    saved = tensors_to_preserve(gemm_tuned.fn, (a, b, out), {"C": out})
    assert len(saved) == 1 and saved[0] is out
    saved = tensors_to_preserve(gemm_tuned.fn, (a, b, out), {"add_to_output": True})
    assert len(saved) == 1 and saved[0] is out
    # A read operand can overlap an output without being the same Tensor.
    saved = tensors_to_preserve(gemm_tuned.fn, (a, b, out), {"C": out[1:3]})
    assert len(saved) == 1 and saved[0] is out
    # Disjoint slices of the same storage do not need restoration.
    storage = torch.zeros(32)
    assert tensors_to_preserve(gemm_tuned.fn, (a, b, storage[:16]), {"C": storage[16:]}) == []


def test_mod_contract_keeps_readwrite_sinks_and_output_input_aliases():
    from quack.epilogue.frontend import gemm_epilogue
    from quack.epilogue.ops import ColVecReduce
    from quack.gemm_runtime.autotune import _make_tuned_fn

    from mlops.quack_runtime.preservation import tensors_to_preserve

    @gemm_epilogue(outputs=("aux",), reduces={"total": ColVecReduce("total")})
    def epi(acc, c):
        return {"D": acc + c, "aux": acc, "total": acc}

    fn = _make_tuned_fn(epi, ("aux", "total"))
    a, b, out, aux = [torch.ones(4, 4) for _ in range(4)]
    total = torch.zeros(4)
    kwargs = {"A": a, "B": b, "D": out, "C": None, "aux": aux, "total": total}
    selected = tensors_to_preserve(fn, (), kwargs)
    assert len(selected) == 1 and selected[0] is total
    kwargs["C"] = aux
    selected = tensors_to_preserve(fn, (), kwargs)
    assert {id(x) for x in selected} == {id(aux), id(total)}


def test_pure_output_sweep_has_no_host_copy_and_uses_real_operands():
    from quack.gemm_interface import gemm_tuned
    a, b, out = [torch.ones(4, 4) for _ in range(3)]
    arg_sets, kwarg_sets = sweeps.rotation_sets((a, b, out), {}, 1)
    assert arg_sets.saved is None
    def bench(fn, sets, kwsets, **kwargs):
        sets[0][2].fill_(3)
        return [1.0]
    result = sweeps.restoring(bench)(gemm_tuned.fn, arg_sets, kwarg_sets, extra_kwargs={})
    assert result == [1.0] and arg_sets.saved._copies == []
    assert torch.equal(out, torch.full_like(out, 3))
    assert torch.equal(a, torch.ones_like(a))


def test_compact_sm90_additions_preserve_baseline_and_respect_cooperative_flag():
    from quack import gemm_config as gc

    from mlops.quack_runtime.autotune import _SM90_MEASURED_ADDITIONS
    wrapped = gc._get_sm90_configs
    original = wrapped.__wrapped__
    for epilogue in (None, "gated", "lse"):
        for tune_coop in (False, True):
            before=set(original(epilogue, tune_coop))
            after=wrapped(epilogue, tune_coop)
            assert before <= set(after) and len(after) == len(set(after))
            # Existing N=96 additions plus the nine concrete measured choices.
            assert len(set(after)-before) <= 13
            for m,n,pp,cm,cn in _SM90_MEASURED_ADDITIONS:
                if not tune_coop and not pp: continue
                assert any(c.tile_m==m and c.tile_n==n and c.pingpong==pp
                           and c.cluster_m==cm and c.cluster_n==cn
                           and not c.swap_ab and c.device_capacity==9 for c in after)
            if not tune_coop:
                assert all(c.pingpong for c in set(after)-before)
