"""MoonEP plan metadata and stream/EP phase synchronization."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import fields, is_dataclass

import torch
from torch import Tensor

_PLAN_FIELDS = (
    "dst",
    "experts_to_copy",
    "zero_fill_ranges",
    "remote_stats",
    "dup_groups",
    "dup_loffs",
    "dup_counts",
)
_STATE_LEN = 13


def _walk_tensors(value):
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _walk_tensors(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from _walk_tensors(v)
    elif is_dataclass(value) and not isinstance(value, type):
        for f in fields(value):
            yield from _walk_tensors(getattr(value, f.name))


class _CudaPhases:
    """One compute stream, one communication lane, EP-wide join EVERY phase.

    All MoonEP operations (including its internal barriers) and explicit phase
    barriers are launched on one lane. Never concurrently reuse the same MoonEP
    meta/grid-sync counters from separate streams.
    """

    def __init__(self, runtime, inputs):
        self.r = runtime
        self.cuda = getattr(
            runtime, "_cuda", torch.cuda
        )  # Test-only CUDA dependency injection.
        self.main = self.cuda.current_stream(runtime.device)
        self.comm = (
            runtime.buffer._comm_stream if runtime.overlap_enabled else self.main
        )
        # record_stream protects asynchronous uses without extending Python
        # ownership. Retention is an explicit, bounded debugging facility.
        self.live = []
        self.retained = {} if runtime.cfg.retain_intermediates else None
        runtime.retained_intermediates = self.retained
        self.events = []
        self._record(inputs, "inputs")
        # Input producers and any previous call's returned-stream work precede entry.
        self.comm.wait_event(self.main.record_event())
        self.phase("entry")

    def _record(self, value, label):
        if self.retained is not None:
            self.retained[label] = value
        for t in _walk_tensors(value):
            if self.retained is not None:
                self.live.append(t)
            if t.is_cuda:
                # MoonEP VMM allocations live until synchronized runtime close;
                # they must not be registered with PyTorch's caching allocator.
                if any(
                    base <= t.data_ptr() < base + size
                    for base, size in getattr(self.r, "external_storage_extents", ())
                ):
                    continue
                t.record_stream(self.main)
                if self.comm != self.main:
                    t.record_stream(self.comm)

    @contextmanager
    def _range(self, label):
        if not self.r.cfg.profile_ranges:
            yield
            return
        with (
            torch.profiler.record_function("moon_te/" + label),
            self.cuda.nvtx.range("moon_te/" + label),
        ):
            yield

    def phase(
        self,
        label,
        compute: Callable | None = None,
        communication: Callable | None = None,
    ):
        # The previous phase already put a global-release dependency on BOTH lanes.
        # Run communication first on the host so it has an early launch opportunity.
        comm_value = compute_value = None
        hook = getattr(self.r, "_test_delay_hook", None)
        if hook is not None:
            hook(label, self.main, self.comm)
        if communication is not None:
            with self.cuda.stream(self.comm), self._range(label + "/comm"):
                comm_value = communication()
            self._record(comm_value, label + "/comm")
        if compute is not None:
            with self.cuda.stream(self.main), self._range(label + "/compute"):
                compute_value = compute()
            self._record(compute_value, label + "/compute")
        compute_done = self.main.record_event()
        # COMM currently ends after this phase's comm kernels. Waiting on the
        # compute event joins BOTH local lanes before entering the group barrier.
        with self.cuda.stream(self.comm):
            self.comm.wait_event(compute_done)
            with self._range(label + "/group_join"):
                self.r.rank_sync(self.r.ctx)
            released = self.comm.record_event()
        self.main.wait_event(released)
        self.events.extend((compute_done, released))
        return compute_value, comm_value
