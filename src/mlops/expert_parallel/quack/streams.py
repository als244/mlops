"""CUDA streams, tensor lifetimes and NVTX scopes for the single scheduler."""

from contextlib import contextmanager

import torch

from .communication import _walk_tensors


class Streams:
    def __init__(self, runtime, inputs):
        self.r = runtime
        self.main = torch.cuda.current_stream(runtime.device)
        self.comm = (
            runtime.buffer._comm_stream if runtime.overlap_enabled else self.main
        )
        self.transfer = (
            runtime.transfer_stream if runtime.overlap_enabled else self.main
        )
        self.lanes = tuple(dict.fromkeys((self.main, self.comm, self.transfer)))
        self.record(inputs)
        entry = self.main.record_event()
        for stream in self.lanes[1:]:
            stream.wait_event(entry)

    def record(self, value):
        for tensor in _walk_tensors(value):
            if tensor.is_cuda and not any(
                base <= tensor.data_ptr() < base + size
                for base, size in self.r.external_storage_extents
            ):
                for stream in self.lanes:
                    tensor.record_stream(stream)
        return value

    @contextmanager
    def annotation(self, label, lane):
        """Nested chunk/lane attribution without introducing device waits."""
        if not self.r.cfg.profile_ranges:
            yield
            return
        parts = label.split(".", 2)
        if len(parts) == 3 and parts[1].isdigit():
            with (
                torch.cuda.nvtx.range(
                    f"moon_quack/{parts[0]}/chunk{int(parts[1]):03d}"
                ),
                torch.cuda.nvtx.range(f"moon_quack/chunk_stream/{lane}"),
                torch.cuda.nvtx.range("moon_quack/chunks/" + label),
            ):
                yield
        else:
            with torch.cuda.nvtx.range("moon_quack/chunks/" + label):
                yield

    @contextmanager
    def range(self, label, *, communication=False):
        stream = self.comm if communication else self.main
        with torch.cuda.stream(stream):
            hook = getattr(self.r, "_test_delay_hook", None)
            if hook is not None:
                hook(label, self.main, self.comm)
            with self.annotation(
                label, "local_communication" if communication else "compute"
            ):
                yield

    @contextmanager
    def bulk(self, label):
        with torch.cuda.stream(self.transfer), self.annotation(label, "bulk_transfer"):
            yield

    def finish(self, value):
        for stream in reversed(self.lanes[1:]):
            self.main.wait_event(stream.record_event())
        self.record(value)
