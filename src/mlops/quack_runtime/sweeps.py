"""Quack's tuning sweeps, fitted to a bounded allocator; installed Quack files
stay unchanged.

Quack's autotuner times each candidate config on copies of the call's
tensors, rotated so that L2 cannot serve them: at least four copy sets, as
many as half the device's free memory holds. Under an allocator that owns a
fixed pool -- a ShadowSpill execution pool -- the device's free memory says
nothing about the pool, and four copies of GiB-scale operands do not fit:
every candidate fails, the pool runs out of memory in the middle of a
measurement, and the sweep's all-infinite timings are cached, so every later
process runs the first-listed config untuned. Three changes:

* as many copy sets as it takes for one rotation round to move three L2s of
  bytes (at most 16), with no floor and no reading of the device's free
  memory, so the copies stay within a few L2 sizes;
* no copy when a single set already moves that much: the sweep times the
  call's own tensors, and a host copy of their bytes restores them after each
  candidate -- in-place GEMMs (``gemm_add_inplace``) read and write their
  output, so the real call must see it untouched;
* a sweep in which every candidate failed is neither written to the disk
  cache nor taken from it, so a later process tunes that shape again.
"""

import builtins
import hashlib
import json
import os
import sys

import torch

from .pipeline import _checked_source

_applied = False
_copy_sets = None

# What one rotation round moves, in L2 sizes, and the most copy sets a round
# takes: Quack's own targets.
L2_ROUNDS = 3
MAX_SETS = 16

_SOURCES = {
    "_pick_l2_rotate_count": (
        "6fa31ecddc5e6726ccfdf8c5e26979b629a02dcb9fd8dfd3da0a181fb802cf0c"
    ),
    "_clone_l2_rotate_inputs": (
        "08f0a2c67a354e05fab3a54fe8be58987f9edce0ad74a873f5d515faaa343d8d"
    ),
    "_bench_cuda_graph_l2_rotate": (
        "46fb36e9f43c6808ca803df60054f3dbc6058397321bd53df6e9311bf0a368f8"
    ),
    "check_disk_cache": (
        "c8313cf2cd92c243c5e637c28a1a42f3af60aabe9fc649f15423ea0cee4d2781"
    ),
    "_bench": "aafd26cb6730066f2a63ce3334902c3a43a8642cabc37ec45be560c58694f052",
}


def _tensors(args, kwargs):
    return [
        value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)
    ]


def rotation_count(args, kwargs, target_ratio=L2_ROUNDS, max_buffers=MAX_SETS):
    """Copy sets per rotation round: enough that a round moves
    ``target_ratio`` L2s of bytes, at most ``max_buffers``; one -- the call's
    own tensors, not copied -- when a single set already does."""
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in _tensors(args, kwargs)
    )
    if tensor_bytes == 0 or not torch.cuda.is_available():
        return 1
    l2 = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    return max(1, min(max_buffers, -(-target_ratio * l2 // tensor_bytes)))


def _byte_view(tensor):
    """The storage bytes ``tensor``'s elements occupy, as one flat uint8 view
    (strides are never negative, so the span runs from the first element's
    first byte to the last element's last)."""
    size = tensor.element_size()
    first = tensor.storage_offset() * size
    span = size * (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride())))
    view = torch.empty(0, dtype=torch.uint8, device=tensor.device)
    return view.set_(tensor.untyped_storage(), first, (span,), (1,))


class SavedTensors:
    """Host copies of the bytes a call's tensors occupy, and their restoration.

    Spans are saved once each, so a tensor passed twice (an in-place GEMM's
    output as both C and D) is copied once; overlapping spans are each
    restored from the same moment, so restoring them in turn agrees."""

    def __init__(self, args, kwargs):
        self._copies = []
        seen = set()
        for tensor in _tensors(args, kwargs):
            if tensor.numel() == 0 or tensor.device.type == "meta":
                continue
            view = _byte_view(tensor)
            span = (view.device, view.data_ptr(), view.numel())
            if span not in seen:
                seen.add(span)
                self._copies.append((view, view.to("cpu", copy=True)))

    def restore(self):
        for view, copy in self._copies:
            view.copy_(copy)


class RotationSets(list):
    """A sweep's rotation sets when the one set is the call's own tensors,
    with the copies that restore them after each candidate."""

    def __init__(self, sets, saved):
        super().__init__(sets)
        self.saved = saved


def rotation_sets(args, kwargs, n_buffers):
    """``n_buffers`` copies of the call's tensors, as Quack makes them; for
    one set, the call's own tensors, saved first (`RotationSets`)."""
    if n_buffers > 1:
        return _copy_sets(args, kwargs, n_buffers)
    saved = SavedTensors(args, kwargs)
    return RotationSets([tuple(args)], saved), [dict(kwargs)]


def restoring(bench):
    """``bench`` (Quack's rotation benchmark of one candidate), restoring the
    call's own tensors afterwards when they were the rotation set."""

    def bench_rotation(fn, arg_sets, kwarg_sets, *args, **kwargs):
        try:
            return bench(fn, arg_sets, kwarg_sets, *args, **kwargs)
        finally:
            if isinstance(arg_sets, RotationSets):
                arg_sets.saved.restore()

    return bench_rotation


def _failed(timings):
    """Whether every candidate of a sweep failed: Quack times a failure as
    infinite."""
    return all(timing[0] == float("inf") for timing in timings)


def check_disk_cache(self, tuning_key, configs, bench_fn):
    """Quack's ``Autotuner.check_disk_cache``, except that a sweep in which
    every candidate failed is neither taken from the disk cache nor written
    to it."""
    import quack.autotuner as tuner

    if not tuning_key:
        bench_fn()
        return

    fn = self.fn
    config_str_list = [str(c) for c in configs]
    assert len(config_str_list) == len(set(config_str_list)), (
        "Config strings must be unique"
    )
    cache_key = [tuner.VERSION, str(tuning_key)] + config_str_list
    cache_key = hashlib.sha256("-".join(cache_key).encode("utf-8")).hexdigest()
    cache = tuner.FileCacheManager(tuner._base32(cache_key))
    file_name = f"{fn.__name__[:150]}.autotune.json"
    path = cache.get_file(file_name)
    # There's an environment variable to force cache update
    if path and not os.environ.get(f"{tuner.PACKAGE_NAME.upper()}_FORCE_CACHE_UPDATE"):
        str2config = {s: c for s, c in zip(config_str_list, configs)}
        with open(path, "r") as cached_configs:
            timings = json.load(cached_configs)["configs_timings"]
        timings = {str2config[config]: timing for config, timing in timings}
        if not _failed(timings.values()):
            self.cache[tuning_key] = builtins.min(timings, key=timings.get)
            self.configs_timings = timings
            self.bench_time = 0
            return

    bench_fn()
    if _failed(self.configs_timings.values()):
        print(
            f"{tuner.PACKAGE_NAME} autotune: every config failed for "
            f"{fn.__name__}; the sweep is not cached",
            file=sys.stderr,
        )
        return
    cache.put(
        json.dumps(
            {
                # str(): the key tuple holds torch dtypes/shape tuples, and
                # this field is informational (only configs_timings is read back)
                "key": str(tuning_key),
                "configs_timings": [
                    (str(config), timings)
                    for config, timings in self.configs_timings.items()
                ],
            }
        ),
        file_name,
        binary=False,
    )


def apply():
    """Install the sweep changes; repeated calls are no-ops."""
    global _applied, _copy_sets
    if _applied:
        return False
    import quack.autotuner as tuner
    from quack.bench import bench_utils

    helpers = (
        "_pick_l2_rotate_count",
        "_clone_l2_rotate_inputs",
        "_bench_cuda_graph_l2_rotate",
    )
    for name in helpers:
        _checked_source(getattr(bench_utils, name), _SOURCES[name])
        if getattr(tuner, name) is not getattr(bench_utils, name):
            raise RuntimeError(
                f"Unsupported Quack autotuner: it no longer uses bench_utils.{name}"
            )
    for name in ("check_disk_cache", "_bench"):
        _checked_source(getattr(tuner.Autotuner, name), _SOURCES[name])

    _copy_sets = bench_utils._clone_l2_rotate_inputs
    # The autotuner calls the helpers through its own module's names.
    tuner._pick_l2_rotate_count = rotation_count
    tuner._clone_l2_rotate_inputs = rotation_sets
    tuner._bench_cuda_graph_l2_rotate = restoring(
        bench_utils._bench_cuda_graph_l2_rotate
    )
    tuner.Autotuner.check_disk_cache = torch.compiler.disable(check_disk_cache)
    _applied = True
    return True


__all__ = [
    "L2_ROUNDS",
    "MAX_SETS",
    "RotationSets",
    "SavedTensors",
    "apply",
    "check_disk_cache",
    "restoring",
    "rotation_count",
    "rotation_sets",
]
