"""Preserve reduction-buffer backing in Quack's warm-call recipes.

The pinned frontend allocates a sweep-sized buffer on a cold tuned call, then
caches only the winning slice's shape. Replaying that recipe changes both the
allocation size and the slice's strides. Record storage extent and view layout
together so cold and warm calls have the same memory contract. Kernel selection,
the warm launch fast path, and caller-provided buffers remain unchanged.
"""

import importlib
import linecache

import torch

from .pipeline import _checked_source

_applied = False


def _recipe(tensor):
    if tensor.dtype != torch.float32:
        raise ValueError("Quack reduction storage must be float32")
    return (
        tensor.untyped_storage().nbytes() // tensor.element_size(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
    )


def _allocate(recipe, device):
    elements, shape, stride, offset = recipe
    storage = torch.empty(elements, dtype=torch.float32, device=device)
    return storage.as_strided(shape, stride, offset)


def apply():
    """Install the source-checked compatibility fix before Quack is used."""
    global _applied
    if _applied:
        return False
    frontend = importlib.import_module("quack.epilogue.frontend")
    original = frontend.EpiMod.__call__
    source = _checked_source(
        original, "2b042a5525e686d8736a45842e132b4c0cc5d4b29cfd881288317fbb2624b6da"
    )
    replacements = {
        "for name, shape in sink_shapes:": "for name, recipe in sink_shapes:",
        "buf = _t.empty(shape, dtype=_t.float32, device=A.device)": "buf = _mlops_allocate_sink(recipe, A.device)",
        "sink_shapes = tuple((name, tuple(b.shape)) for name, b in sink_bufs.items())": "sink_shapes = tuple((name, _mlops_sink_recipe(b)) "
        "for name, b in sink_bufs.items())",
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError("Unsupported Quack reduction storage recipe")
        source = source.replace(before, after)
    filename = "<mlops.quack_runtime.sink_storage.EpiMod.__call__>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(
        vars(frontend), _mlops_sink_recipe=_recipe, _mlops_allocate_sink=_allocate
    )
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 -- source checked above
    replacement = namespace["__call__"]
    replacement.__qualname__ = original.__qualname__
    frontend.EpiMod.__call__ = replacement
    _applied = True
    return True
