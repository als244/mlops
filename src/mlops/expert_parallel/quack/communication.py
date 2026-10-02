"""MoonEP saved-plan fields and tensor traversal."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

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
