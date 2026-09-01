"""Context-local implementation overrides and warmup-only dispatch tracing."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Mapping

import torch


_OVERRIDES: ContextVar[Mapping[str, str]] = ContextVar(
    "operation_implementation_overrides", default=MappingProxyType({})
)
_TRACE: ContextVar[dict[str, dict[str, int]] | None] = ContextVar(
    "operation_implementation_trace", default=None
)


@torch.compiler.assume_constant_result
def implementation_override(operation: str) -> str | None:
    """Return the exact context-local override, if one exists."""
    return _OVERRIDES.get().get(str(operation))


@contextmanager
def use_implementations(overrides: Mapping[str, str]):
    """Apply validated exact overrides, inheriting outer context selections."""
    from .registry import implementations_for

    canonical: dict[str, str] = {}
    for operation, implementation_id in overrides.items():
        operation = str(operation)
        implementation_id = str(implementation_id)
        implementations = implementations_for(operation)
        if implementation_id not in implementations:
            raise ValueError(
                f"unknown implementation {implementation_id!r} for {operation!r}; "
                f"choose one of {sorted(implementations)}"
            )
        canonical[operation] = implementation_id
    merged = dict(_OVERRIDES.get())
    merged.update(canonical)
    token = _OVERRIDES.set(MappingProxyType(merged))
    try:
        yield MappingProxyType(canonical)
    finally:
        _OVERRIDES.reset(token)


@contextmanager
def use_implementation(operation: str, implementation_id: str):
    """Convenience context manager for one exact implementation override."""
    with use_implementations({operation: implementation_id}) as selected:
        yield selected[str(operation)]


@contextmanager
def capture_dispatch():
    """Capture exact implementations during one isolated uncaptured warmup."""
    trace: dict[str, dict[str, int]] = {}
    token = _TRACE.set(trace)
    try:
        yield trace
    finally:
        _TRACE.reset(token)


def record_dispatch(operation: str, implementation_id: str) -> None:
    """Record one resolution without allowing diagnostics into compiled graphs."""
    if torch.compiler.is_compiling():
        return
    trace = _TRACE.get()
    if trace is None:
        return
    counts = trace.setdefault(str(operation), {})
    implementation_id = str(implementation_id)
    counts[implementation_id] = counts.get(implementation_id, 0) + 1


def dispatch_manifest(trace: Mapping[str, Mapping[str, int]]) -> Mapping[str, str]:
    """Convert an unambiguous warmup trace to an exact frozen manifest."""
    manifest: dict[str, str] = {}
    for operation, counts in trace.items():
        exercised = [name for name, count in counts.items() if count]
        if len(exercised) != 1:
            raise ValueError(
                f"operation {operation!r} exercised {sorted(exercised)}; "
                "a frozen manifest requires exactly one implementation"
            )
        manifest[str(operation)] = exercised[0]
    return MappingProxyType(manifest)


__all__ = [
    "capture_dispatch",
    "dispatch_manifest",
    "implementation_override",
    "record_dispatch",
    "use_implementation",
    "use_implementations",
]
