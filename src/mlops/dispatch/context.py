"""Context-local implementation overrides and warmup-only dispatch tracing.

Each choice can be made for one block -- ``use_implementations``,
``deterministic_kernels``, ``weight_gradients_at`` -- or from a point on, for a
caller that makes it once: ``set_implementations``,
``set_deterministic_kernels``, ``set_weight_gradient_dtype``.
"""

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
_DETERMINISTIC: ContextVar[bool] = ContextVar(
    "operation_deterministic_kernels", default=False
)
_WEIGHT_GRADIENT_DTYPE: ContextVar[torch.dtype | None] = ContextVar(
    "operation_weight_gradient_dtype", default=None
)


@torch.compiler.assume_constant_result
def implementation_override(operation: str) -> str | None:
    """Return the exact context-local override, if one exists."""
    return _OVERRIDES.get().get(str(operation))


def _validated(overrides: Mapping[str, str]) -> dict[str, str]:
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
    return canonical


def _merged(canonical: Mapping[str, str]) -> Mapping[str, str]:
    merged = dict(_OVERRIDES.get())
    merged.update(canonical)
    return MappingProxyType(merged)


@contextmanager
def use_implementations(overrides: Mapping[str, str]):
    """Apply validated exact overrides, inheriting outer context selections."""
    canonical = _validated(overrides)
    token = _OVERRIDES.set(_merged(canonical))
    try:
        yield MappingProxyType(canonical)
    finally:
        _OVERRIDES.reset(token)


def set_implementations(overrides: Mapping[str, str]) -> None:
    """Apply validated exact overrides from here on, beside earlier selections.

    What ``use_implementations`` applies for one block, kept for the rest of the
    calling context: for a process that makes the choice once.
    """
    _OVERRIDES.set(_merged(_validated(overrides)))


@contextmanager
def use_implementation(operation: str, implementation_id: str):
    """Convenience context manager for one exact implementation override."""
    with use_implementations({operation: implementation_id}) as selected:
        yield selected[str(operation)]


@torch.compiler.assume_constant_result
def deterministic_required() -> bool:
    """Return whether the caller has asked for run-to-run reproducibility."""
    return _DETERMINISTIC.get()


@contextmanager
def deterministic_kernels(enabled: bool = True):
    """Ask every operation for kernels that repeat bit for bit.

    Some kernels reach their answer by an order that varies run to run --
    an atomic accumulation finishes in whatever order blocks retire -- so
    two runs of one step from one seed can end in different states.  The
    ordered variant costs throughput, which is why it is not the default;
    qualification turns it on to compare a run against a reference or
    against itself.  Operations that are already ordered ignore this.
    """
    token = _DETERMINISTIC.set(bool(enabled))
    try:
        yield bool(enabled)
    finally:
        _DETERMINISTIC.reset(token)


def set_deterministic_kernels(enabled: bool = True) -> None:
    """Ask for kernels that repeat bit for bit -- or stop asking -- from here on:
    what ``deterministic_kernels`` asks for one block, kept for the rest of the
    calling context."""
    _DETERMINISTIC.set(bool(enabled))


@torch.compiler.assume_constant_result
def weight_gradient_dtype() -> torch.dtype | None:
    """Return the dtype operations give their weights' gradients at, as an
    artifact constant; ``None`` is each weight's own dtype."""
    return _WEIGHT_GRADIENT_DTYPE.get()


def _floating(dtype: torch.dtype | None) -> torch.dtype | None:
    if dtype is not None and (
        not isinstance(dtype, torch.dtype) or not dtype.is_floating_point
    ):
        raise ValueError(f"weight gradients need a floating dtype or None; got {dtype!r}")
    return dtype


@contextmanager
def weight_gradients_at(dtype: torch.dtype | None):
    """Ask every operation for the gradients of its weights at ``dtype``.

    An operation that sums a weight's gradient over rows -- a norm's weight, an
    embedding table, a head over its chunks, an expert's weights -- keeps the
    sum at fp32 and rounds it to the weight's dtype as it returns it. Asked for
    another dtype, it returns the sum at that one instead: a caller that keeps
    gradients at fp32 gets them without that rounding. The dtype is read when
    an operation is called and travels with it, so a captured graph keeps the
    one it was captured under. ``None``, the default, is each weight's own
    dtype. Operations whose weight gradients a library computes and rounds
    itself -- FLA's, Liger's -- return them as that library does.
    """
    token = _WEIGHT_GRADIENT_DTYPE.set(_floating(dtype))
    try:
        yield dtype
    finally:
        _WEIGHT_GRADIENT_DTYPE.reset(token)


def set_weight_gradient_dtype(dtype: torch.dtype | None) -> None:
    """Ask for weight gradients at ``dtype`` from here on: what
    ``weight_gradients_at`` asks for one block, kept for the rest of the
    calling context."""
    _WEIGHT_GRADIENT_DTYPE.set(_floating(dtype))


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
    "set_deterministic_kernels",
    "set_implementations",
    "use_implementation",
    "use_implementations",
]
