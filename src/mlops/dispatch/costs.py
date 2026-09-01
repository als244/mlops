"""Optional scalar-only operation and implementation cost hints."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from .context import use_implementation
from .registry import implementations_for
from .resolution import explain_implementation


@dataclass(frozen=True)
class CostHints:
    """Advisory roofline and workspace hints for one concrete invocation.

    ``None`` always means undefined. Byte counts describe traffic, not allocator
    reservations. Workspace is provider scratch allocated for this one forward
    or backward invocation and dead when that call returns. It excludes inputs,
    outputs, saved autograd residuals, reusable tables/state, allocator cache,
    and runtime leeway.
    """

    logical_flops: int | None = None
    logical_bytes_accessed: int | None = None
    implementation_flops: int | None = None
    implementation_bytes_accessed: int | None = None
    workspace_bytes: int | None = None
    notes: tuple[str, ...] = ()

    @property
    def has_roofline_estimate(self) -> bool:
        """Whether this result contains a complete physical roofline pair."""
        return (
            self.implementation_flops is not None
            and self.implementation_bytes_accessed is not None
        )

    def merged(self, other: "CostHints") -> "CostHints":
        """Return ``other`` overlaid on this result without inventing values."""
        values = {}
        for name in (
            "logical_flops",
            "logical_bytes_accessed",
            "implementation_flops",
            "implementation_bytes_accessed",
            "workspace_bytes",
        ):
            replacement = getattr(other, name)
            values[name] = getattr(self, name) if replacement is None else replacement
        values["notes"] = self.notes + other.notes
        return CostHints(**values)


_OPERATION_ESTIMATORS: dict[str, Callable[..., CostHints]] = {}


def register_operation_estimator(operation: str, estimator: Callable[..., CostHints]):
    """Register one canonical mathematical estimator."""
    operation = str(operation)
    if operation in _OPERATION_ESTIMATORS:
        raise ValueError(f"duplicate operation cost estimator for {operation!r}")
    _OPERATION_ESTIMATORS[operation] = estimator
    return estimator


def operation_estimators() -> Mapping[str, Callable[..., CostHints]]:
    """Return registered canonical estimators without evaluating them."""
    from ..providers import ensure_implementations_registered

    ensure_implementations_registered()
    return MappingProxyType(dict(_OPERATION_ESTIMATORS))


def estimate_implementation(
    operation: str,
    *args,
    implementation_id: str | None = None,
    surface: str = "semantic",
    entrypoint: str = "forward",
    **kwargs,
) -> CostHints:
    """Estimate an entrypoint from the operation's ordinary forward arguments.

    Estimators may inspect tensor metadata such as shape, stride, dtype, and
    device properties. They must not execute kernels or retain tensor objects.
    ``None`` in the returned record is an intentional unknown, never zero.
    """
    if entrypoint not in {"forward", "backward"}:
        raise ValueError("entrypoint must be 'forward' or 'backward'")
    operation = str(operation)
    if implementation_id is None:
        explanation = explain_implementation(
            operation, *args, surface=surface, **kwargs
        )
    else:
        with use_implementation(operation, implementation_id):
            explanation = explain_implementation(
                operation, *args, surface=surface, **kwargs
            )
    if explanation.selected is None:
        reasons = "; ".join(
            f"{name}: {result.reason}"
            for name, result in sorted(explanation.candidates.items())
        )
        raise RuntimeError(f"cannot estimate unsupported {operation!r}: {reasons}")
    implementation = implementations_for(operation)[explanation.selected]
    canonical_estimator = operation_estimators().get(operation)
    canonical = (
        CostHints(notes=("canonical operation estimate is undefined",))
        if canonical_estimator is None
        else canonical_estimator(*args, entrypoint=entrypoint, **kwargs)
    )
    implementation_hints = (
        CostHints(notes=(f"{implementation.implementation_id} provides no physical estimate",))
        if implementation.estimate is None
        else implementation.estimate(*args, entrypoint=entrypoint, **kwargs)
    )
    return canonical.merged(implementation_hints)


__all__ = [
    "CostHints",
    "estimate_implementation",
    "operation_estimators",
    "register_operation_estimator",
]
