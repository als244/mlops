"""Immutable implementation records and the process-local operation catalog."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

@dataclass(frozen=True)
class SupportResult:
    """Whether an implementation accepts one concrete invocation."""

    supported: bool
    reason: str = ""

    @classmethod
    def yes(cls) -> "SupportResult":
        return cls(True, "supported")

    @classmethod
    def no(cls, reason: str) -> "SupportResult":
        return cls(False, str(reason))

    def __bool__(self) -> bool:
        return self.supported


@dataclass(frozen=True)
class Implementation:
    """One stateless implementation of one exact semantic operation."""

    operation: str
    implementation_id: str
    provider: str
    priority: int
    deterministic: bool
    supports: Callable[..., SupportResult]
    apply: Callable
    forward: Callable | None = None
    backward: Callable | None = None
    estimate: Callable | None = None

    def __post_init__(self) -> None:
        if not self.operation or not self.implementation_id or not self.provider:
            raise ValueError("operation, implementation_id, and provider must be non-empty")
        if (self.forward is None) != (self.backward is None):
            raise ValueError(
                f"{self.implementation_id}: forward and backward must both be present "
                "or both be absent"
            )


_REGISTRY: dict[str, dict[str, Implementation]] = {}


def register_implementation(implementation: Implementation) -> Implementation:
    """Register and return ``implementation``; duplicate IDs are an error."""
    operation = str(implementation.operation)
    implementations = _REGISTRY.setdefault(operation, {})
    implementation_id = str(implementation.implementation_id)
    if implementation_id in implementations:
        raise ValueError(
            f"duplicate implementation ID {implementation_id!r} for {operation!r}"
        )
    implementations[implementation_id] = implementation
    return implementation


def implementation_registry() -> Mapping[str, Mapping[str, Implementation]]:
    """Return a read-only snapshot of all registered implementations."""
    from ..providers import ensure_implementations_registered

    ensure_implementations_registered()
    return MappingProxyType(
        {
            operation: MappingProxyType(dict(implementations))
            for operation, implementations in sorted(_REGISTRY.items())
        }
    )


def implementations_for(operation: str) -> Mapping[str, Implementation]:
    """Return implementations for one exact operation."""
    registry = implementation_registry()
    try:
        return registry[str(operation)]
    except KeyError as error:
        raise ValueError(f"unknown semantic operation {operation!r}") from error


def frozen_implementation(operation: str, implementation_id: str) -> Implementation:
    """Return one prevalidated record as a compiler-time Python constant."""
    from ..providers import ensure_implementations_registered

    ensure_implementations_registered()
    return _REGISTRY[str(operation)][str(implementation_id)]


__all__ = [
    "Implementation",
    "SupportResult",
    "implementation_registry",
    "implementations_for",
    "frozen_implementation",
    "register_implementation",
]
