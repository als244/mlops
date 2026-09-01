"""Support filtering, deterministic selection, and dispatch diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from .context import implementation_override, record_dispatch
from .registry import (
    Implementation,
    SupportResult,
    frozen_implementation,
    implementations_for,
)


@dataclass(frozen=True)
class ImplementationExplanation:
    """Human-readable resolution result for one concrete invocation."""

    operation: str
    surface: str
    selected: str | None
    forced: str | None
    candidates: Mapping[str, SupportResult]


def _support_result(
    implementation: Implementation,
    *args,
    surface: str,
    **kwargs,
) -> SupportResult:
    if surface == "explicit" and implementation.forward is None:
        return SupportResult.no("implementation does not expose explicit forward/backward")
    if surface not in {"semantic", "explicit"}:
        return SupportResult.no(f"unknown surface {surface!r}")
    try:
        result = implementation.supports(*args, surface=surface, **kwargs)
    except Exception as error:
        return SupportResult.no(f"support check failed: {type(error).__name__}: {error}")
    if isinstance(result, SupportResult):
        return result
    if isinstance(result, bool):
        return SupportResult(result, "supported" if result else "rejected")
    raise TypeError(
        f"{implementation.implementation_id}.supports returned "
        f"{type(result).__name__}, expected SupportResult"
    )


def explain_implementation(
    operation: str,
    *args,
    surface: str = "semantic",
    **kwargs,
) -> ImplementationExplanation:
    """Explain support and selection for one exact operation invocation."""
    operation = str(operation)
    implementations = implementations_for(operation)
    results = {
        implementation_id: _support_result(
            implementation, *args, surface=surface, **kwargs
        )
        for implementation_id, implementation in implementations.items()
    }
    forced = implementation_override(operation)
    if forced is not None:
        selected = forced if results[forced].supported else None
    else:
        supported = [
            implementation
            for implementation_id, implementation in implementations.items()
            if results[implementation_id].supported
        ]
        supported.sort(key=lambda item: (-item.priority, item.implementation_id))
        selected = supported[0].implementation_id if supported else None
    return ImplementationExplanation(
        operation=operation,
        surface=str(surface),
        selected=selected,
        forced=forced,
        candidates=MappingProxyType(results),
    )


def resolve_implementation(
    operation: str,
    *args,
    surface: str = "semantic",
    **kwargs,
) -> Implementation:
    """Resolve one implementation or fail with all concrete rejection reasons."""
    if torch.compiler.is_compiling():
        forced = implementation_override(str(operation))
        if forced is None:
            raise RuntimeError(
                f"operation {operation!r} was not frozen before graph capture"
            )
        return frozen_implementation(str(operation), forced)
    explanation = explain_implementation(
        operation, *args, surface=surface, **kwargs
    )
    if explanation.selected is None:
        reasons = "; ".join(
            f"{name}: {result.reason}"
            for name, result in sorted(explanation.candidates.items())
        )
        if explanation.forced is not None:
            prefix = f"forced implementation {explanation.forced!r} is unsupported"
        else:
            prefix = "no supported implementation"
        raise RuntimeError(
            f"{prefix} for {explanation.operation!r} on {surface!r} surface; {reasons}"
        )
    implementation = implementations_for(explanation.operation)[explanation.selected]
    record_dispatch(explanation.operation, implementation.implementation_id)
    return implementation


def implementation_pairs(operations) -> Mapping[str, Mapping[str, object]]:
    """Build default/native-Torch verifier pairs from exact registered IDs."""
    pairs: dict[str, Mapping[str, object]] = {}
    for operation_value in operations:
        operation = str(operation_value)
        implementations = implementations_for(operation)
        ordered = sorted(
            implementations.values(),
            key=lambda item: (-item.priority, item.implementation_id),
        )
        native = [
            item for item in ordered if item.provider == "native_torch"
        ]
        if not native:
            raise ValueError(f"operation {operation!r} has no native_torch implementation")
        pairs[operation] = MappingProxyType(
            {
                "default": MappingProxyType(
                    {operation: ordered[0].implementation_id}
                ),
                "native_torch": MappingProxyType(
                    {operation: native[0].implementation_id}
                ),
            }
        )
    return MappingProxyType(pairs)


__all__ = [
    "ImplementationExplanation",
    "explain_implementation",
    "implementation_pairs",
    "resolve_implementation",
]
