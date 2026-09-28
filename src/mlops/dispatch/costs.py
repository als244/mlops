"""Optional scalar-only operation and implementation cost hints."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

import torch
from torch.utils.flop_counter import flop_registry
from torch.utils.flop_counter import register_flop_formula as _register_flop_formula

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
    from . import logical_costs  # noqa: F401  # registers every operation's estimator

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


def _overload_packet(operator) -> torch._ops.OpOverloadPacket:
    if isinstance(operator, torch._ops.OpOverloadPacket):
        return operator
    if isinstance(operator, torch._ops.OpOverload):
        return operator.overloadpacket
    overload = getattr(operator, "_opoverload", None)
    if isinstance(overload, torch._ops.OpOverload):
        return overload.overloadpacket
    raise TypeError(
        "flop_formula takes a registered custom operator or its torch.ops packet, "
        f"not {type(operator).__name__}"
    )


def flop_formula(*operators):
    """Register the decorated function as the FLOP count of ``operators``.

    ``operators`` are registered custom operators -- the objects
    ``torch.library.custom_op`` returns -- or their ``torch.ops`` packets. The
    decorated function takes the operator's own arguments, positionally or by
    name, plus ``out_val``, the operator's result, and returns an ``int``.
    ``torch.utils.flop_counter.FlopCounterMode`` calls it whenever the operator
    runs under it, with fake tensors as readily as real ones, so a formula
    reads shapes, dtypes and static arguments and nothing else.

    The count is logical work, not what one kernel happens to do: two per
    multiply-add of a matrix product, a small constant per element of an
    elementwise or normalizing pass, zero for a gather. Work whose extent
    depends on values a shape cannot show -- the sequence lengths behind a
    packed attention, the routing behind a mixture of experts -- is bounded
    from the shapes instead. Every operator this package registers has one,
    forward and backward alike, so a consumer pricing a graph by its
    arithmetic sees each mlops operator as it is rather than as unknown.
    """
    packets = tuple(_overload_packet(operator) for operator in operators)
    if not packets:
        raise ValueError("flop_formula requires at least one operator")

    def register(formula):
        for packet in packets:
            _register_flop_formula(packet, get_raw=True)(formula)
        return formula

    return register


def has_flop_formula(operator) -> bool:
    """Whether ``operator`` has a registered FLOP formula."""
    return _overload_packet(operator) in flop_registry


__all__ = [
    "CostHints",
    "estimate_implementation",
    "flop_formula",
    "has_flop_formula",
    "operation_estimators",
    "register_operation_estimator",
]
