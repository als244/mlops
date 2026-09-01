"""Development helpers for checking an implementation's registered VJP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch

from .context import use_implementation
from .resolution import explain_implementation
from .registry import implementations_for


@dataclass(frozen=True)
class GradcheckResult:
    """Outcome of one exact implementation gradcheck."""

    operation: str
    implementation_id: str
    status: str
    reason: str


@dataclass(frozen=True)
class GradcheckCase:
    """One operation invocation to check across selected implementations."""

    operation: str
    inputs: tuple[torch.Tensor, ...]
    kwargs: Mapping[str, object] | None = None
    implementation_ids: tuple[str, ...] | None = None


def gradcheck_implementation(
    operation: str,
    implementation_id: str,
    inputs: Sequence[torch.Tensor],
    *,
    kwargs: Mapping[str, object] | None = None,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = False,
) -> GradcheckResult:
    """Run ``torch.autograd.gradcheck`` on one exact semantic implementation.

    Callers must supply the operation's normal differentiable float64 inputs.
    Implementations that reject those inputs are reported as ``skipped``; the
    helper never substitutes another implementation.
    """
    operation = str(operation)
    implementation_id = str(implementation_id)
    call_kwargs = dict(kwargs or {})
    implementations = implementations_for(operation)
    if implementation_id not in implementations:
        raise ValueError(
            f"unknown implementation {implementation_id!r} for {operation!r}"
        )
    with use_implementation(operation, implementation_id):
        explanation = explain_implementation(
            operation, *inputs, surface="semantic", **call_kwargs
        )
    if explanation.selected is None:
        reason = explanation.candidates[implementation_id].reason
        return GradcheckResult(operation, implementation_id, "skipped", reason)

    implementation = implementations[implementation_id]

    def checked(*values):
        return implementation.apply(*values, **call_kwargs)

    try:
        passed = torch.autograd.gradcheck(
            checked,
            tuple(inputs),
            eps=float(eps),
            atol=float(atol),
            rtol=float(rtol),
            raise_exception=bool(raise_exception),
        )
    except Exception as error:
        if raise_exception:
            raise
        return GradcheckResult(
            operation,
            implementation_id,
            "failed",
            f"{type(error).__name__}: {error}",
        )
    return GradcheckResult(
        operation,
        implementation_id,
        "passed" if passed else "failed",
        "torch.autograd.gradcheck passed" if passed else "gradcheck returned False",
    )


def gradcheck_implementations(
    cases: Iterable[GradcheckCase],
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = False,
) -> tuple[GradcheckResult, ...]:
    """Gradcheck every requested exact implementation for each case.

    When a case omits ``implementation_ids``, every registered implementation
    of that exact operation is attempted in deterministic ID order. Unsupported
    float64/device/layout combinations are returned as explicit ``skipped``
    results, so a development report never substitutes an easier backend.
    """
    results = []
    for case in cases:
        registered = implementations_for(case.operation)
        implementation_ids = (
            tuple(sorted(registered))
            if case.implementation_ids is None
            else tuple(case.implementation_ids)
        )
        for implementation_id in implementation_ids:
            results.append(
                gradcheck_implementation(
                    case.operation,
                    implementation_id,
                    case.inputs,
                    kwargs=case.kwargs,
                    eps=eps,
                    atol=atol,
                    rtol=rtol,
                    raise_exception=raise_exception,
                )
            )
    return tuple(results)


__all__ = [
    "GradcheckCase",
    "GradcheckResult",
    "gradcheck_implementation",
    "gradcheck_implementations",
]
