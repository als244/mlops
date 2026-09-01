"""Packed projection-graph linear mixer implementation."""

from __future__ import annotations

from ...dispatch.registry import Implementation, SupportResult, register_implementation


def _supports(forward_segment, hidden, lengths, *, surface, **_kwargs):
    del surface, lengths
    if not callable(forward_segment):
        return SupportResult.no("forward_segment must be callable")
    if hidden.ndim < 2:
        return SupportResult.no("hidden must include batch and token dimensions")
    return SupportResult.yes()


def apply(forward_segment, hidden, lengths):
    return forward_segment(hidden, tuple(int(length) for length in lengths))


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="linear_mixer",
        implementation_id="builtin.linear_mixer.packed",
        provider="builtin",
        priority=100,
        deterministic=True,
        supports=_supports,
        apply=apply,
    )
)

__all__ = ["IMPLEMENTATION", "apply"]
