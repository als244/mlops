"""Provider facts consumed by compiler adapters, independent of any engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from torch.fx import Node


@dataclass(frozen=True)
class AxisFrom:
    """Inherit an input's logical axis after removing trailing feature dims."""

    input_index: int
    trailing_dims: int = 0

    def __post_init__(self):
        if any(
            type(x) is not int or x < 0 for x in (self.input_index, self.trailing_dims)
        ):
            raise ValueError(
                "input indices and trailing dimensions must be nonnegative integers"
            )


@dataclass(frozen=True)
class ChunkAxis:
    """Physical dimension ordered as ``before * logical_extent * after``."""

    dim: int
    before: int = 1
    after: int = 1

    def __post_init__(self):
        if any(type(x) is not int for x in (self.dim, self.before, self.after)):
            raise TypeError("axis dimensions and factors must be integers")
        if self.before < 1 or self.after < 1:
            raise ValueError("axis factors must be positive")


@dataclass(frozen=True)
class ChunkLocal:
    """Invocation-local coordinates; cannot escape as an ordinary full output."""

    dim: int | None = None

    def __post_init__(self):
        if self.dim is not None and type(self.dim) is not int:
            raise TypeError("a local dimension must be an integer or None")


@dataclass(frozen=True)
class ChunkContract:
    """Output-axis and additive-reduction semantics for one dispatcher target.

    ``None`` denotes a shared output, or an additive result when its position is
    listed in ``reductions``. These facts do not choose axes or chunk sizes for
    the caller's graph and do not implement an execution loop.
    """

    target: str
    version: str
    output_axes: tuple[AxisFrom | ChunkAxis | ChunkLocal | None, ...]
    reductions: tuple[int, ...] = ()


class GradientDestination(Protocol):
    """FX output and destination presented to a provider's fusion resolver.

    ``operations`` is ordered from final gradient back toward the producer;
    the producer tuple's getitem is excluded. Tensor metadata is in ``val``.
    The resolver inspects the graph but must not modify it.
    """

    output_index: int
    destination: Node
    operations: tuple[Node, ...]


class GradientContext(Protocol):
    producer: Node
    destinations: tuple[GradientDestination, ...]
    fuse_narrow_gradient_accumulation: bool


@dataclass(frozen=True)
class GradientMapping:
    """Selected outputs and literal keywords for the accumulating overload."""

    output_indices: tuple[int, ...]
    keywords: Mapping[str, Any]


@dataclass(frozen=True)
class GradientContract:
    """Replace one backward with its destination-writing overload once.

    The caller passes original arguments plus ``destinations``,
    ``output_indices``, ``contract_version`` and the resolved keywords. Selected
    outputs use their destinations; other outputs retain their original values.
    Returning None from the resolver declines the replacement.
    """

    target: str
    replacement: str
    output_indices: tuple[int, ...]
    version: int
    resolve: Callable[[GradientContext], GradientMapping | None]


@dataclass(frozen=True)
class ValueContract:
    """Read-only operator with layout-independent values at listed input ports."""

    target: str
    version: str
    inputs: tuple[int, ...]


@dataclass(frozen=True)
class KernelPreparation:
    """Importable build/load factory and immutable scalar options.

    Factories accept phase, device and options. Builders capture/export a task's
    actual calls; loaders bind validated manifests and blobs. Both own close().
    See docs/PREPARED_KERNELS.md for lifecycle and stream ownership requirements.
    """

    factory: str
    options: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
