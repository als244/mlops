"""CPU-only descriptions of optional external-kernel preparation factories."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import torch


@dataclass(frozen=True)
class KernelPreparation:
    """Importable build/load factory and immutable scalar options.

    Factories accept phase, device and options. Builders capture/export a task's
    actual calls; loaders bind validated manifests and blobs. Both own close().
    See docs/PREPARED_KERNELS.md for lifecycle and stream ownership requirements.
    """

    factory: str
    options: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


def sonicmoe_preparation(*, cache_root: str | None = None) -> KernelPreparation:
    """Selected BF16 Sonic kernels; FP8 and expert parallelism are excluded."""
    return KernelPreparation(
        factory="mlops.preparation.sonic:factory",
        options={} if cache_root is None else {"cache_root": cache_root},
    )


def operator_preparations(
    *, dtype: torch.dtype = torch.bfloat16, cache_root: str | None = None
) -> tuple[KernelPreparation, ...]:
    """Describe BF16 Sonic and Hopper sparse-forward selection preparation.

    Availability and device capability are checked only when the caller opens
    the factory. Other providers retain ordinary dispatch. This is not binary
    preparation support for every operation or for sparse-attention backward.
    """
    if dtype != torch.bfloat16:
        return ()
    return (
        KernelPreparation(
            factory="mlops.preparation.available:factory",
            options={} if cache_root is None else {"cache_root": cache_root},
        ),
        KernelPreparation(
            factory="mlops.providers.builtin.sparse_latent_preparation:factory"
        ),
    )
