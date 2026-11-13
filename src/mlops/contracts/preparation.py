"""CPU-only descriptions of optional external-kernel preparation factories."""

import torch

from .types import KernelPreparation


def sonicmoe_preparation(*, cache_root: str | None = None) -> KernelPreparation:
    """Selected BF16 Sonic kernels; FP8 and expert parallelism are excluded."""
    return KernelPreparation(
        "mlops.preparation.sonic:factory",
        {} if cache_root is None else {"cache_root": cache_root},
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
            "mlops.preparation.available:factory",
            {} if cache_root is None else {"cache_root": cache_root},
        ),
        KernelPreparation("mlops.providers.builtin.sparse_latent_preparation:factory"),
    )
