"""Public construction helper for caller-owned rotary tables."""

from __future__ import annotations

import torch

from .kernels.rope import build_rope_tables as _build_rope_tables


def build_rope_tables(
    rows: int,
    rotary_dim: int,
    base: float,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reusable FP32 cosine/sine tables with shape ``[rows, rotary_dim]``."""
    return _build_rope_tables(
        torch.device(device), int(rows), int(rotary_dim), float(base)
    )


__all__ = ["build_rope_tables"]
