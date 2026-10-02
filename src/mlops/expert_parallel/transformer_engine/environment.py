"""Installed dependency and CUDA diagnostics."""

from __future__ import annotations

import importlib.metadata
import os

import torch

from .operators import _EFFECT_ERROR


def environment_report():
    out = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "ordered_effect_api": _EFFECT_ERROR is None,
        "NVTE_EXT_MARGIN_SM": os.environ.get("NVTE_EXT_MARGIN_SM"),
    }
    for p in ("moonep", "transformer_engine", "triton"):
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    return out
