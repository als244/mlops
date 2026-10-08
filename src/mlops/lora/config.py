"""Precision and scaling of explicit low-rank factors."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 32
    alpha: float = 32.0
    factor_dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if self.factor_dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise ValueError("factor_dtype must be a floating dtype")

    @property
    def scale(self):
        return self.alpha / self.rank
