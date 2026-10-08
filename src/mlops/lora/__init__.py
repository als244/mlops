"""Explicit low-rank modules and model-independent conversion."""

from .config import LoRAConfig
from .linear import LoRAHead, LoRALinear
from .selection import apply_lora, parameter_report

__all__ = ["LoRAConfig", "LoRAHead", "LoRALinear", "apply_lora", "parameter_report"]
