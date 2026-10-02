"""MoonEP transport and Transformer Engine grouped expert math."""

from importlib import import_module

from ..lora import LoRAConfig
from .config import MoEConfig, config_signature

TEMoEConfig = MoEConfig
_initialized = False


def _initialize():
    global _initialized
    if _initialized:
        return
    from .._compat import moonep_rank1  # noqa: F401
    from .parameters import components  # noqa: F401

    _initialized = True


def __getattr__(name):
    if name in ("MoELayer", "TEMoE"):
        _initialize()
        return import_module(f"{__name__}.layer").TEMoE
    if name == "TEMoELoRA":
        _initialize()
        return import_module(f"{__name__}.lora").TEMoELoRA
    raise AttributeError(name)


__all__ = [
    "LoRAConfig",
    "MoEConfig",
    "MoELayer",
    "TEMoE",
    "TEMoEConfig",
    "TEMoELoRA",
    "config_signature",
]
