"""MoonEP transport, Quack GEMMs and SonicMoE routing."""

from importlib import import_module

from ..lora import LoRAConfig
from .config import MoEConfig, config_signature

QuackMoEConfig = MoEConfig
_initialized = False


def _initialize():
    global _initialized
    if _initialized:
        return
    from .._compat import quack_autotune, quack_pipeline

    quack_pipeline.apply()
    quack_autotune.apply()
    from .._compat import initialize_moonep

    initialize_moonep()
    from .parameters import components  # noqa: F401

    _initialized = True


def __getattr__(name):
    if name in ("MoELayer", "QuackMoE"):
        _initialize()
        return import_module(f"{__name__}.layer").QuackMoE
    if name == "QuackMoELoRA":
        _initialize()
        return import_module(f"{__name__}.lora").QuackMoELoRA
    if name == "ChunkBufferPool":
        return import_module(f"{__name__}.pipeline.buffers").ChunkBufferPool
    raise AttributeError(name)


__all__ = [
    "ChunkBufferPool",
    "LoRAConfig",
    "MoEConfig",
    "MoELayer",
    "QuackMoE",
    "QuackMoEConfig",
    "QuackMoELoRA",
    "config_signature",
]
