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
    from . import patch_quack_autotune, patch_quack_runtime

    patch_quack_runtime.apply()
    patch_quack_autotune.apply()
    from .. import _patch_moonep_rank1  # noqa: F401
    from .parameters import components  # noqa: F401

    _initialized = True


def __getattr__(name):
    if name in ("MoELayer", "QuackMoE"):
        _initialize()
        return import_module(f"{__name__}.layer").MoELayer
    if name == "QuackMoELoRA":
        _initialize()
        return import_module(f"{__name__}.lora").QuackMoELoRA
    if name == "ChunkBufferPool":
        return import_module(f"{__name__}.chunk_buffers").ChunkBufferPool
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
