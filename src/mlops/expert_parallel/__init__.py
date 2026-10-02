"""Optional expert-parallel layers with caller-owned groups and buffers.

Importing this namespace or its configurations does not load accelerator
libraries. Access a layer class after selecting the process's compute device.
"""

from importlib import import_module

from .buffers import buffer_context, create_buffer
from .lora import LoRAConfig

_EXPORTS = {
    "QuackMoE": ("quack", "QuackMoE"),
    "QuackMoELoRA": ("quack", "QuackMoELoRA"),
    "QuackMoEConfig": ("quack", "MoEConfig"),
    "TEMoE": ("transformer_engine", "TEMoE"),
    "TEMoELoRA": ("transformer_engine", "TEMoELoRA"),
    "TEMoEConfig": ("transformer_engine", "MoEConfig"),
    "ChunkBufferPool": ("quack", "ChunkBufferPool"),
}
__all__ = [
    "ChunkBufferPool",
    "LoRAConfig",
    "QuackMoE",
    "QuackMoEConfig",
    "QuackMoELoRA",
    "TEMoE",
    "TEMoEConfig",
    "TEMoELoRA",
    "buffer_context",
    "create_buffer",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), symbol)
    globals()[name] = value
    return value
