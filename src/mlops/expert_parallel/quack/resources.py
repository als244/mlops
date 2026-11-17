"""Portable initialization of named runtimes sharing one MoonEP token buffer.

This module has no dependency on a particular training engine. Its factory accepts
plain options and already resolved process groups; modules only save stable names.
"""

import json
from contextlib import ExitStack

import torch

from ..buffers import create_buffer
from .buffers import bind_buffer
from .config import _configuration
from .registry import _REGISTRY_LOCK, _RUNTIMES, _register_runtime


class Resources:
    def __init__(self):
        self._stack = ExitStack()

    def close(self):
        self._stack.close()


def _close_runtime(handle):
    with _REGISTRY_LOCK:
        runtime = _RUNTIMES.pop(handle)
    runtime.close()


def _settings(spec):
    """Decode a full-training or LoRA module's portable operator signature."""
    if "lora" in json.loads(spec):
        from ..lora import parse_signature

        base, lora = parse_signature(spec)
        return _configuration(base), lora
    return _configuration(spec), None


def initialize(*, name, device, groups, options):
    """options['layers'] contains {name, config} records; group is 'exchange'.

    `config` is a layer's resource_spec, including fixed GEMM and LoRA settings. Parameters
    are never allocated, initialized or copied here. Shared scratch/banks belong
    to this resource, and are destroyed before the caller closes process groups.
    """
    from . import _initialize

    _initialize()
    from .runtime import _Runtime

    descriptions = options["layers"]
    if not descriptions:
        raise ValueError("provide at least one named layer runtime")
    if len({item["name"] for item in descriptions}) != len(descriptions):
        raise ValueError("runtime resource names must be unique")
    settings = [_settings(item["config"]) for item in descriptions]
    configs = [config for config, _ in settings]
    if any(c.compute_precision == "fp8_block" and lora is None for c, lora in settings):
        raise ValueError("Block FP8 named resources require a LoRA configuration")
    if any(c.gemm_tuned for c in configs):
        raise ValueError("portable runtimes require fixed GEMM policies")
    if any(c.tokens_per_rank is None for c in configs):
        raise ValueError("portable runtimes require tokens_per_rank")
    group = groups["exchange"]
    resources = Resources()
    try:
        with torch.cuda.device(device), torch.device("cpu"):
            buffer = create_buffer(configs[0], configs[0].tokens_per_rank, group)
            resources._stack.callback(buffer.destroy)
            for description, (config, lora) in zip(descriptions, settings, strict=True):
                config, bound = bind_buffer(config, buffer, group, device)
                if lora is None:
                    runtime = _Runtime(config, bound, device, buffer)
                else:
                    from .lora.runtime import LoRARuntime

                    runtime = LoRARuntime(config, lora, bound, device, buffer)
                try:
                    handle = _register_runtime(runtime, name=description["name"])
                except BaseException:
                    runtime.close()
                    raise
                resources._stack.callback(_close_runtime, handle)
        return resources
    except BaseException:
        resources.close()
        raise
