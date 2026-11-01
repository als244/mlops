"""Register the package's custom operators before artifact loading.

The implementation catalog registers the operators of every implementation
adapter; the modules below define the LoRA, router and sequential-MoE
operators outside it. Expert-parallel operators register when the modules
defining them are imported, and accessing a layer class imports its own.
"""

from importlib import import_module

from .providers import ensure_implementations_registered

_OPERATOR_MODULES = (
    "mlops.kernels.router_op",
    "mlops.lora.grouped",
    "mlops.lora.routing",
    "mlops.providers.sonicmoe.lora",
    "mlops.sequential_moe.experts.autograd",
    "mlops.sequential_moe.experts.operators",
    "mlops.sequential_moe.experts.reuse",
    "mlops.sequential_moe.precision.linear",
    "mlops.sequential_moe.router",
    "mlops.sequential_moe.routing",
    "mlops.sequential_moe.shared",
)


ensure_implementations_registered()
for _module_name in _OPERATOR_MODULES:
    import_module(_module_name)


__all__ = []
