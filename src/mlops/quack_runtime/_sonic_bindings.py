"""Private registration of the pinned BF16 Sonic/Quack GEMM dispatch surface.

This covers Quack GEMM objects, including lazily created epilogues. It does not
prepare Triton kernels, tuning choices, or the CuTe occupancy helper. A complete
provider adapter must handle those before claiming compilation-free execution.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

from . import initialize
from ._cache_scope import QuackCalls

_SOURCES = {
    "quack.gemm": "2aa12bc6a85a1a8dcdf2b1a6d19e70b6a3ce56d598bda0196d206b9fb1341fcf",
    "quack.gemm_interface": (
        "6215031bf8a870d16a1be224ecad21042076d62096f4621f477d63371eab92d3"
    ),
    "quack.gemm_runtime.host": (
        "96da5c3fee70d909e4b19a1c8d13b6e63346e05ff930ed36ce4d93d6bb9ac987"
    ),
    "quack.epilogue.frontend": (
        "1fc3c4a0914a23c04e69dfef03232f7efb85e03bbb8a7c6e515509d9582d6f4a"
    ),
}


def sonic_quack_calls() -> QuackCalls:
    """Register both GEMM compilers and every pinned outer launch-plan cache.

    Construction is CPU-only. No activation happens until capture/bind; eager
    caches remain separate. Data descriptors cover existing epilogues and ones
    created during a task, without scanning the process or patching constructors.
    """
    initialize()
    modules = {name: importlib.import_module(name) for name in _SOURCES}
    for name, expected in _SOURCES.items():
        if (
            hashlib.sha256(Path(modules[name].__file__).read_bytes()).hexdigest()
            != expected
        ):
            raise ValueError(f"unsupported Sonic Quack dispatch contract: {name}")
    gemm = modules["quack.gemm"]
    interface = modules["quack.gemm_interface"]
    host = modules["quack.gemm_runtime.host"]
    epilogue = modules["quack.epilogue.frontend"].EpiMod
    return QuackCalls(
        (gemm._compile_gemm, host._compile_gemm_epi),
        host_caches=(
            (gemm, "_gemm_plan_cache"),
            (interface, "_gemm_iface_plan_cache"),
        ),
        host_cache_types=((epilogue, "_plan_cache"), (epilogue, "_call_cache")),
    )
