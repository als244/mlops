"""Hardware gates for GPU tests.

A test marked with one of the features below skips on a device whose compute
capability is below the feature's, and runs on every device that has it. The
device is queried only when a marked test is collected.
"""

from functools import cache

import pytest
import torch

# Marker: (lowest compute capability, what needs it).
FEATURES = {
    # Triton has no E4M3 type (fp8e4nv) before Ada and Hopper.
    "e4m3": ((8, 9), "Triton's E4M3 type"),
    # Quack's GEMM has kernels for SM90, SM100 and SM120, none for SM80.
    "quack_gemm": ((9, 0), "Quack's GEMM"),
}


def pytest_configure(config):
    for name, ((major, minor), feature) in FEATURES.items():
        config.addinivalue_line(
            "markers",
            f"{name}: needs {feature}, compute capability {major}.{minor} or newer",
        )


@cache
def _capability():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


def pytest_collection_modifyitems(items):
    for item in items:
        for name, (lowest, feature) in FEATURES.items():
            if item.get_closest_marker(name) is None:
                continue
            capability = _capability()
            if capability is not None and capability >= lowest:
                continue
            required = f"{feature} needs compute capability {lowest[0]}.{lowest[1]}"
            found = (
                "no CUDA device"
                if capability is None
                else f"this GPU is {capability[0]}.{capability[1]}"
            )
            item.add_marker(pytest.mark.skip(reason=f"{required} or newer; {found}"))
