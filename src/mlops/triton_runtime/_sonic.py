"""Explicit BF16 SonicMoE routing/activation/gather Triton registration.

The caller still freezes autotuning choices independently. This registration
does not cover FP8, expert parallelism, unrelated attention providers, or a
provider implementation that bypasses these JITFunctions with its own launcher.
"""

import importlib

from ._scope import TritonCalls


def sonic_triton_calls():
    from mlops.kernels import moe_router
    from mlops.quack_runtime import initialize
    from mlops.quack_runtime.autotune import sonicmoe_functional

    initialize()
    functional = sonicmoe_functional()
    metadata = functional.triton_kernels
    bitmatrix = importlib.import_module("sonicmoe.functional.triton_kernels.bitmatrix")
    reduction = importlib.import_module("sonicmoe.functional.reduction_over_k_gather")
    activation = importlib.import_module("mlops.kernels.quack_moe.swiglu_rows")
    return TritonCalls(
        (
            moe_router._topk_kernel,
            moe_router._backward_kernel,
            metadata._compute_col_partial_sum_kernel,
            bitmatrix._bitmatrix_metadata_compute_stage1,
            bitmatrix._bitmatrix_metadata_compute_stage2,
            reduction.token_gather_sum_kernel.fn,
            activation._swiglu_rows,
            activation._swiglu_rows_backward,
            activation._routed_swiglu_rows_backward,
        )
    )
