"""Compatibility alias for :mod:`mlops.kernels.quack_moe.block_fp8_gemm`.

The implementation is local expert math shared by DP and EP providers.
"""

import sys

from mlops.kernels.quack_moe import block_fp8_gemm as _implementation

sys.modules[__name__] = _implementation
