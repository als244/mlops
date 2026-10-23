"""Compatibility alias for :mod:`mlops.kernels.quack_moe.fp8`.

The implementation is local expert math shared by DP and EP providers.
"""

import sys

from mlops.kernels.quack_moe import fp8 as _implementation

sys.modules[__name__] = _implementation
