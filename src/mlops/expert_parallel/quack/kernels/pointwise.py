"""Compatibility alias for :mod:`mlops.kernels.quack_moe.pointwise`.

The implementation is local expert math shared by DP and EP providers.
"""

import sys

from mlops.kernels.quack_moe import pointwise as _implementation

sys.modules[__name__] = _implementation
