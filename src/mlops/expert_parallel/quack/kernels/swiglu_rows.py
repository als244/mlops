"""Compatibility alias for :mod:`mlops.kernels.quack_moe.swiglu_rows`.

The implementation is local expert math shared by DP and EP providers.
"""

import sys

from mlops.kernels.quack_moe import swiglu_rows as _implementation

sys.modules[__name__] = _implementation
