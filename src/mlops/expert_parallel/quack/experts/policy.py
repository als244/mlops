"""Compatibility alias for :mod:`mlops.kernels.quack_moe.policy`.

The implementation is local expert math shared by DP and EP providers.
"""

import sys

from mlops.kernels.quack_moe import policy as _implementation

sys.modules[__name__] = _implementation
