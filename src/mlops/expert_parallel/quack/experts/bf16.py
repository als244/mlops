"""Compatibility alias for :mod:`mlops.kernels.quack_moe.bf16`.

The implementation is local expert math shared by DP and EP providers.
"""

import sys

from mlops.kernels.quack_moe import bf16 as _implementation

sys.modules[__name__] = _implementation
