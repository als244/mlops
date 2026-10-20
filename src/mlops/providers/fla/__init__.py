"""Optional Flash Linear Attention implementation adapters.

FLA memoizes its sequence-index helpers (``fla.utils.tensor_cache``, such as
the chunk offsets its delta-rule kernels derive from the cumulative lengths)
by argument identity, and keeps the tensors it computed. An operation routed
here must allocate the same way on every call: a planner profiles a task once
and replays it, and reclaims what a closed plan's tasks allocated, so a result
cached in one plan and found by the next changes what a task allocates. The
adapters therefore turn that cache off when they load, for FLA imported before
or after them; FLA then recomputes a few small index tensors per call.
"""

from __future__ import annotations

import os
import sys

os.environ["FLA_DISABLE_TENSOR_CACHE"] = "1"
_fla_utils = sys.modules.get("fla.utils")
if _fla_utils is not None:
    _fla_utils.FLA_DISABLE_TENSOR_CACHE = True
del _fla_utils
