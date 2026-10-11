"""Optional NVTX ranges; independent of any training engine."""

from contextlib import contextmanager, nullcontext

import torch

_enabled = False


@contextmanager
def nvtx_annotations(enabled=True):
    """Enable nested ranges around the operators invoked inside this context."""
    global _enabled
    previous, _enabled = _enabled, enabled
    try:
        yield
    finally:
        _enabled = previous


def annotate(name):
    return torch.cuda.nvtx.range("seqmoe/" + name) if _enabled else nullcontext()
