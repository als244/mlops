"""Preserve only GEMM arguments whose original values tuning can destroy.

Contracts are installed on the pinned Quack entry points, not inferred from
argument names on arbitrary functions. Unknown functions keep the conservative
all-tensor fallback. Read-only weights and overwritten outputs need no backup.
"""

import inspect
from functools import wraps

import torch

from .pipeline import _checked_source

_applied = False


def _overlaps(left, right):
    if left.device != right.device or not left.numel() or not right.numel():
        return False
    def span(tensor):
        size = tensor.element_size()
        length = size * (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride())))
        return tensor.data_ptr(), tensor.data_ptr() + length
    a, b = span(left), span(right)
    return a[0] < b[1] and b[0] < a[1]


def tensors_to_preserve(fn, args, kwargs):
    """Return mutable inputs, including outputs aliasing a read operand."""
    contract = getattr(fn, "_mlops_tune_mutation_contract", None)
    if contract is None:
        return [v for v in (*args, *kwargs.values()) if isinstance(v, torch.Tensor)]
    parameters = inspect.signature(fn).parameters
    named = dict(zip(parameters, args)) | kwargs
    writes, readwrites = map(set, contract)
    if named.get("add_to_output", False):
        readwrites.add("out")
    tensors = {n: v for n, v in named.items() if isinstance(v, torch.Tensor)}
    reads = [v for n, v in tensors.items() if n not in writes or n in readwrites]
    return [
        value for name, value in tensors.items()
        if name in readwrites
        or (name in writes and any(_overlaps(value, read) for read in reads))
    ]


def apply():
    """Attach contracts only after checking the supported Quack source."""
    global _applied
    if _applied:
        return False
    from quack.gemm_interface import gemm_tuned
    from quack.gemm_runtime import autotune

    _checked_source(gemm_tuned.fn, "7db21cb28c2358f777b47aa252b44e342c54999a0d516b230e4e28ad09afb313")
    original = autotune._make_tuned_fn
    _checked_source(original, "7280a143b8e3b0494b3e045281cd31db698395e48439a53eee9022a8c04037ab")
    gemm_tuned.fn._mlops_tune_mutation_contract = (("out", "SFD", "SFDCol"), ())

    @wraps(original)
    def make_tuned_fn(mod, epi_names, transform_a=None, ta_names=()):
        fn = original(mod, epi_names, transform_a, ta_names)
        writes = {"D", *mod.outputs, *mod.sinks}
        # Keep sink/extra-op state conservatively; ordinary tile outputs are
        # overwritten. An output also used by the epilogue is read/write.
        readwrites = set(mod.sinks) | (writes & set(mod.operand_names))
        readwrites.update(op.name for op in mod.extra_ops)
        readwrites.update(writes & set(mod.prepass_operand_names))
        fn._mlops_tune_mutation_contract = (tuple(writes), tuple(readwrites))
        return fn

    autotune._make_tuned_fn = make_tuned_fn
    _applied = True
    return True
