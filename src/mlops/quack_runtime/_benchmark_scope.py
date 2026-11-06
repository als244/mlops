"""Keep autotuner benchmark launches out of selected binary inventories.

Both pinned tuners benchmark inside `_bench`, then invoke their winner once
after selection. Leave benchmarking and its mutation-preservation hooks intact;
suspend only the task-local inventory while each benchmark runs. No extra task
invocation is introduced. Prepared replay may never enter a benchmark.
"""

import hashlib
import importlib
import inspect
from contextlib import ExitStack
from functools import wraps
from threading import RLock

_SOURCES = {
    "quack.autotuner": "2f1cdb7274067cfa075204e1e77a7e3c1f61c66e35b004b2785c8bc604888ad6",
    "triton.runtime.autotuner": "4338ab90358c9930b1f7d9c7882f0b3433225c2aa9078c8601f453bc5ba9be76",
}
_lock = RLock()
_installed = False


def _bench(original, *args, **kwargs):
    from mlops.triton_runtime._scope import _active as triton

    from ._cache_scope import _active as quack

    scopes = (quack, triton)
    for variable in scopes:
        scope = variable.get()
        if scope is not None and scope.prepared is not None:
            raise RuntimeError("prepared replay cannot benchmark operator candidates")
    with ExitStack() as cleanup:
        for variable in scopes:
            if variable.get() is not None:
                cleanup.callback(variable.reset, variable.set(None))
        return original(*args, **kwargs)


def install():
    global _installed
    from . import initialize

    initialize()
    with _lock:
        if _installed:
            return
        checked = []
        for name, expected in _SOURCES.items():
            owner = importlib.import_module(name).Autotuner
            original = owner._bench
            if (
                hasattr(original, "__wrapped__")
                or hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
                != expected
            ):
                raise ValueError(
                    f"unsupported selected-kernel benchmark contract: {name}"
                )
            checked.append((owner, original))
        for owner, original in checked:

            def make_wrapper(function):
                @wraps(function)
                def call(*args, **kwargs):
                    return _bench(function, *args, **kwargs)

                return call

            owner._bench = make_wrapper(original)
        _installed = True
