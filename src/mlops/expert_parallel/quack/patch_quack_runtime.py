"""Process-local SM90 pipeline fix; installed Quack files stay unchanged."""

import hashlib
import importlib
import inspect
import linecache
import textwrap
from pathlib import Path

_applied = False
_original_pool_initializer = None


def _checked_source(function, expected):
    source = textwrap.dedent(inspect.getsource(function))
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise RuntimeError(
            "Unsupported Quack source for runtime patch: "
            + function.__qualname__
            + ". Install the unmodified Quack revision pinned by quack-moe."
        )
    return source


def _initialize_compile_worker(quack_arch, cute_dsl_arch):
    # Public imports are lazy. A spawned compiler process must explicitly
    # install the same patches before accepting compile work.
    from . import _initialize

    _initialize()
    _original_pool_initializer(quack_arch, cute_dsl_arch)


def apply():
    """Install before Quack compilation; repeated package imports are harmless."""
    global _applied, _original_pool_initializer
    if _applied:
        return False

    from quack import cache
    from quack.cache import async_compile
    from quack.cache.jit import _compute_source_fingerprint
    from quack.gemm_sm90 import GemmSm90

    if _compute_source_fingerprint.cache_info().currsize:
        raise RuntimeError(
            "Import mlops.expert_parallel.quack before compiling or launching Quack kernels"
        )
    if (
        async_compile._shared_executor is not None
        or async_compile._active_pool is not None
    ):
        raise RuntimeError(
            "Import mlops.expert_parallel.quack before starting a Quack compile pool"
        )

    source = _checked_source(
        GemmSm90.mma, "00d3897d7532bd035d39a0b7f4af2a350d24ed5f77342aa51de7b6b74ec82178"
    )
    source = source.replace(
        "    k_pipe_mmas = 1\n",
        "    # One stage must be released before the next tile can be loaded.\n"
        "    k_pipe_mmas = min(1, self.ab_stage - 1)\n",
    ).replace(
        "    if const_expr(self.fp8_slow_accum):\n"
        "        warpgroup.wait_group(0)\n"
        "        acc_slow.store(acc.load())\n",
        "    if const_expr(self.fp8_slow_accum):\n"
        "        if const_expr(k_pipe_mmas > 0):\n"
        "            warpgroup.wait_group(0)\n"
        "            acc_slow.store(acc.load())\n"
        "        else:\n"
        "            acc_slow.fill(0.0)\n",
    )
    # CuTe parses inspected source. Keep the replacement available to inspect,
    # including in compile workers, without writing the dependency checkout.
    filename = "<mlops.expert_parallel.quack.patch_quack_runtime.GemmSm90.mma>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(vars(importlib.import_module(GemmSm90.__module__)))
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 -- source hash checked above
    replacement = namespace["mma"]
    replacement.__qualname__ = GemmSm90.mma.__qualname__
    inspect.unwrap(replacement).__qualname__ = replacement.__qualname__

    GemmSm90.mma = replacement
    cache.EXTRA_SOURCE_DIRS.append(Path(__file__).resolve().parent)
    _original_pool_initializer = async_compile._pool_initializer
    async_compile._pool_initializer = _initialize_compile_worker
    _applied = True
    return True
