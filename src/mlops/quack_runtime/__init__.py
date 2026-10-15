"""Process-wide Quack setup shared by the expert-parallel layers and the
SonicMoE provider.

:func:`initialize` runs once per process before Quack compiles anything, and
again inside each of Quack's compile workers: it turns Quack's tuning-results
cache on, installs the source-checked patches (:mod:`.pipeline`,
:mod:`.autotune`, :mod:`.sweeps`) and registers every extra kernel-source
directory in Quack's cache fingerprint. :func:`autotune.sonicmoe_functional` imports SonicMoE's
kernels without adopting its global tuning policy.
"""

from __future__ import annotations

import os
from pathlib import Path

SOURCE_DIRS_VARIABLE = "MLOPS_QUACK_SOURCE_DIRS"
_initialized = False


def register_source_dir(path) -> None:
    """Add a directory of CuTe DSL kernel source to Quack's cache fingerprint.

    The list also goes into the environment, so Quack's compile workers --
    fresh processes started later -- fingerprint the same sources.
    """
    resolved = Path(path).resolve()
    listed = [
        item
        for item in os.environ.get(SOURCE_DIRS_VARIABLE, "").split(os.pathsep)
        if item
    ]
    if str(resolved) not in listed:
        listed.append(str(resolved))
        os.environ[SOURCE_DIRS_VARIABLE] = os.pathsep.join(listed)
    from quack import cache

    if resolved not in [Path(item).resolve() for item in cache.EXTRA_SOURCE_DIRS]:
        cache.EXTRA_SOURCE_DIRS.append(resolved)


def initialize() -> None:
    """Install the patches and the cache settings; repeated calls are no-ops."""
    global _initialized
    if _initialized:
        return
    # Quack keeps tuning results on disk only when asked. Every mlops process
    # asks, so separate processes pick the same configurations; QUACK_CACHE_DIR
    # says where (Quack's default is ~/.quack/cache).
    os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
    from . import autotune, pipeline, sweeps

    pipeline.apply()
    autotune.apply()
    sweeps.apply()
    # The patched kernel source lives here, so it is part of the fingerprint.
    register_source_dir(Path(__file__).resolve().parent)
    for item in os.environ.get(SOURCE_DIRS_VARIABLE, "").split(os.pathsep):
        if item:
            register_source_dir(item)
    _initialized = True


__all__ = ["SOURCE_DIRS_VARIABLE", "initialize", "register_source_dir"]
