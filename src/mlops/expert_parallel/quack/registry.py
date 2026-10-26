"""Process-local runtime handles; no tensor payloads in graph metadata."""

from __future__ import annotations

import hashlib
import itertools
import threading

_RUNTIMES = {}
_IDS = itertools.count(1)
_REGISTRY_LOCK = threading.Lock()


def resource_handle(name):
    """Stable negative handles cannot collide with ordinary positive live IDs."""
    if not isinstance(name, str) or not name:
        raise ValueError("resource name must be a nonempty string")
    return -(
        int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big") % (2**63 - 1)
        + 1
    )


def _register_runtime(r, *, name=None):
    with _REGISTRY_LOCK:
        handle = next(_IDS) if name is None else resource_handle(name)
        if handle in _RUNTIMES:
            raise ValueError(f"runtime resource is already open: {name}")
        _RUNTIMES[handle] = r
        return handle


def _runtime(handle):
    if handle not in _RUNTIMES:
        raise RuntimeError("Unknown/closed process-local MoonEP runtime handle")
    return _RUNTIMES[handle]
