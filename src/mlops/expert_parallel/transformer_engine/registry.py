"""Process-local runtime handles; no tensor payloads in graph metadata."""

from __future__ import annotations

import itertools
import threading

_RUNTIMES = {}
_IDS = itertools.count(1)
_REGISTRY_LOCK = threading.Lock()
_MATH_STREAMS = {}


def _register_runtime(r):
    with _REGISTRY_LOCK:
        handle = next(_IDS)
        _RUNTIMES[handle] = r
        return handle


def _runtime(handle):
    if handle not in _RUNTIMES:
        raise RuntimeError("Unknown/closed process-local MoonEP runtime handle")
    return _RUNTIMES[handle]
