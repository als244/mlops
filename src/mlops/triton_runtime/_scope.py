"""Task-local interception of Triton's actual kernel-cache lookup.

Compilation hooks alone miss warm calls. Keep the original JITFunction object
and binder, and scope its device/kernel caches instead. Prepared bindings never
fall back to an eager cache or compiler. Only registered functions participate.
The artifact owner must validate/load binaries before binding them.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

_JIT_SOURCE = "5fe46fbfac257a2b8645b7a2cc709232273d42dd5e1d55de9770b412a9476d77"
_lock = RLock()


@dataclass(frozen=True, order=True)
class CallKey:
    function: str
    specialization: str


@dataclass
class _Scope:
    functions: frozenset[object]
    recorded: dict[CallKey, Any] | None = None
    prepared: Mapping[CallKey, Any] | None = None
    device: int | None = None
    caches: dict[tuple[object, int], tuple] = field(default_factory=dict)


_active: ContextVar[_Scope | None] = ContextVar("mlops_triton_calls", default=None)


class _Kernels(dict):
    def __init__(self, scope, owner, eager):
        self.scope, self.owner, self.eager = scope, owner, eager

    def get(self, key, default=None):
        identity = self.owner.identity(key)
        if self.scope.prepared is not None:
            try:
                return self.scope.prepared[identity]
            except KeyError as error:
                raise ValueError(
                    f"missing prepared Triton specialization: {identity}"
                ) from error
        value = self.eager.get(key, default)
        if value is not None:
            self.scope.recorded[identity] = value
        return value

    def __setitem__(self, key, value):
        if self.scope.prepared is not None:
            raise RuntimeError("prepared Triton replay tried to publish a kernel")
        self.eager[key] = value
        self.scope.recorded[self.owner.identity(key)] = value


class _DeviceCaches(MutableMapping):
    def __init__(self, function):
        self.function = function
        self.eager = function.device_caches
        self.token = object()
        self.name = function.__module__ + "." + function.__qualname__
        self.identities = {}

    def identity(self, key):
        # Runtime specialization strings include argument types, alignment,
        # constexprs and launch options. Never evaluate these strings on load.
        if not isinstance(key, str):
            raise TypeError("unsupported Triton runtime cache key")
        if key not in self.identities:
            self.identities[key] = CallKey(
                self.name, hashlib.sha256(key.encode()).hexdigest()
            )
        return self.identities[key]

    def __getitem__(self, device):
        scope = _active.get()
        if scope is None or self.token not in scope.functions:
            return self.eager[device]
        if scope.prepared is not None and device != scope.device:
            raise ValueError("prepared Triton binding used on another device")
        key = self.token, device
        if key not in scope.caches:
            # Creating the ordinary binder assigns Triton's ASTSource/compile
            # fields but does not compile. Preserve these even when a task was
            # first encountered in a fresh reader.
            eager, _, target, backend, binder = self.eager[device]
            scope.caches[key] = (
                _Kernels(scope, self, eager),
                {},
                target,
                backend,
                binder,
            )
        return scope.caches[key]

    def __setitem__(self, key, value):
        if _active.get() is not None:
            raise RuntimeError("cannot replace device caches inside Triton scope")
        self.eager[key] = value

    def __delitem__(self, key):
        if _active.get() is not None:
            raise RuntimeError("cannot clear device caches inside Triton scope")
        del self.eager[key]

    def __iter__(self):
        return iter(self.eager)

    def __len__(self):
        return len(self.eager)


class TritonCalls:
    """Observe cold/warm registered launches and bind already loaded objects.

    Capture follows ordinary eager compilation, including its disk cache. Binary
    export separately checks actual source/target/options provenance. Replay has
    persistent, task-owned binder caches and refuses absent specializations even
    if another task or eager execution already warmed them.
    """

    def __init__(self, functions: Sequence[Any]):
        jit = importlib.import_module("triton.runtime.jit")
        if hashlib.sha256(Path(jit.__file__).read_bytes()).hexdigest() != _JIT_SOURCE:
            raise ValueError("unsupported Triton device-cache contract")
        self.functions = {}
        with _lock:
            for function in functions:
                if type(function) is not jit.JITFunction:
                    raise TypeError("prepared Triton calls require plain JITFunctions")
                name = function.__module__ + "." + function.__qualname__
                if name in self.functions and self.functions[name] is not function:
                    raise ValueError("distinct Triton functions share one identity")
                self.functions[name] = function
            for function in self.functions.values():
                if not isinstance(function.device_caches, _DeviceCaches):
                    function.device_caches = _DeviceCaches(function)
        self.tokens = frozenset(
            function.device_caches.token for function in self.functions.values()
        )

    @contextmanager
    def capture(self) -> Iterator[dict[CallKey, Any]]:
        from triton.runtime import _async_compile

        if _async_compile.active_mode.get() is not None:
            raise ValueError("prepared Triton capture requires completed compilation")
        selected = {}
        token = _active.set(_Scope(self.tokens, recorded=selected))
        try:
            yield selected
        finally:
            _active.reset(token)

    def bind(self, prepared: Mapping[CallKey, Any], *, device: int):
        if type(device) is not int or device < 0:
            raise ValueError("prepared Triton binding needs an explicit device index")
        if any(key.function not in self.functions for key in prepared):
            raise ValueError("unregistered prepared Triton function")
        return BoundTritonCalls(
            _Scope(self.tokens, prepared=dict(prepared), device=device)
        )


class BoundTritonCalls:
    def __init__(self, scope):
        self.scope = scope

    @contextmanager
    def activate(self):
        token = _active.set(self.scope)
        try:
            yield
        finally:
            _active.reset(token)
