"""Private task-local binding for explicitly registered Quack compile calls.

This is the cache-dispatch mechanism, not a binary artifact or target contract.
The owner must validate and load immutable binaries before entering replay.
Unregistered functions keep ordinary behavior; complete compiler coverage needs
separate qualification. No profiler or process-wide cache switch is required.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock
from types import CodeType, FunctionType
from typing import Any

_SOURCE = "4ff940b933d9ef8b6b3217741a7f9184a7ce50631801fa392ea61f216af7e160"
_lock = RLock()


@dataclass(frozen=True)
class CallKey:
    function: str
    specialization: str


@dataclass(frozen=True)
class _Scope:
    functions: frozenset[int]
    recorded: dict[CallKey, Any] | None = None
    prepared: Mapping[CallKey, Any] | None = None
    hosts: Mapping[int, dict] = field(default_factory=dict)


_active: ContextVar[_Scope | None] = ContextVar("mlops_quack_binaries", default=None)


class _HostCache(MutableMapping):
    def __init__(self, original):
        self.eager = original

    def _data(self):
        scope = _active.get()
        if scope is not None and id(self) in scope.hosts:
            return scope.hosts[id(self)]
        return self.eager

    def __getitem__(self, key):
        return self._data()[key]

    def __setitem__(self, key, value):
        self._data()[key] = value

    def __delitem__(self, key):
        del self._data()[key]

    def __iter__(self):
        return iter(self._data())

    def __len__(self):
        return len(self._data())


class _Cache(dict):
    def __init__(self, original, function, key_hash):
        super().__init__(original)
        self.function = function.__module__ + "." + function.__qualname__
        self.qualname = function.__qualname__
        self.key_hash = key_hash
        self.identities = {}

    def _key(self, key):
        if key not in self.identities:
            self.identities[key] = CallKey(
                self.function, self.key_hash((self.qualname,) + key)
            )
        return self.identities[key]

    def _scope(self):
        scope = _active.get()
        return scope if scope is not None and id(self) in scope.functions else None

    def _prepared(self, scope, key):
        identity = self._key(key)
        try:
            return scope.prepared[identity]
        except KeyError as error:
            raise ValueError(
                f"missing prepared Quack specialization: {identity.function} "
                f"{identity.specialization}"
            ) from error

    def __contains__(self, key):
        if (scope := self._scope()) is not None and scope.prepared is not None:
            self._prepared(scope, key)
            return True
        return super().__contains__(key)

    def __getitem__(self, key):
        scope = self._scope()
        if scope is not None and scope.prepared is not None:
            return self._prepared(scope, key)
        value = super().__getitem__(key)
        if scope is not None:
            scope.recorded[self._key(key)] = value
        return value

    def __setitem__(self, key, value):
        scope = self._scope()
        if scope is not None and scope.prepared is not None:
            raise RuntimeError(
                "prepared Quack replay tried to publish a compiled kernel"
            )
        super().__setitem__(key, value)
        if scope is not None:
            scope.recorded[self._key(key)] = value


def _install(functions: Sequence[Callable[..., Any]]) -> frozenset[int]:
    module = importlib.import_module("quack.cache.jit")
    source = hashlib.sha256(inspect.getsource(module.jit_cache).encode()).hexdigest()
    if source != _SOURCE:
        raise ValueError("unsupported Quack binary cache contract")
    codes = [
        code
        for code in module.jit_cache.__code__.co_consts
        if isinstance(code, CodeType) and code.co_name == "wrapper"
    ]
    if len(codes) != 1:
        raise ValueError("unsupported Quack binary cache wrapper")
    names = {}
    with _lock:
        for function in functions:
            if (
                not isinstance(function, FunctionType)
                or function.__code__ is not codes[0]
            ):
                raise ValueError(
                    "prepared Quack calls require checked jit_cache functions"
                )
            cells = dict(
                zip(function.__code__.co_freevars, function.__closure__, strict=True)
            )
            original = cells["fn"].cell_contents
            cache = cells["cache"].cell_contents
            if function.cache is not cache or not isinstance(cache, dict):
                raise ValueError(
                    "Quack cache storage differs from its checked contract"
                )
            if not isinstance(cache, _Cache):
                cache = _Cache(cache, original, module._key_to_hash)
                # Preserve the function object: aliases imported before this
                # registration, cache_info and cache_clear share this cell.
                cells["cache"].cell_contents = cache
                function.cache = cache
            previous = names.setdefault(cache.function, id(cache))
            if previous != id(cache):
                raise ValueError("distinct Quack functions share a prepared identity")
    return frozenset(names.values())


class QuackCalls:
    """Validate registered calls once and isolate their declared host caches.

    The provider adapter must enumerate every outer dispatch cache that can
    bypass a compile call. Slots are module/object attributes, never a cache
    discovered from tensor contents or a model-specific naming convention.
    """

    def __init__(
        self,
        functions: Sequence[Callable[..., Any]],
        *,
        host_caches: Sequence[tuple[Any, str]] = (),
    ):
        self.functions = tuple(functions)
        self.identities = _install(self.functions)
        self.hosts = []
        with _lock:
            for owner, name in host_caches:
                cache = getattr(owner, name)
                if not isinstance(cache, _HostCache):
                    if not isinstance(cache, dict):
                        raise TypeError(
                            "Quack host-cache slot must contain a dictionary"
                        )
                    cache = _HostCache(cache)
                    setattr(owner, name, cache)
                self.hosts.append(cache)

    def _host_state(self):
        return {id(cache): {} for cache in self.hosts}

    @contextmanager
    def capture(self) -> Iterator[dict[CallKey, Any]]:
        """Record selected calls, including already-warm memory hits."""
        recorded = {}
        token = _active.set(
            _Scope(self.identities, recorded=recorded, hosts=self._host_state())
        )
        try:
            yield recorded
        finally:
            _active.reset(token)

    @contextmanager
    def use(self, prepared: Mapping[CallKey, Any]) -> Iterator[None]:
        """Use prevalidated objects; missing calls refuse before compilation."""
        with self.bind(prepared).activate():
            yield

    def bind(self, prepared: Mapping[CallKey, Any]) -> BoundQuackCalls:
        """Create persistent host dispatch state owned by this prepared task."""
        return BoundQuackCalls(
            _Scope(self.identities, prepared=dict(prepared), hosts=self._host_state())
        )


class BoundQuackCalls:
    def __init__(self, scope: _Scope):
        self.scope = scope

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _active.set(self.scope)
        try:
            yield
        finally:
            _active.reset(token)
