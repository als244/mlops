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
from functools import wraps
from pathlib import Path
from threading import RLock
from types import CodeType, FunctionType
from typing import Any

_SOURCE = "4ff940b933d9ef8b6b3217741a7f9184a7ce50631801fa392ea61f216af7e160"
_lock = RLock()
_path_installed = False


@dataclass(frozen=True)
class CallKey:
    function: str
    specialization: str


@dataclass(frozen=True)
class _Scope:
    functions: frozenset[int]
    recorded: dict[CallKey, Any] | None = None
    prepared: Mapping[CallKey, Any] | None = None
    hosts: dict[object, dict] = field(default_factory=dict)
    host_groups: frozenset[object] = frozenset()
    directory: Path | None = None
    objects: dict[tuple[int, tuple], Any] = field(default_factory=dict)


_active: ContextVar[_Scope | None] = ContextVar("mlops_quack_binaries", default=None)


class _HostCache(MutableMapping):
    def __init__(self, original, group=None):
        self.eager = original
        self.group = group
        self.key = object()

    def _data(self):
        scope = _active.get()
        if scope is not None:
            if self.key in scope.hosts:
                return scope.hosts[self.key]
            if self.group in scope.host_groups:
                return scope.hosts.setdefault(self.key, {})
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

    def __reduce__(self):
        # Workers reconstruct their own dispatch state. Never pickle a task's
        # bound functions or ContextVar state through a provider object.
        return dict, (self.eager,)


class _HostSlot:
    """A declared instance dictionary, including instances created lazily."""

    def __init__(self, name):
        self.name = name
        self.group = object()

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        value = instance.__dict__[self.name]
        if not isinstance(value, _HostCache):
            with _lock:
                value = instance.__dict__[self.name]
                if not isinstance(value, _HostCache):
                    self.__set__(instance, value)
        elif value.group is not self.group:
            raise ValueError("Quack host cache belongs to a different registration")
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if not isinstance(value, dict):
            raise TypeError("Quack instance host-cache slot must contain a dictionary")
        instance.__dict__[self.name] = _HostCache(value, self.group)


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
        if scope is not None and scope.directory is not None:
            return (id(self), key) in scope.objects
        return super().__contains__(key)

    def __getitem__(self, key):
        scope = self._scope()
        if scope is not None and scope.prepared is not None:
            return self._prepared(scope, key)
        value = (
            scope.objects[id(self), key]
            if scope is not None and scope.directory is not None
            else super().__getitem__(key)
        )
        if scope is not None:
            scope.recorded[self._key(key)] = value
        return value

    def __setitem__(self, key, value):
        scope = self._scope()
        if scope is not None and scope.prepared is not None:
            raise RuntimeError(
                "prepared Quack replay tried to publish a compiled kernel"
            )
        if scope is not None and scope.directory is not None:
            scope.objects[id(self), key] = value
        else:
            super().__setitem__(key, value)
        if scope is not None:
            scope.recorded[self._key(key)] = value


def _install_cache_path():
    global _path_installed
    with _lock:
        if _path_installed:
            return
        module = importlib.import_module("quack.cache.jit")
        expected = "09a1d8f626d31500e175556b3cd8f68dfc60599cf102f1df9c22aac781ac7193"
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise ValueError("unsupported Quack binary directory contract")
        original = module.get_cache_path

        @wraps(original)
        def cache_path():
            scope = _active.get()
            if scope is None or scope.directory is None:
                return original()
            pool = importlib.import_module("quack.cache.async_compile")
            if pool.get_active_pool() is not None:
                raise RuntimeError(
                    "prepared Quack capture requires selected calls outside autotuning"
                )
            scope.directory.mkdir(parents=True, exist_ok=True)
            return scope.directory

        module.get_cache_path = cache_path
        _path_installed = True


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
        host_cache_types: Sequence[tuple[type, str]] = (),
    ):
        self.functions = tuple(functions)
        self.identities = _install(self.functions)
        self.hosts = []
        self.host_groups = set()
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
            for owner, name in host_cache_types:
                slot = inspect.getattr_static(owner, name, None)
                if slot is None:
                    slot = _HostSlot(name)
                    setattr(owner, name, slot)
                elif not isinstance(slot, _HostSlot):
                    raise TypeError(
                        "Quack instance host-cache slot has an unsupported descriptor"
                    )
                self.host_groups.add(slot.group)

    def _host_state(self):
        return {cache.key: {} for cache in self.hosts}

    @contextmanager
    def capture(
        self, *, cache_directory: Path | None = None
    ) -> Iterator[dict[CallKey, Any]]:
        """Record selected calls, including already-warm memory hits.

        An artifact owner can supply a source/target-qualified directory. That
        mode isolates binary memory caches too: ordinary eager entries carry no
        target provenance and cannot establish a prepared artifact's identity.
        Selected calls must run outside an asynchronous compilation pool.
        """
        if cache_directory is not None:
            _install_cache_path()
            state = importlib.import_module("quack.cache")
            if not state.CACHE_ENABLED:
                raise ValueError(
                    "prepared Quack capture requires its binary disk cache"
                )
            cache_directory = Path(cache_directory)
        recorded = {}
        token = _active.set(
            _Scope(
                self.identities,
                recorded=recorded,
                hosts=self._host_state(),
                host_groups=frozenset(self.host_groups),
                directory=cache_directory,
            )
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
            _Scope(
                self.identities,
                prepared=dict(prepared),
                hosts=self._host_state(),
                host_groups=frozenset(self.host_groups),
            )
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
