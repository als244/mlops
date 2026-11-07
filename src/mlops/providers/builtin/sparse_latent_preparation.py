"""Save Hopper sparse-forward selection without probing inside execution.

This binds the FlashMLA LSE convention or the selected TileLang fallback. It
does not package attention binaries or prepare the cuDNN backward compiler.
Ordinary eager calls keep their existing detection/cache behavior.
"""

from contextlib import contextmanager
from hashlib import file_digest
from importlib import import_module
from pathlib import Path

from . import sparse_latent_hopper as provider


def _installed_flashmla():
    from flash_mla import flash_mla_sparse_fwd

    return flash_mla_sparse_fwd


def _identity(backend, function=None):
    modules = {"provider": provider}
    if backend == "flashmla":
        modules.update(
            package=import_module("flash_mla"),
            interface=import_module(function.__module__),
            extension=import_module("flash_mla.cuda"),
        )
    identity = {}
    for name, module in modules.items():
        with Path(module.__file__).open("rb") as stream:
            identity[name] = file_digest(stream, "sha256").hexdigest()
    return identity


class _Owner:
    def __init__(self):
        self.closed = False
        self.identities = {}

    def identity(self, backend, function=None):
        if backend not in self.identities:
            self.identities[backend] = _identity(backend, function)
        return self.identities[backend]

    def close(self):
        self.closed = True


class _Builder(_Owner):
    @contextmanager
    def capture(self):
        if self.closed:
            raise RuntimeError("sparse attention preparation is closed")
        captured = {}

        def select():
            entry = provider._detect_flashmla()
            if "entry" in captured and captured["entry"] != entry:
                raise ValueError("sparse forward selection changed within a task")
            captured["entry"] = entry
            return entry

        token = provider._flashmla_preparation.set(select)
        try:
            yield captured
        finally:
            provider._flashmla_preparation.reset(token)

    def export(self, captured):
        selection = None
        if "entry" in captured:
            entry = captured["entry"]
            backend = "tilelang" if entry is None else "flashmla"
            selection = {
                "backend": backend,
                "lse_to_log2": None if entry is None else entry[1],
                "identity": self.identity(backend, None if entry is None else entry[0]),
            }
        return {"version": 1, "selection": selection}, {}


class _Binding:
    def __init__(self, owner, used, entry):
        self.owner, self.used, self.entry = owner, used, entry

    def select(self):
        if not self.used:
            raise ValueError("sparse forward was not prepared for this task")
        return self.entry

    @contextmanager
    def activate(self):
        if self.owner.closed:
            raise RuntimeError("sparse attention preparation is closed")
        token = provider._flashmla_preparation.set(self.select)
        try:
            yield
        finally:
            provider._flashmla_preparation.reset(token)


class _Loader(_Owner):
    def bind(self, manifest, read_blob):
        if self.closed:
            raise RuntimeError("sparse attention preparation is closed")
        if set(manifest) != {"version", "selection"} or manifest["version"] != 1:
            raise ValueError("invalid sparse attention preparation")
        selection = manifest["selection"]
        if selection is None:
            return _Binding(self, False, None)
        if not isinstance(selection, dict) or set(selection) != {
            "backend",
            "lse_to_log2",
            "identity",
        }:
            raise ValueError("invalid saved sparse forward selection")
        backend, factor = selection["backend"], selection["lse_to_log2"]
        if backend == "flashmla":
            if type(factor) is not float or factor not in (1.0, provider._LOG2E):
                raise ValueError("invalid saved FlashMLA LSE convention")
            function = _installed_flashmla()
            entry = (function, factor)
        elif backend == "tilelang" and factor is None:
            function, entry = None, None
        else:
            raise ValueError("unsupported saved sparse forward selection")
        if self.identity(backend, function) != selection["identity"]:
            raise ValueError("saved sparse attention implementation differs")
        return _Binding(self, True, entry)


def factory(*, phase, device, options):
    """Capture actual calls; bind their backend and LSE convention without a GPU probe."""
    if options or phase not in ("build", "load"):
        raise ValueError("invalid sparse attention preparation options or phase")
    return (_Builder if phase == "build" else _Loader)()
