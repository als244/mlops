"""Prepare the optional runtime used by MLOps' ordinary BF16 device dispatch.

The caller opts in before build. Device-dependent selection happens here when
the engine opens the factory, never while creating a description on the CPU.
Saved loading follows the recorded provider, not today's dispatch defaults.
"""

from contextlib import contextmanager, nullcontext

import torch


class _Builder:
    def __init__(self, device, options):
        from mlops.providers.sonicmoe.moe import CAPABILITIES, available

        self.owner = None
        if torch.cuda.get_device_capability(device)[0] in CAPABILITIES and available():
            from .sonic import factory

            self.owner = factory(phase="build", device=device, options=options)

    @contextmanager
    def capture(self):
        with self.owner.capture() if self.owner else nullcontext() as captured:
            yield captured

    def export(self, captured):
        if self.owner is None:
            return {"version": 1, "provider": None, "manifest": None}, {}
        manifest, blobs = self.owner.export(captured)
        return {"version": 1, "provider": "sonic", "manifest": manifest}, blobs

    def close(self):
        if self.owner is not None:
            self.owner.close()


class _Empty:
    def activate(self):
        return nullcontext()


class _Loader:
    def __init__(self, device, options):
        self.device, self.options = device, options
        self.owner = None

    def bind(self, manifest, read_blob):
        if (
            set(manifest) != {"version", "provider", "manifest"}
            or manifest["version"] != 1
        ):
            raise ValueError("invalid MLOps preparation manifest")
        if manifest["provider"] is None:
            if manifest["manifest"] is not None:
                raise ValueError("invalid empty MLOps preparation")
            return _Empty()
        if manifest["provider"] != "sonic":
            raise ValueError("unsupported saved MLOps preparation provider")
        if self.owner is None:
            from .sonic import factory

            self.owner = factory(phase="load", device=self.device, options=self.options)
        return self.owner.bind(manifest["manifest"], read_blob)

    def close(self):
        if self.owner is not None:
            self.owner.close()


def factory(*, phase, device, options):
    """Provider-owned selection for supported BF16 Sonic, otherwise no assets.

    The description does not select operators or force Sonic onto another GPU.
    This is not preparation support for FP8, EP or arbitrary MLOps providers.
    """
    if set(options) - {"cache_root"}:
        raise ValueError("unknown MLOps preparation option")
    if phase not in {"build", "load"}:
        raise ValueError("unknown MLOps preparation lifecycle phase")
    return (_Builder if phase == "build" else _Loader)(device, options)
