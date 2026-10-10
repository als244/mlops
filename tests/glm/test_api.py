"""The installed APIs and numerical reference do not depend on experiments."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from mlops import glm


def test_namespace_import_does_not_load_accelerator_dependencies():
    subprocess.run(
        [sys.executable, "-c", (
            "import sys; import mlops.glm; "
            "assert 'tilelang' not in sys.modules; "
            "assert 'mlops.glm.kernels.kda_intra' not in sys.modules"
        )],
        check=True,
    )


def test_public_exports_have_installed_implementations():
    for name in glm.__all__:
        value = getattr(glm, name)
        assert callable(value)
        assert value.__module__.startswith("mlops.glm.")
    import mlops.glm.bootstrap  # noqa: F401


def test_reference_source_matches_its_pinned_hash():
    directory = Path(__file__).parent / "references"
    manifest = json.loads((directory / "manifest.json").read_text())
    actual = hashlib.sha256((directory / "modeling_glm5_next.py").read_bytes()).hexdigest()
    assert actual == manifest["sha256"]
