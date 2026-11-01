"""General APIs, lazy dependencies, artifact bootstrap, and source provenance."""

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

import mlops
from mlops.kernels.router import routed_logits
from mlops.lora.grouped import grouped_linear


def test_namespace_import_does_not_load_accelerator_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import mlops; import mlops.bootstrap; "
                "assert 'fla' not in sys.modules; "
                "assert 'tilelang' not in sys.modules; "
                "assert 'mlops.kernels.kda.intra' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_public_exports_have_installed_implementations():
    for name in (
        "kimi_delta_attention",
        "kda_decay",
        "sparse_latent_attention",
        "sparse_mla",
        "pooled_topk",
        "sigmoid_topk",
        "mhc_normalize_streams",
        "mhc_coefficients",
        "mhc_combine",
    ):
        value = getattr(mlops, name)
        assert callable(value)
        assert value.__module__.startswith("mlops.")
        assert ".glm." not in value.__module__


def test_reference_source_matches_its_pinned_hash():
    directory = Path(__file__).parent / "references"
    manifest = json.loads((directory / "manifest.json").read_text())
    actual = hashlib.sha256(
        (directory / "modeling_glm5_next.py").read_bytes()
    ).hexdigest()
    assert actual == manifest["sha256"]


class _ExportedOperations(torch.nn.Module):
    def forward(self, q, kv, decay, beta, boundaries, chunks, indices, host_boundaries):
        kda = torch.ops.mlops.kimi_delta_attention_fla_fwd(
            q, q, q, decay, beta, boundaries, chunks, 0.125
        )
        sparse = torch.ops.mlops.sparse_latent_attention_tilelang_fwd(
            q, kv, indices, 0.125
        )
        selection = torch.ops.mlops.pooled_topk_builtin_triton(
            q, kv, kv, beta, kv[:2], host_boundaries, 4
        )
        return kda, sparse, selection


class _ExportedGroupedLinear(torch.nn.Module):
    def forward(self, x, weight, offsets):
        return grouped_linear(x, weight, offsets)


class _ExportedRoutedLogits(torch.nn.Module):
    def forward(self, hidden, weight, bias):
        return routed_logits(hidden, weight, bias)


def test_export_load_needs_only_normal_bootstrap(tmp_path):
    from torch._subclasses.fake_tensor import FakeTensorMode

    import mlops.bootstrap  # noqa: F401

    with FakeTensorMode():
        programs = {
            "operations": (
                _ExportedOperations(),
                (
                    torch.empty(17, 16, 64, dtype=torch.bfloat16, device="cuda"),
                    torch.empty(17, 64, dtype=torch.bfloat16, device="cuda"),
                    torch.empty(17, 16, 64, dtype=torch.float32, device="cuda"),
                    torch.empty(17, 16, dtype=torch.bfloat16, device="cuda"),
                    torch.empty(2, dtype=torch.int64, device="cuda"),
                    torch.empty(1, 2, dtype=torch.int64, device="cuda"),
                    torch.empty(17, 4, dtype=torch.int32, device="cuda"),
                    torch.empty(2, dtype=torch.int64),
                ),
            ),
            "grouped_linear": (
                _ExportedGroupedLinear(),
                (
                    torch.empty(17, 64, dtype=torch.bfloat16, device="cuda"),
                    torch.empty(4, 32, 64, dtype=torch.bfloat16, device="cuda"),
                    torch.empty(4, dtype=torch.int32, device="cuda"),
                ),
            ),
            "routed_logits": (
                _ExportedRoutedLogits(),
                (
                    torch.empty(17, 64, dtype=torch.bfloat16, device="cuda"),
                    torch.empty(8, 64, dtype=torch.bfloat16, device="cuda"),
                    torch.empty(8, dtype=torch.float32, device="cuda"),
                ),
            ),
        }
        paths = []
        for name, (module, args) in programs.items():
            path = tmp_path / f"{name}.pt2"
            torch.export.save(torch.export.export(module, args), path)
            paths.append(str(path))
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
import mlops.bootstrap
targets = set()
for path in sys.argv[1:]:
    program = torch.export.load(path)
    targets |= {str(n.target) for n in program.graph.nodes if n.op == "call_function"}
assert "mlops.kimi_delta_attention_fla_fwd.default" in targets
assert "mlops.sparse_latent_attention_tilelang_fwd.default" in targets
assert "mlops.pooled_topk_builtin_triton.default" in targets
assert "mlops.lora_grouped_linear_fwd.default" in targets
assert "mlops_router.logits.default" in targets
assert "fla" not in sys.modules
assert "tilelang" not in sys.modules
assert not any(n.startswith("mlops.glm") for n in sys.modules)
""",
            *paths,
        ],
        check=True,
    )


def test_bootstrap_registers_every_operator_outside_expert_parallel():
    root = Path(mlops.__file__).parent
    names = sorted(
        node.args[0].value
        for path in root.rglob("*.py")
        if "expert_parallel" not in path.relative_to(root).parts
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "torch.library.custom_op"
    )
    assert names
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
import mlops.bootstrap
missing = []
for name in sys.argv[1:]:
    namespace, operator = name.split("::")
    if not hasattr(getattr(torch.ops, namespace), operator):
        missing.append(name)
assert not missing, missing
""",
            *names,
        ],
        check=True,
    )
