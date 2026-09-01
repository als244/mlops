"""Dispatcher-contract checks for every opaque training operation."""

import ast
import importlib
from pathlib import Path

import mlops
import pytest
import torch
from mlops.preparation.packed_sequence import _prepare_op
from mlops.providers.builtin.adamw import (
    _functional_op,
    _in_place_op,
    _master_functional_op,
    _master_in_place_op,
    _master_out_op,
    _out_op,
)


# Import the defining modules without reaching through their public function
# aliases. The model-facing package intentionally exports only semantic calls.


fla_hybrid = importlib.import_module(
    "mlops.providers.fla.hybrid"
)
dsa = importlib.import_module(
    "mlops.providers.builtin.dsa"
)
cross_entropy = importlib.import_module(
    "mlops.providers.builtin.cross_entropy"
)
embedding = importlib.import_module(
    "mlops.providers.builtin.embedding"
)
flash_attention = importlib.import_module(
    "mlops.providers.builtin.flash_attention"
)
gelu = importlib.import_module(
    "mlops.providers.builtin.gelu"
)
head = importlib.import_module(
    "mlops.providers.builtin.head"
)
layer_norm = importlib.import_module(
    "mlops.providers.builtin.layer_norm"
)
moe = importlib.import_module(
    "mlops.providers.builtin.moe"
)
moe_composed = importlib.import_module(
    "mlops.providers.builtin.moe_composed"
)
partial_rope = importlib.import_module(
    "mlops.providers.builtin.partial_rope"
)
rms_norm = importlib.import_module("mlops.rms_norm")
rms_norm_builtin = importlib.import_module(
    "mlops.providers.builtin.rms_norm"
)
rms_norm_liger = importlib.import_module(
    "mlops.providers.liger.rms_norm"
)
scattermoe = importlib.import_module(
    "mlops.providers.scattermoe.moe"
)
rope = importlib.import_module(
    "mlops.providers.builtin.rope"
)
swiglu = importlib.import_module(
    "mlops.providers.builtin.swiglu"
)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


PACKAGE_ROOT = Path(mlops.__file__).resolve().parent


FORWARD_OPERATORS = {
    "causal_conv_silu_fla_fwd": fla_hybrid._causal_forward_op,
    "cross_entropy_builtin_triton_fwd": cross_entropy._forward_op,
    "dsa_attention_builtin_sparse_fwd": dsa._attention_forward_op,
    "embedding_builtin_deterministic_fwd": embedding._forward_op,
    "flash_attention_builtin_aten_fwd": flash_attention._forward_op,
    "gated_rms_norm_fla_fwd": fla_hybrid._gated_forward_op,
    "gelu_builtin_aten_explicit_backward_fwd": gelu._forward_op,
    "head_loss_builtin_chunked_fwd": head._forward_op,
    "l2_norm_fla_fwd": fla_hybrid._l2_forward_op,
    "layer_norm_builtin_triton_fwd": layer_norm._forward_op,
    "linear_attention_fla_gated_delta_rule_fwd": fla_hybrid._linear_forward_op,
    "moe_builtin_grouped_gemm_fwd": moe._forward_op,
    "moe_builtin_grouped_gemm_prepare_fwd": moe_composed._prepare_op,
    "moe_builtin_grouped_gemm_finish_fwd": moe_composed._finish_op,
    "moe_scattermoe_fwd": scattermoe._forward_op,
    "partial_rope_builtin_triton_fwd": partial_rope._forward_op,
    "rms_norm_builtin_triton_fwd": rms_norm_builtin._forward_op,
    "rms_norm_liger_fwd": rms_norm_liger._forward_op,
    "rope_builtin_triton_table_fwd": rope._table_forward_op,
    "rope_builtin_triton_analytic_fwd": rope._analytic_forward_op,
    "packed_swiglu_builtin_triton_fwd": swiglu._packed_forward_op,
    "packed_swiglu_builtin_triton_legacy_fwd": swiglu._packed_legacy_forward_op,
    "swiglu_builtin_triton_fwd": swiglu._forward_op,
    "swiglu_builtin_triton_legacy_fwd": swiglu._legacy_forward_op,
}

BACKWARD_OPERATORS = {
    "causal_conv_silu_fla_bwd",
    "cross_entropy_builtin_triton_bwd",
    "dsa_attention_builtin_sparse_bwd",
    "embedding_builtin_deterministic_bwd",
    "flash_attention_builtin_aten_bwd",
    "gated_rms_norm_fla_bwd",
    "gelu_builtin_aten_explicit_backward_bwd",
    "l2_norm_fla_bwd",
    "layer_norm_builtin_triton_bwd",
    "linear_attention_fla_gated_delta_rule_bwd",
    "moe_builtin_grouped_gemm_bwd",
    "moe_builtin_grouped_gemm_prepare_bwd",
    "moe_builtin_grouped_gemm_finish_bwd",
    "moe_scattermoe_bwd",
    "partial_rope_builtin_triton_bwd",
    "rms_norm_builtin_triton_bwd",
    "rms_norm_liger_bwd",
    "rope_builtin_triton_table_bwd",
    "rope_builtin_triton_analytic_bwd",
    "packed_swiglu_builtin_triton_bwd",
    "packed_swiglu_builtin_triton_legacy_bwd",
    "swiglu_builtin_triton_bwd",
    "swiglu_builtin_triton_legacy_bwd",
}

AUXILIARY_OPERATORS = {
    "prepare_packed_sequence_metadata": _prepare_op,
    "adamw": _functional_op,
    "adamw_out": _out_op,
    "adamw_": _in_place_op,
    "master_adamw": _master_functional_op,
    "master_adamw_out": _master_out_op,
    "master_adamw_": _master_in_place_op,
    "moe_builtin_grouped_gemm_finish_bwd_donate_h13": (
        moe_composed._finish_backward_donate_h13_op
    ),
}


def _parameter(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)


def _embedding_args():
    return torch.tensor([[1, 3, 1, 5]], device="cuda"), _parameter(17, 16)


def _rms_norm_args():
    return _parameter(6, 16), _parameter(16), 1e-5


def _layer_norm_args():
    return _parameter(6, 16), _parameter(16), _parameter(16), 1e-5


def _rope_args():
    rows, width = 6, 16
    inverse = 1.0 / (
        10_000.0
        ** (
            torch.arange(0, width, 2, device="cuda", dtype=torch.float32)
            / width
        )
    )
    frequencies = torch.outer(
        torch.arange(rows, device="cuda", dtype=torch.float32), inverse
    )
    table = torch.cat((frequencies, frequencies), dim=-1)
    return (
        _parameter(rows, 2, width),
        torch.arange(rows, device="cuda", dtype=torch.int32),
        10_000.0,
        table.cos(),
        table.sin(),
    )


def _partial_rope_args():
    rows, width = 6, 8
    inverse = 1.0 / (
        10_000.0
        ** (
            torch.arange(0, width, 2, device="cuda", dtype=torch.float32)
            / width
        )
    )
    frequencies = torch.outer(
        torch.arange(rows, device="cuda", dtype=torch.float32), inverse
    )
    table = torch.cat((frequencies, frequencies), dim=-1)
    return (
        _parameter(6, 2, 16),
        torch.arange(6, device="cuda", dtype=torch.int32),
        10_000.0,
        width,
        table.cos(),
        table.sin(),
    )


def _swiglu_args():
    return _parameter(6, 32), _parameter(6, 32)


def _packed_swiglu_args():
    return (_parameter(6, 64),)


def _gelu_args():
    return (_parameter(6, 32),)


def _cross_entropy_args():
    return (
        _parameter(7, 31),
        torch.randint(0, 31, (7,), device="cuda"),
        -100,
    )


def _head_args():
    return (
        _parameter(7, 16),
        _parameter(31, 16),
        torch.randint(0, 31, (7,), device="cuda"),
        4,
        7,
    )


def _flash_attention_args():
    query = _parameter(8, 4, 64)
    key = _parameter(8, 2, 64)
    value = _parameter(8, 2, 64)
    return query, key, value, [3, 5], True, None


def _dsa_args():
    query = _parameter(8, 2, 16)
    key = _parameter(8, 2, 16)
    value = _parameter(8, 2, 12)
    rows = torch.arange(8, device="cuda", dtype=torch.int32)
    indices = torch.stack((rows, torch.maximum(rows - 1, rows.new_zeros(()))), dim=1)
    return query, key, value, indices, [8]


def _causal_conv_args():
    cumulative = torch.tensor([0, 3, 8], dtype=torch.int64, device="cuda")
    return _parameter(8, 16), _parameter(16, 1, 3), [3, 5], cumulative


def _l2_norm_args():
    return _parameter(8, 2, 16), 1e-6


def _gated_norm_args():
    return _parameter(8, 2, 16), _parameter(8, 2, 16), _parameter(16), 1e-5


def _linear_attention_args():
    empty_cumulative = torch.empty(0, dtype=torch.int64, device="cuda")
    empty_chunks = torch.empty((0, 2), dtype=torch.int64, device="cuda")
    return (
        _parameter(8, 2, 16),
        _parameter(8, 2, 16),
        _parameter(8, 4, 16),
        _parameter(8, 4),
        _parameter(8, 4),
        _parameter(4, dtype=torch.float32),
        _parameter(4, dtype=torch.float32),
        empty_cumulative,
        empty_chunks,
        0.25,
    )


def _moe_args():
    return (
        _parameter(8, 32),
        _parameter(8, 32),
        # Model code supplies ``nn.Linear.weight.T`` rather than a newly
        # contiguous router matrix; fake output strides must match this path.
        _parameter(4, 32).T,
        _parameter(4, 32, 64),
        _parameter(4, 32, 32),
        torch.empty(0, device="cuda", dtype=torch.bfloat16),
        2,
        "softmax_then_topk",
        1,
        1,
        1.0,
        [8],
        "float32",
    )


def _scattermoe_args():
    arguments = _moe_args()
    return (
        *arguments[:5],
        arguments[5],
        arguments[6],
        arguments[7],
        arguments[8],
        arguments[9],
        arguments[10],
        arguments[11],
        arguments[12],
    )


def _moe_prepare_args():
    arguments = _moe_args()
    return (
        arguments[0],
        arguments[2],
        arguments[3],
        arguments[5],
        arguments[6],
        arguments[7],
        arguments[8],
        arguments[9],
        arguments[10],
        arguments[11],
        arguments[12],
    )


def _moe_finish_args():
    arguments = _moe_args()
    prepared = moe_composed._prepare_op(*_moe_prepare_args())
    return (
        prepared[9],
        arguments[4],
        prepared[4],
        prepared[6],
        prepared[7],
        prepared[8],
        arguments[1],
        arguments[6],
    )


OPCHECK_CASES = (
    ("cross_entropy", cross_entropy._forward_op, _cross_entropy_args),
    ("embedding", embedding._forward_op, _embedding_args),
    ("rms_norm_builtin", rms_norm_builtin._forward_op, _rms_norm_args),
    ("rms_norm_liger", rms_norm_liger._forward_op, _rms_norm_args),
    ("layer_norm", layer_norm._forward_op, _layer_norm_args),
    ("rope_table", rope._table_forward_op, _rope_args),
    ("rope_analytic", rope._analytic_forward_op, _rope_args),
    ("partial_rope", partial_rope._forward_op, _partial_rope_args),
    ("swiglu", swiglu._forward_op, _swiglu_args),
    ("swiglu_legacy", swiglu._legacy_forward_op, _swiglu_args),
    ("packed_swiglu", swiglu._packed_forward_op, _packed_swiglu_args),
    (
        "packed_swiglu_legacy",
        swiglu._packed_legacy_forward_op,
        _packed_swiglu_args,
    ),
    ("gelu", gelu._forward_op, _gelu_args),
    ("head", head._forward_op, _head_args),
    ("flash_attention", flash_attention._forward_op, _flash_attention_args),
    ("dsa_attention", dsa._attention_forward_op, _dsa_args),
    ("causal_conv", fla_hybrid._causal_forward_op, _causal_conv_args),
    ("l2_norm", fla_hybrid._l2_forward_op, _l2_norm_args),
    ("gated_rms_norm", fla_hybrid._gated_forward_op, _gated_norm_args),
    ("linear_attention", fla_hybrid._linear_forward_op, _linear_attention_args),
    ("moe", moe._forward_op, _moe_args),
    ("moe_prepare", moe_composed._prepare_op, _moe_prepare_args),
    ("moe_finish", moe_composed._finish_op, _moe_finish_args),
    ("scattermoe", scattermoe._forward_op, _scattermoe_args),
)


def test_custom_op_manifest_covers_dispatch_namespace():
    registered = set(torch.ops.mlops._dir)
    assert registered == (
        set(FORWARD_OPERATORS) | BACKWARD_OPERATORS | set(AUXILIARY_OPERATORS)
    )


def test_dependency_layers_do_not_bypass_semantic_operations():
    forbidden_roots = {
        "reference_models",
        "dataflow",
        "reference_models.prototypes",
    }
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any(
                    node.module.startswith(root) for root in forbidden_roots
                ), path
            elif isinstance(node, ast.Import):
                assert not any(
                    alias.name.startswith(root)
                    for alias in node.names
                    for root in forbidden_roots
                ), path

    for path in (PACKAGE_ROOT / "kernels").glob("*.py"):
        source = path.read_text()
        assert "provider_registry" not in source, path
        assert "torch.library.custom_op" not in source, path


@pytest.mark.parametrize("name", sorted(set(FORWARD_OPERATORS) | BACKWARD_OPERATORS))
def test_registered_operator_schema_is_functional(name):
    schema = getattr(torch.ops.mlops, name).default._schema
    assert all(argument.alias_info is None for argument in schema.arguments)
    assert all(result.alias_info is None for result in schema.returns)
    assert "!" not in str(schema)


@pytest.mark.parametrize(
    "_name,operator,make_args",
    OPCHECK_CASES,
    ids=[case[0] for case in OPCHECK_CASES],
)
def test_custom_op_registration_contract(_name, operator, make_args):
    results = torch.library.opcheck(operator, make_args())
    assert set(results.values()) == {"SUCCESS"}
