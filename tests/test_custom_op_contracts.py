"""Dispatcher-contract checks for every opaque training operation."""

import ast
import importlib
from functools import partial
from pathlib import Path

import pytest
import torch

import mlops
from mlops.lora import grouped as lora_grouped
from mlops.lora import routing as lora_routing
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


fla_hybrid = importlib.import_module("mlops.providers.fla.hybrid")
dsa = importlib.import_module("mlops.providers.builtin.dsa")
cross_entropy = importlib.import_module("mlops.providers.builtin.cross_entropy")
embedding = importlib.import_module("mlops.providers.builtin.embedding")
flash_attention = importlib.import_module("mlops.providers.builtin.flash_attention")
gelu = importlib.import_module("mlops.providers.builtin.gelu")
head = importlib.import_module("mlops.providers.builtin.head")
lora_head = importlib.import_module("mlops.providers.builtin.lora_head")
layer_norm = importlib.import_module("mlops.providers.builtin.layer_norm")
moe = importlib.import_module("mlops.providers.builtin.moe")
moe_composed = importlib.import_module("mlops.providers.builtin.moe_composed")
partial_rope = importlib.import_module("mlops.providers.builtin.partial_rope")
rms_norm = importlib.import_module("mlops.rms_norm")
rms_norm_builtin = importlib.import_module("mlops.providers.builtin.rms_norm")
sonicmoe_moe = importlib.import_module("mlops.providers.sonicmoe.moe")
sonicmoe_moe_composed = importlib.import_module("mlops.providers.sonicmoe.moe_composed")
rope = importlib.import_module("mlops.providers.builtin.rope")
swiglu = importlib.import_module("mlops.providers.builtin.swiglu")


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


PACKAGE_ROOT = Path(mlops.__file__).resolve().parent


kda = importlib.import_module("mlops.providers.fla.kda")
sparse_latent = importlib.import_module("mlops.providers.builtin.sparse_latent")
pooled_indexer = importlib.import_module("mlops.providers.builtin.pooled_indexer")

FORWARD_OPERATORS = {
    "kimi_delta_attention_fla_fwd": kda._forward,
    "sparse_latent_attention_tilelang_fwd": sparse_latent._forward,
    "pooled_topk_builtin_triton": pooled_indexer._select,
    "lora_grouped_linear_fwd": lora_grouped._forward,
    "lora_route": lora_routing._route,
    "lora_dispatch": lora_routing._dispatch,
    "lora_combine": lora_routing._combine,
    "causal_conv_silu_fla_fwd": fla_hybrid._causal_forward_op,
    "cross_entropy_builtin_triton_fwd": cross_entropy._forward_op,
    "dsa_attention_builtin_sparse_fwd": dsa._attention_forward_op,
    "embedding_builtin_deterministic_fwd": embedding._forward_op,
    "flash_attention_builtin_aten_fwd": flash_attention._forward_op,
    "gated_rms_norm_fla_fwd": fla_hybrid._gated_forward_op,
    "gelu_builtin_aten_explicit_backward_fwd": gelu._forward_op,
    "head_loss_builtin_chunked_fwd": head._forward_op,
    "lora_head_loss_builtin_chunked_fwd": lora_head._forward_op,
    "l2_norm_fla_fwd": fla_hybrid._l2_forward_op,
    "layer_norm_builtin_triton_fwd": layer_norm._forward_op,
    "linear_attention_fla_gated_delta_rule_fwd": fla_hybrid._linear_forward_op,
    "moe_builtin_grouped_gemm_fwd": moe._forward_op,
    "moe_builtin_grouped_gemm_prepare_fwd": moe_composed._prepare_op,
    "moe_builtin_grouped_gemm_finish_fwd": moe_composed._finish_op,
    "moe_sonicmoe_fwd": sonicmoe_moe._forward_op,
    "moe_sonicmoe_prepare_fwd": sonicmoe_moe_composed._prepare_op,
    "moe_sonicmoe_finish_fwd": sonicmoe_moe_composed._finish_op,
    "partial_rope_builtin_triton_fwd": partial_rope._forward_op,
    "rms_norm_builtin_triton_fwd": rms_norm_builtin._forward_op,
    "rope_builtin_triton_table_fwd": rope._table_forward_op,
    "rope_builtin_triton_analytic_fwd": rope._analytic_forward_op,
    "packed_swiglu_builtin_triton_fwd": swiglu._packed_forward_op,
    "packed_swiglu_builtin_triton_legacy_fwd": swiglu._packed_legacy_forward_op,
    "swiglu_builtin_triton_fwd": swiglu._forward_op,
    "swiglu_builtin_triton_legacy_fwd": swiglu._legacy_forward_op,
}

BACKWARD_OPERATORS = {
    "kimi_delta_attention_fla_bwd",
    "sparse_latent_attention_tilelang_bwd",
    "lora_grouped_linear_bwd",
    "lora_route_bwd",
    "lora_dispatch_bwd",
    "lora_combine_bwd",
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
    "moe_sonicmoe_bwd",
    "moe_sonicmoe_prepare_bwd",
    "moe_sonicmoe_finish_bwd",
    "partial_rope_builtin_triton_bwd",
    "rms_norm_builtin_triton_bwd",
    "rope_builtin_triton_table_bwd",
    "rope_builtin_triton_analytic_bwd",
    "packed_swiglu_builtin_triton_bwd",
    "packed_swiglu_builtin_triton_legacy_bwd",
    "swiglu_builtin_triton_bwd",
    "swiglu_builtin_triton_legacy_bwd",
}

AUXILIARY_OPERATORS = {
    "lora_assignment_layout": lora_routing.layout,
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


def _embedding_args(weight_grad_dtype=None):
    tokens = torch.tensor([[1, 3, 1, 5]], device="cuda")
    return tokens, _parameter(17, 16), weight_grad_dtype


def _rms_norm_args():
    return _parameter(6, 16), _parameter(16), 1e-5


def _builtin_rms_norm_args(weight_grad_dtype=None):
    return (*_rms_norm_args(), weight_grad_dtype)


def _layer_norm_args(weight_grad_dtype=None):
    return _parameter(6, 16), _parameter(16), _parameter(16), 1e-5, weight_grad_dtype


def _rope_args():
    rows, width = 6, 16
    inverse = 1.0 / (
        10_000.0
        ** (torch.arange(0, width, 2, device="cuda", dtype=torch.float32) / width)
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
        ** (torch.arange(0, width, 2, device="cuda", dtype=torch.float32) / width)
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


def _lora_head_args(weight_grad_dtype=None):
    hidden = _parameter(7, 16)
    head_weight = _parameter(31, 16).requires_grad_(False)
    a, b = _parameter(3, 16), _parameter(31, 3)
    targets = torch.randint(0, 31, (7,), device="cuda")
    return (
        hidden,
        head_weight,
        a,
        b,
        targets,
        0.7,
        4,
        7,
        weight_grad_dtype,
        True,
        False,
        True,
        True,
    )


def _head_args(weight_grad_dtype=None):
    return (
        _parameter(7, 16),
        _parameter(31, 16),
        torch.randint(0, 31, (7,), device="cuda"),
        4,
        7,
        weight_grad_dtype,
    )


def _flash_attention_args():
    query = _parameter(8, 4, 64)
    key = _parameter(8, 2, 64)
    value = _parameter(8, 2, 64)
    cu_seqlens = torch.tensor([0, 3, 8, 8], dtype=torch.int32, device="cuda")
    return query, key, value, cu_seqlens, 5, True, None, False


def _dsa_args():
    query = _parameter(8, 2, 16)
    key = _parameter(8, 2, 16)
    value = _parameter(8, 2, 12)
    rows = torch.arange(8, device="cuda", dtype=torch.int32)
    indices = torch.stack((rows, torch.maximum(rows - 1, rows.new_zeros(()))), dim=1)
    return query, key, value, indices, [8]


def _causal_conv_args():
    cumulative = torch.tensor([0, 3, 8], dtype=torch.int64, device="cuda")
    return _parameter(8, 16), _parameter(16, 1, 3), cumulative


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


def _moe_args(weight_grad_dtype=None):
    return (
        _parameter(8, 32),
        _parameter(8, 32),
        # nn.Linear orientation, with an expert width that is neither the
        # model width nor half of it, so an orientation slip changes a shape.
        _parameter(4, 32),
        _parameter(4, 48, 32),
        _parameter(4, 32, 24),
        torch.empty(0, device="cuda", dtype=torch.bfloat16),
        2,
        "softmax_then_topk",
        1,
        1,
        1.0,
        [8],
        "float32",
        weight_grad_dtype,
    )


def _moe_prepare_args(weight_grad_dtype=None):
    arguments = _moe_args(weight_grad_dtype)
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
        arguments[13],
    )


def _moe_finish_args(weight_grad_dtype=None):
    arguments = _moe_args(weight_grad_dtype)
    prepared = moe_composed._prepare_op(*_moe_prepare_args(weight_grad_dtype))
    return (
        prepared[9],
        arguments[4],
        prepared[4],
        prepared[6],
        prepared[7],
        prepared[8],
        arguments[1],
        arguments[6],
        weight_grad_dtype,
    )


def _fp8_moe_args(weight_grad_dtype=None):
    """The MoE arguments with every width a multiple of 16 (E4M3 row pitch),
    the expert weights as the operators see FP8 containers: their E4M3 rows
    in place and the other components in the trailing list."""
    from mlops.expert_parallel.quack.parameters.fp8 import QuackFP8Weight

    arguments = list(_moe_args(weight_grad_dtype))
    w13 = QuackFP8Weight.quantize(_parameter(4, 96, 32).detach(), interleaved=True)
    w2 = QuackFP8Weight.quantize(_parameter(4, 32, 48).detach())  # w2 [E, D, F]
    arguments[3], arguments[4] = w13._data, w2._data
    components = [
        w13._transposed,
        w13._scale,
        w13._transposed_scale,
        w2._transposed,
        w2._scale,
        w2._transposed_scale,
    ]
    arguments[13] = weight_grad_dtype or torch.bfloat16  # resolved for FP8 math
    return tuple(arguments), components


def _sonicmoe_moe_args(weight_grad_dtype=None, compute_precision="native"):
    if compute_precision == "native":
        arguments, components = _moe_args(weight_grad_dtype), []
    else:
        arguments, components = _fp8_moe_args(weight_grad_dtype)
    probe = list(arguments[:5])
    if compute_precision != "native":
        probe[3] = _parameter(4, 96, 32)
        probe[4] = _parameter(4, 32, 48)
    if not sonicmoe_moe.supports(
        *probe,
        surface="semantic",
        top_k=arguments[6],
        routing_mode=arguments[7],
        lengths=arguments[11],
        compute_precision=None if compute_precision == "native" else compute_precision,
    ):
        pytest.skip(
            "SonicMoE needs a Hopper or Blackwell GPU with its extras"
            + (" (SM90 for fp8_current)" if compute_precision != "native" else "")
        )
    return (*arguments, compute_precision, components)


def _sonicmoe_prepare_args(weight_grad_dtype=None, compute_precision="native"):
    arguments = _sonicmoe_moe_args(weight_grad_dtype, compute_precision)
    return (
        *(arguments[index] for index in (0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)),
        arguments[15][:3],
    )


def _sonicmoe_finish_args(weight_grad_dtype=None, compute_precision="native"):
    arguments = _sonicmoe_moe_args(weight_grad_dtype, compute_precision)
    prepared = sonicmoe_moe_composed._prepare_op(
        *_sonicmoe_prepare_args(weight_grad_dtype, compute_precision)
    )
    return (
        prepared[10],
        arguments[4],
        prepared[4],
        prepared[6],
        prepared[7],
        prepared[8],
        prepared[9],
        arguments[1],
        arguments[6],
        arguments[13],
        compute_precision,
        arguments[15][3:],
    )


def _sonicmoe_fp8_args(weight_grad_dtype=None):
    return _sonicmoe_moe_args(weight_grad_dtype, "fp8_current")


def _sonicmoe_fp8_prepare_args(weight_grad_dtype=None):
    return _sonicmoe_prepare_args(weight_grad_dtype, "fp8_current")


def _sonicmoe_fp8_finish_args(weight_grad_dtype=None):
    return _sonicmoe_finish_args(weight_grad_dtype, "fp8_current")


OPCHECK_CASES = (
    ("cross_entropy", cross_entropy._forward_op, _cross_entropy_args),
    ("embedding", embedding._forward_op, _embedding_args),
    ("rms_norm_builtin", rms_norm_builtin._forward_op, _builtin_rms_norm_args),
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
    ("lora_head", lora_head._forward_op, _lora_head_args),
    ("flash_attention", flash_attention._forward_op, _flash_attention_args),
    ("dsa_attention", dsa._attention_forward_op, _dsa_args),
    ("causal_conv", fla_hybrid._causal_forward_op, _causal_conv_args),
    ("l2_norm", fla_hybrid._l2_forward_op, _l2_norm_args),
    ("gated_rms_norm", fla_hybrid._gated_forward_op, _gated_norm_args),
    ("linear_attention", fla_hybrid._linear_forward_op, _linear_attention_args),
    ("moe", moe._forward_op, _moe_args),
    ("moe_prepare", moe_composed._prepare_op, _moe_prepare_args),
    ("moe_finish", moe_composed._finish_op, _moe_finish_args),
    ("moe_sonicmoe", sonicmoe_moe._forward_op, _sonicmoe_moe_args),
    ("moe_sonicmoe_prepare", sonicmoe_moe_composed._prepare_op, _sonicmoe_prepare_args),
    ("moe_sonicmoe_finish", sonicmoe_moe_composed._finish_op, _sonicmoe_finish_args),
    ("moe_sonicmoe_fp8", sonicmoe_moe._forward_op, _sonicmoe_fp8_args),
    (
        "moe_sonicmoe_fp8_prepare",
        sonicmoe_moe_composed._prepare_op,
        _sonicmoe_fp8_prepare_args,
    ),
    (
        "moe_sonicmoe_fp8_finish",
        sonicmoe_moe_composed._finish_op,
        _sonicmoe_fp8_finish_args,
    ),
    # The operations that return weight gradients, asked for them at fp32.
    *(
        (f"{name}_fp32_weight_gradients", operator, partial(make, torch.float32))
        for name, operator, make in (
            ("embedding", embedding._forward_op, _embedding_args),
            ("rms_norm_builtin", rms_norm_builtin._forward_op, _builtin_rms_norm_args),
            ("layer_norm", layer_norm._forward_op, _layer_norm_args),
            ("head", head._forward_op, _head_args),
            ("lora_head", lora_head._forward_op, _lora_head_args),
            ("moe", moe._forward_op, _moe_args),
            ("moe_prepare", moe_composed._prepare_op, _moe_prepare_args),
            ("moe_finish", moe_composed._finish_op, _moe_finish_args),
            ("moe_sonicmoe", sonicmoe_moe._forward_op, _sonicmoe_moe_args),
            (
                "moe_sonicmoe_prepare",
                sonicmoe_moe_composed._prepare_op,
                _sonicmoe_prepare_args,
            ),
            (
                "moe_sonicmoe_finish",
                sonicmoe_moe_composed._finish_op,
                _sonicmoe_finish_args,
            ),
            (
            ),
        )
    ),
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
    utilities = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
        "test_aot_dispatch_dynamic",
    ]
    if "fp8" in _name:
        # SchemaCheckMode's mutation check does arithmetic on every input,
        # which E4M3 tensors have no kernels for; the schema itself is the
        # same one the bf16 case checks.
        utilities.remove("test_schema")
    results = torch.library.opcheck(operator, make_args(), test_utils=tuple(utilities))
    assert set(results.values()) == {"SUCCESS"}
