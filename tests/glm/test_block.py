"""Attention-block fixture; final model definitions will live in ShadowSpill workloads."""

import copy
from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch

import mlops
from mlops.glm.gates import gated_rms_norm, kda_decay
from mlops.glm.kda import kimi_delta_attention

from .reference import official


def optimized_block(model, hidden, cumulative, chunks):
    heads, width = model.num_heads, model.head_dim
    mixed = torch.cat(
        (model.q_proj(hidden), model.k_proj(hidden), model.v_proj(hidden)), -1
    )
    mixed = mlops.causal_conv_silu(mixed, model.conv1d.weight, cumulative, chunks)
    query, key, value = [t.reshape(-1, heads, width) for t in mixed.chunk(3, -1)]
    query, key = mlops.l2_norm(query), mlops.l2_norm(key)
    raw = model.forget_gate.f_b_proj(model.forget_gate.f_a_proj(hidden)).reshape(
        -1, heads, width
    )
    decay = kda_decay(raw, model.forget_gate.dt_bias, model.forget_gate.A_log)
    beta = model.b_proj(hidden).sigmoid()
    attended = kimi_delta_attention(query, key, value, decay, beta, cumulative, chunks)
    gate = model.g_b_proj(model.g_a_proj(hidden)).reshape_as(attended)
    normalized = gated_rms_norm(
        attended, gate, model.o_norm.weight, eps=model.layer_norm_epsilon
    )
    return model.o_proj(normalized.flatten(-2))


def make_reference():
    names = (
        "l2norm",
        "apply_mask_to_padding_states",
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "recurrent_kimi_delta_attention",
        "chunk_kimi_delta_attention",
        "Glm5NextTextForgetGate",
        "Glm5NextTextRMSNormGated",
        "Glm5NextTextLinearAttention",
    )
    definitions = dict(zip(names, official(*names)))
    cls = definitions["Glm5NextTextLinearAttention"]
    # Use the published FP32 recurrent reference to avoid chunked-reference
    # exponent underflow at the model's -5 decay bound. No optimized dependency.
    cls.forward.__globals__["chunk_kimi_delta_attention"] = definitions[
        "recurrent_kimi_delta_attention"
    ]
    config = SimpleNamespace(
        hidden_size=64,
        linear_num_heads=2,
        linear_head_dim=64,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        linear_lower_bound=-5.0,
        layer_types=["linear_attention"],
    )
    model = cls(config, 0).to(device="cuda", dtype=torch.bfloat16)
    # Match HF _keep_in_fp32_modules_strict, not a blanket model.bfloat16().
    model.conv1d.float()
    model.forget_gate.dt_bias = torch.nn.Parameter(model.forget_gate.dt_bias.float())
    model.forget_gate.A_log = torch.nn.Parameter(model.forget_gate.A_log.float())
    with torch.no_grad():
        model.forget_gate.A_log.zero_()
        # Published inverse-softplus initialization.
        dt = (
            torch.empty_like(model.forget_gate.dt_bias)
            .uniform_(
                torch.log(torch.tensor(1e-3)).item(),
                torch.log(torch.tensor(1e-1)).item(),
            )
            .exp()
            .clamp_min(1e-4)
        )
        model.forget_gate.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
    return model


class LoRALinear(torch.nn.Module):
    """Test-only ordinary linear + low-rank update; no kernel knows about LoRA."""

    def __init__(self, base, rank=4):
        super().__init__()
        self.base = base
        self.a = torch.nn.Parameter(
            base.weight.new_empty(rank, base.in_features).normal_(std=0.1)
        )
        self.b = torch.nn.Parameter(
            base.weight.new_empty(base.out_features, rank).normal_(std=0.1)
        )

    def forward(self, x):
        return self.base(x) + torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.a), self.b
        )


def add_lora(model):
    model.requires_grad_(False)
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, LoRALinear(module))


@pytest.mark.parametrize("stress_decay", [False, True])
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("lengths", [(79,), (31, 67)])
@pytest.mark.parametrize("compiled", [False, True])
def test_kda_block(lengths, compiled, lora, stress_decay):
    torch.manual_seed(111)
    reference = make_reference()
    if stress_decay:
        with torch.no_grad():
            reference.forget_gate.dt_bias.zero_()
    assert reference.conv1d.weight.dtype == torch.float32
    assert reference.forget_gate.dt_bias.dtype == torch.float32
    assert reference.forget_gate.A_log.dtype == torch.float32
    if lora:
        add_lora(reference)
    model = copy.deepcopy(reference)
    x = torch.randn(
        sum(lengths), 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    x_ref = x.detach().clone().requires_grad_()
    ends = [0]
    for n in lengths:
        ends.append(ends[-1] + n)
    cumulative = torch.tensor(ends, device="cuda", dtype=torch.int64)
    chunks = torch.tensor(
        [(i, j) for i, n in enumerate(lengths) for j in range((n + 63) // 64)],
        device="cuda",
        dtype=torch.int64,
    )
    fn = torch.compile(optimized_block, fullgraph=True) if compiled else optimized_block
    actual = fn(model, x, cumulative, chunks)
    expected = torch.cat(
        [
            reference(x_ref[a:b].unsqueeze(0)).squeeze(0)
            for a, b in pairwise(ends)
        ]
    )
    dy = torch.randn_like(actual)
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    ref_params = {
        name: p for name, p in reference.named_parameters() if p.requires_grad
    }
    actual_grads = torch.autograd.grad(actual, (x, *params.values()), dy)
    expected_grads = torch.autograd.grad(expected, (x_ref, *ref_params.values()), dy)
    names = ("output", "input", *params)
    errors = {}
    for name, a, b in zip(names, (actual, *actual_grads), (expected, *expected_grads)):
        error = float(
            ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)).detach()
        )
        errors[name] = error
        assert torch.isfinite(a).all() and error < 0.025, errors
    print("Full GLM KDA block relative L2 by output/parameter:", errors, flush=True)


pytestmark = pytest.mark.gpu
