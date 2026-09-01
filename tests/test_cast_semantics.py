"""Focused regressions for semantic operation cast/decision contracts."""

from __future__ import annotations

import pytest
import torch

from mlops.flash_attention import flash_attention
from mlops.embedding import embedding
from mlops.head import head_loss
from mlops.mla import mla_attention
from mlops.l2_norm import l2_norm
from mlops.rope import rope
from mlops.rms_norm import rms_norm
from mlops.kernels.layer_norm import (
    layer_norm_backward,
    layer_norm_forward,
)
from mlops.kernels.moe_router import (
    route,
    routing_override,
)
from mlops.kernels.rms_norm import (
    rms_norm_backward,
    rms_norm_forward,
)
from mlops.kernels.swiglu import swiglu_backward
from mlops.dispatch import use_implementation


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_float = left.detach().float().reshape(-1)
    right_float = right.detach().float().reshape(-1)
    dot = (left_float * right_float).sum(dtype=torch.float64)
    left_norm = left_float.square().sum(dtype=torch.float64).sqrt()
    right_norm = right_float.square().sum(dtype=torch.float64).sqrt()
    denominator = left_norm * right_norm
    if float(denominator) == 0.0:
        return 1.0 if float(left_norm + right_norm) == 0.0 else 0.0
    return float((dot / denominator).clamp(-1.0, 1.0))


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    left_float = left.detach().float()
    right_float = right.detach().float()
    difference = (left_float - right_float).square().sum(dtype=torch.float64).sqrt()
    denominator = torch.maximum(
        left_float.square().sum(dtype=torch.float64).sqrt(),
        right_float.square().sum(dtype=torch.float64).sqrt(),
    ).clamp_min(torch.finfo(torch.float64).tiny)
    return float(difference / denominator)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def test_rmsnorm_triton_casts_match_raw_parameter_gradient():
    torch.manual_seed(8101)
    x = torch.randn(80, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn_like(x)
    x_float = x.float()
    rstd = torch.rsqrt(x_float.square().mean(-1, keepdim=True) + 1e-5)
    output = ((x_float * rstd).to(x.dtype) * weight).to(x.dtype)
    output.backward(grad)

    with torch.no_grad(), use_implementation(
        "rms_norm", "builtin.rms_norm.triton"
    ):
        _optimized, saved_rstd = rms_norm_forward(x.detach(), weight.detach(), 1e-5)
        grad_x, grad_weight = rms_norm_backward(
            grad, x.detach(), weight.detach(), saved_rstd
        )
    assert torch.equal(weight.grad, grad_weight)
    assert _cosine_similarity(x.grad, grad_x) > 0.99999999


def test_layernorm_triton_casts_match_raw_affine_gradients():
    torch.manual_seed(8102)
    x = torch.randn(80, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn_like(x)
    x_float = x.float()
    mean = x_float.mean(-1, keepdim=True)
    rstd = torch.rsqrt((x_float - mean).square().mean(-1, keepdim=True) + 1e-5)
    normalized = ((x_float - mean) * rstd).to(x.dtype)
    output = ((normalized * weight).to(x.dtype) + bias).to(x.dtype)
    output.backward(grad)

    with torch.no_grad(), use_implementation(
        "layer_norm", "builtin.layer_norm.triton"
    ):
        _optimized, saved_mean, saved_rstd = layer_norm_forward(
            x.detach(), weight.detach(), bias.detach(), 1e-5
        )
        grad_x, grad_weight, grad_bias = layer_norm_backward(
            grad, x.detach(), weight.detach(), saved_mean, saved_rstd
        )
    assert torch.equal(weight.grad, grad_weight)
    assert torch.equal(bias.grad, grad_bias)
    assert _cosine_similarity(x.grad, grad_x) > 0.99999999


def test_swiglu_triton_backward_is_bit_identical_to_raw_autograd():
    torch.manual_seed(8103)
    gate = torch.randn(
        80, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    up = torch.randn_like(gate, requires_grad=True)
    grad = torch.randn_like(gate)
    output = torch.nn.functional.silu(gate.float()).to(gate.dtype) * up
    output.backward(grad)
    with use_implementation("swiglu", "builtin.swiglu.triton"):
        grad_gate, grad_up = swiglu_backward(
            grad,
            gate.detach(),
            up.detach(),
            variant="triton",
        )
    assert torch.equal(gate.grad, grad_gate)
    assert torch.equal(up.grad, grad_up)


def test_native_torch_l2_norm_backward_uses_precast_normalized_value():
    torch.manual_seed(8110)
    raw_x = torch.randn(
        257, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    optimized_x = raw_x.detach().clone().requires_grad_(True)
    raw_float = raw_x.float()
    raw_rstd = torch.rsqrt(raw_float.square().sum(-1, keepdim=True) + 1e-6)
    raw_output = (raw_float * raw_rstd).to(raw_x.dtype)
    with use_implementation("l2_norm", "native_torch.l2_norm"):
        optimized_output = l2_norm(optimized_x)
    assert torch.equal(raw_output, optimized_output)
    gradient = torch.randn_like(raw_output)
    raw_output.backward(gradient)
    optimized_output.backward(gradient)
    assert torch.equal(raw_x.grad, optimized_x.grad)


def test_default_fla_l2_norm_backward_has_documented_bf16_approximation():
    torch.manual_seed(8110)
    raw_x = torch.randn(
        257, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    fla_x = raw_x.detach().clone().requires_grad_(True)
    raw_float = raw_x.float()
    raw_output = (
        raw_float * torch.rsqrt(raw_float.square().sum(-1, keepdim=True) + 1e-6)
    ).to(raw_x.dtype)
    fla_output = l2_norm(fla_x)
    assert torch.equal(raw_output, fla_output)
    gradient = torch.randn_like(raw_output)
    raw_output.backward(gradient)
    fla_output.backward(gradient)
    assert not torch.equal(raw_x.grad, fla_x.grad)
    assert _cosine_similarity(raw_x.grad, fla_x.grad) > 0.999
    assert _relative_l2(raw_x.grad, fla_x.grad) < 0.005


def test_router_uses_fp32_weights_and_accepts_forced_indices():
    torch.manual_seed(8104)
    logits = torch.randn(37, 8, device="cuda", dtype=torch.bfloat16)
    forced = torch.stack(
        (torch.zeros(37, device="cuda"), torch.ones(37, device="cuda")),
        dim=1,
    ).long()
    with routing_override((forced,)):
        weights, ids = route(logits, 2, "softmax_then_topk")
    expected = torch.softmax(logits.float(), -1).gather(1, forced)
    assert weights.dtype == torch.float32
    assert torch.equal(ids.long(), forced)
    torch.testing.assert_close(weights, expected, atol=0.0, rtol=0.0)


def test_reference_sdpa_ablation_matches_packed_raw_attention_and_backward():
    torch.manual_seed(8105)
    lengths = (11, 7, 13)
    tokens = sum(lengths)
    raw_q = torch.randn(
        tokens,
        4,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    raw_k = torch.randn(
        tokens,
        2,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    raw_v = torch.randn_like(raw_k, requires_grad=True)
    optimized_q = raw_q.detach().clone().requires_grad_(True)
    optimized_k = raw_k.detach().clone().requires_grad_(True)
    optimized_v = raw_v.detach().clone().requires_grad_(True)

    mask = torch.full((tokens, tokens), float("-inf"), device="cuda")
    start = 0
    for length in lengths:
        stop = start + length
        mask[start:stop, start:stop] = torch.triu(
            torch.full((length, length), float("-inf"), device="cuda"),
            diagonal=1,
        )
        start = stop
    raw_output = (
        torch.nn.functional.scaled_dot_product_attention(
            raw_q.transpose(0, 1).unsqueeze(0),
            raw_k.repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0),
            raw_v.repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0),
            attn_mask=mask.to(raw_q.dtype),
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    with use_implementation(
        "flash_attention", "native_torch.flash_attention"
    ):
        optimized_output = flash_attention(
            optimized_q, optimized_k, optimized_v, lengths
        )

    assert torch.equal(raw_output, optimized_output)
    grad = torch.randn_like(raw_output)
    raw_output.backward(grad)
    optimized_output.backward(grad)
    assert torch.equal(raw_q.grad, optimized_q.grad)
    assert torch.equal(raw_k.grad, optimized_k.grad)
    assert torch.equal(raw_v.grad, optimized_v.grad)


def test_native_torch_attention_preserves_mla_value_width():
    torch.manual_seed(8106)
    lengths = (9, 6)
    tokens = sum(lengths)
    raw_q = torch.randn(
        tokens,
        4,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    raw_k = torch.randn_like(raw_q, requires_grad=True)
    raw_v = torch.randn(
        tokens,
        4,
        32,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    optimized_q = raw_q.detach().clone().requires_grad_(True)
    optimized_k = raw_k.detach().clone().requires_grad_(True)
    optimized_v = raw_v.detach().clone().requires_grad_(True)
    mask = torch.full((tokens, tokens), float("-inf"), device="cuda")
    start = 0
    for length in lengths:
        stop = start + length
        mask[start:stop, start:stop] = torch.triu(
            torch.full((length, length), float("-inf"), device="cuda"),
            diagonal=1,
        )
        start = stop
    raw_output = (
        torch.nn.functional.scaled_dot_product_attention(
            raw_q.transpose(0, 1).unsqueeze(0),
            raw_k.transpose(0, 1).unsqueeze(0),
            raw_v.transpose(0, 1).unsqueeze(0),
            attn_mask=mask.to(raw_q.dtype),
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    with use_implementation("mla_attention", "native_torch.mla_attention"):
        optimized_output = mla_attention(optimized_q, optimized_k, optimized_v, lengths)
    assert optimized_output.shape[-1] == 32
    assert torch.equal(raw_output, optimized_output)
    grad = torch.randn_like(raw_output)
    raw_output.backward(grad)
    optimized_output.backward(grad)
    assert torch.equal(raw_q.grad, optimized_q.grad)
    assert torch.equal(raw_k.grad, optimized_k.grad)
    assert torch.equal(raw_v.grad, optimized_v.grad)


def test_native_torch_rope_matches_raw_expression_and_autograd():
    torch.manual_seed(8107)
    lengths = (11, 7, 13)
    positions = torch.cat([torch.arange(length, device="cuda") for length in lengths])
    raw_x = torch.randn(
        1,
        sum(lengths),
        4,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    optimized_x = raw_x.detach().clone().requires_grad_(True)
    inverse = 1.0 / (
        500_000.0 ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64)
    )
    frequencies = torch.outer(
        torch.arange(max(lengths), device="cuda", dtype=torch.float32),
        inverse,
    )
    table = torch.cat((frequencies, frequencies), dim=-1)
    cosine_table = table.cos()
    sine_table = table.sin()
    cosine = cosine_table[positions]
    sine = sine_table[positions]
    x_float = raw_x.float()
    rotated = torch.cat((-x_float[..., 32:], x_float[..., :32]), dim=-1)
    raw_output = (
        x_float * cosine[None, :, None, :] + rotated * sine[None, :, None, :]
    ).to(raw_x.dtype)
    with use_implementation("rope", "native_torch.rope"):
        optimized_output = rope(
            optimized_x,
            positions,
            500_000.0,
            cosine_table,
            sine_table,
        )
    assert torch.equal(raw_output, optimized_output)
    grad = torch.randn_like(raw_output)
    raw_output.backward(grad)
    optimized_output.backward(grad)
    assert torch.equal(raw_x.grad, optimized_x.grad)


def test_triton_table_rope_matches_native_torch_for_noncontiguous_mla_slice():
    torch.manual_seed(8111)
    lengths = (11, 7, 13)
    positions = torch.cat([torch.arange(length, device="cuda") for length in lengths])
    parent = torch.randn(
        1,
        sum(lengths),
        4,
        80,
        device="cuda",
        dtype=torch.bfloat16,
    )
    noncontiguous = parent[..., 16:]
    assert not noncontiguous.is_contiguous()
    native_parent = parent.detach().clone().requires_grad_(True)
    triton_parent = parent.detach().clone().requires_grad_(True)
    native_x = native_parent[..., 16:]
    triton_x = triton_parent[..., 16:]
    native_x.retain_grad()
    triton_x.retain_grad()
    assert not native_x.is_contiguous()
    assert not triton_x.is_contiguous()
    gradient = torch.randn_like(native_x)
    inverse = 1.0 / (
        500_000.0
        ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64)
    )
    frequencies = torch.outer(
        torch.arange(max(lengths), device="cuda", dtype=torch.float32),
        inverse,
    )
    table = torch.cat((frequencies, frequencies), dim=-1)
    cosine, sine = table.cos(), table.sin()

    with use_implementation("rope", "native_torch.rope"):
        native_output = rope(
            native_x, positions, 500_000.0, cosine, sine
        )
    with use_implementation("rope", "builtin.rope.triton_table"):
        triton_output = rope(
            triton_x, positions, 500_000.0, cosine, sine
        )

    assert torch.equal(native_output, triton_output)
    native_output.backward(gradient)
    triton_output.backward(gradient)
    assert torch.equal(native_x.grad, triton_x.grad)


def test_native_torch_embedding_matches_repeated_id_accumulation():
    torch.manual_seed(8108)
    tokens = torch.tensor([[7, 2, 7, 7, 11, 2, 7, 11]], device="cuda")
    raw_weight = torch.randn(
        17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    optimized_weight = raw_weight.detach().clone().requires_grad_(True)
    raw_output = torch.nn.functional.embedding(tokens, raw_weight)
    with use_implementation("embedding", "native_torch.embedding"):
        optimized_output = embedding(tokens, optimized_weight)
    assert torch.equal(raw_output, optimized_output)
    grad = torch.randn_like(raw_output)
    raw_output.backward(grad)
    optimized_output.backward(grad)
    assert torch.equal(raw_weight.grad, optimized_weight.grad)


def test_native_torch_head_matches_raw_loss_and_backward():
    torch.manual_seed(8109)
    raw_hidden = torch.randn(
        80, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    raw_norm = torch.randn(256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    raw_head = torch.randn(
        1024, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    targets = torch.randint(0, 1024, (80,), device="cuda")
    optimized_hidden = raw_hidden.detach().clone().requires_grad_(True)
    optimized_norm = raw_norm.detach().clone().requires_grad_(True)
    optimized_head = raw_head.detach().clone().requires_grad_(True)
    hidden_float = raw_hidden.float()
    rstd = torch.rsqrt(hidden_float.square().mean(-1, keepdim=True) + 1e-5)
    normalized = ((hidden_float * rstd).to(raw_hidden.dtype) * raw_norm).to(
        raw_hidden.dtype
    )
    logits = normalized @ raw_head.T
    raw_loss = torch.nn.functional.cross_entropy(logits.float(), targets)
    with use_implementation("rms_norm", "native_torch.rms_norm"), use_implementation(
        "head_loss", "native_torch.head_loss"
    ):
        optimized_normalized = rms_norm(optimized_hidden, optimized_norm)
        optimized_loss = head_loss(
            optimized_normalized,
            optimized_head,
            targets,
            chunk_size=80,
        )
    assert torch.equal(raw_loss, optimized_loss)
    raw_loss.backward()
    optimized_loss.backward()
    assert torch.equal(raw_hidden.grad, optimized_hidden.grad)
    assert torch.equal(raw_norm.grad, optimized_norm.grad)
    assert torch.equal(raw_head.grad, optimized_head.grad)
