"""Chunked LoRA head math, trainability, memory and capture contracts."""

from __future__ import annotations

import importlib
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from mlops import explicit, lora_head_loss
from mlops.dispatch import use_implementation, weight_gradients_at

implementation = importlib.import_module("mlops.providers.builtin.lora_head")


def values(needs=(True, False, True, True), *, dtype=torch.float32, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(176)
    shapes = ((2, 5, 7), (19, 7), (3, 7), (19, 3))
    tensors = tuple(
        (
            torch.randn(shape, generator=generator, dtype=dtype, device=device) * 0.2
        ).requires_grad_(need)
        for shape, need in zip(shapes, needs, strict=True)
    )
    labels = torch.tensor([[2, 1, -100, 5, 12], [0, 18, 4, 7, 1]], device=device)
    return tensors, labels


def reference(hidden, weight, a, b, labels, *, scale, normalizer):
    x = hidden.reshape(-1, hidden.shape[-1])
    logits = x @ weight.T + scale * ((x @ a.to(x.dtype).T) @ b.to(x.dtype).T)
    labels = labels.reshape(-1).long().masked_fill(labels.reshape(-1) < 0, -100)
    return F.cross_entropy(logits.float(), labels, reduction="sum") / normalizer


@pytest.mark.parametrize(
    "needs",
    [
        (True, False, True, True),  # usual frozen-head LoRA
        (False, False, True, True),  # train only the head factors
        (True, True, True, True),  # a tied/full-trained base is legal too
        (True, False, False, False),  # input VJP through frozen factors
        (False, False, True, False),
        (False, False, False, True),
        (False, False, False, False),
    ],
)
@pytest.mark.parametrize("backend", [None, "aot_eager"])
def test_loss_and_requested_gradients(needs, backend):
    tensors, labels = values(needs)
    expected_inputs = tuple(
        x.detach().clone().requires_grad_(x.requires_grad) for x in tensors
    )
    call = partial(lora_head_loss, scale=1.7, chunk_size=3, valid_rows=9)
    if backend:
        torch._dynamo.reset()
        call = torch.compile(call, backend=backend, fullgraph=True)
    loss = call(*tensors, labels)
    expected = reference(*expected_inputs, labels, scale=1.7, normalizer=9)
    torch.testing.assert_close(loss, expected, atol=2e-6, rtol=2e-6)
    if any(needs):
        (loss * -0.37).backward()
        (expected * -0.37).backward()
    for actual, target in zip(tensors, expected_inputs, strict=True):
        if actual.requires_grad:
            torch.testing.assert_close(actual.grad, target.grad, atol=2e-6, rtol=2e-5)
        else:
            assert actual.grad is None


@pytest.mark.parametrize("chunk", [1, 4, 64])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_chunking_reduction_and_pytorch_implementation(chunk, reduction):
    tensors, labels = values()
    labels[0, 2] = -1  # all negative targets follow head_loss's ignored-label rule
    actual = lora_head_loss(
        *tensors, labels, scale=0.7, chunk_size=chunk, reduction=reduction
    )
    with use_implementation("lora_head_loss", "native_torch.lora_head_loss"):
        expected = lora_head_loss(*tensors, labels, scale=0.7, reduction=reduction)
    torch.testing.assert_close(actual, expected)
    selected = tuple(t for t in tensors if t.requires_grad)
    gradients = torch.autograd.grad(actual, selected)
    expected_gradients = torch.autograd.grad(expected, selected)
    for a, b in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_zero_b_is_base_head_and_adamw_updates_only_factors():
    tensors, labels = values()
    hidden, weight, a, b = tensors
    with torch.no_grad():
        b.zero_()
    before = weight.clone()
    loss = lora_head_loss(*tensors, labels, scale=0.7, chunk_size=3)
    expected = (
        F.cross_entropy(
            (hidden @ weight.T).reshape(-1, 19),
            labels.reshape(-1),
            reduction="sum",
        )
        / labels.numel()
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(a.grad, torch.zeros_like(a))
    assert torch.count_nonzero(b.grad) > 0
    assert weight.grad is None
    optimizer = torch.optim.AdamW((a, b), lr=1e-3)
    optimizer.step()
    assert torch.equal(weight, before)
    assert set(optimizer.state) == {a, b}


def test_no_dense_weight_gradient_is_saved_and_backward_does_not_reproject(monkeypatch):
    tensors, labels = values()
    chunk_rows, saved = [], []
    original = implementation.cross_entropy_fwd_bwd

    def record_logits(logits, *args, **kwargs):
        chunk_rows.append(logits.shape[0])
        return original(logits, *args, **kwargs)

    def pack(value):
        saved.append(value)
        return value

    monkeypatch.setattr(implementation, "cross_entropy_fwd_bwd", record_logits)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        loss = lora_head_loss(*tensors, labels, chunk_size=3)
    assert chunk_rows == [3, 3, 3, 1]
    assert sorted(t.numel() for t in saved) == sorted(
        tensors[i].numel() for i in (0, 2, 3)
    )
    assert sum(t.numel() * t.element_size() for t in saved) == (70 + 21 + 57) * 4
    selected = tuple(t for t in tensors if t.requires_grad)
    first = torch.autograd.grad(loss, selected, loss.new_tensor(0.4), retain_graph=True)
    second = torch.autograd.grad(
        loss, selected, loss.new_tensor(-0.2), retain_graph=True
    )
    assert chunk_rows == [3, 3, 3, 1]
    for x, y in zip(first, second, strict=True):
        torch.testing.assert_close(y, -0.5 * x)


def test_explicit_seeds_and_configured_gradient_dtype():
    tensors, labels = values(dtype=torch.bfloat16)
    loss, *seeds = explicit.lora_head_loss.forward(
        *tensors,
        labels,
        scale=1.7,
        chunk_size=3,
        weight_grad_dtype=torch.float32,
    )
    assert seeds[1] is None
    assert seeds[0].dtype == torch.bfloat16
    assert seeds[2].dtype == seeds[3].dtype == torch.float32
    assert all(t is None or t.grad_fn is None for t in seeds)
    actual = explicit.lora_head_loss.backward(loss.new_tensor(-0.25), *seeds)
    for seed, result in zip(seeds, actual, strict=True):
        if seed is None:
            assert result is None
        else:
            torch.testing.assert_close(result, seed * -0.25)
    with weight_gradients_at(torch.float32):
        semantic = lora_head_loss(*tensors, labels, scale=1.7, chunk_size=3)
    selected = tuple(t for t in tensors if t.requires_grad)
    gradients = torch.autograd.grad(semantic, selected)
    for expected, actual in zip((seeds[0], seeds[2], seeds[3]), gradients, strict=True):
        torch.testing.assert_close(actual, expected.to(actual.dtype))
    assert all(t.grad is None for t in tensors)


def test_factor_storage_dtype_can_differ_from_compute():
    tensors, labels = values(dtype=torch.bfloat16)
    tensors = (
        *tensors[:2],
        *(t.detach().float().requires_grad_() for t in tensors[2:]),
    )
    loss = lora_head_loss(*tensors, labels, scale=0.7, chunk_size=3)
    loss.backward()
    assert tensors[2].grad.dtype == tensors[3].grad.dtype == torch.float32
    assert tensors[1].grad is None


@pytest.mark.parametrize(
    "options",
    [
        {"scale": float("nan")},
        {"scale": float("inf")},
        {"chunk_size": 0},
        {"valid_rows": 0},
        {"valid_rows": 11},
        {"reduction": "none"},
        {"reduction": "sum", "valid_rows": 9},
    ],
)
def test_invalid_policy_is_rejected(options):
    tensors, labels = values()
    with pytest.raises(ValueError):
        lora_head_loss(*tensors, labels, **options)


def test_custom_op_schema_fake_and_autograd_registration():
    tensors, labels = values()
    arguments = (*tensors, labels, 0.7, 3, 10, None, True, False, True, True)
    checks = torch.library.opcheck(implementation._forward_op, arguments)
    assert set(checks.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_cuda_first_order_gradients_and_inductor(dtype, compiled):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is unavailable")
    torch.manual_seed(14)
    needs = (True, False, True, True)
    shapes = ((37, 128), (521, 128), (16, 128), (521, 16))
    tensors = tuple(
        (torch.randn(s, device="cuda", dtype=dtype) * 0.02).requires_grad_(n)
        for s, n in zip(shapes, needs, strict=True)
    )
    expected_inputs = tuple(
        t.detach().clone().requires_grad_(n)
        for t, n in zip(tensors, needs, strict=True)
    )
    labels = torch.randint(0, 521, (37,), device="cuda")
    labels[3] = -100
    call = partial(lora_head_loss, scale=0.7, chunk_size=11)
    if compiled:
        torch._dynamo.reset()
        call = torch.compile(call, fullgraph=True)
    with weight_gradients_at(torch.float32):
        loss = call(*tensors, labels)
    expected = reference(*expected_inputs, labels, scale=0.7, normalizer=37)
    (loss * 0.6).backward()
    (expected * 0.6).backward()
    tolerance = {
        torch.float32: (2e-6, 2e-4),
        torch.float16: (2e-5, 1e-2),
        torch.bfloat16: (2e-4, 6e-2),
    }[dtype]
    torch.testing.assert_close(loss, expected, atol=1e-4, rtol=5e-3)
    for actual, target in zip(tensors, expected_inputs, strict=True):
        if actual.requires_grad:
            torch.testing.assert_close(
                actual.grad, target.grad, atol=tolerance[0], rtol=tolerance[1]
            )
            relative_l2 = (
                actual.grad.float() - target.grad.float()
            ).norm() / target.grad.float().norm().clamp_min(1e-12)
            bound = {torch.float32: 2e-4, torch.float16: 0.01, torch.bfloat16: 0.06}[
                dtype
            ]
            assert relative_l2.item() < bound, (dtype, relative_l2.item())
        else:
            assert actual.grad is None


def test_noncontiguous_inputs_have_matching_fake_metadata_and_gradients():
    tensors, labels = values()
    tensors = (tensors[0].transpose(0, 1).detach().requires_grad_(), *tensors[1:])
    labels = labels.T
    args = (*tensors, labels, 0.7, 3, 10, None, True, False, True, True)
    checks = torch.library.opcheck(implementation._forward_op, args)
    assert set(checks.values()) == {"SUCCESS"}
    actual = lora_head_loss(*tensors, labels, scale=0.7, chunk_size=3)
    expected = reference(*tensors, labels, scale=0.7, normalizer=10)
    selected = tuple(t for t in tensors if t.requires_grad)
    for x, y in zip(
        torch.autograd.grad(actual, selected),
        torch.autograd.grad(expected, selected),
        strict=True,
    ):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
