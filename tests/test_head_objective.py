"""One-pass head losses preserve arbitrary row cotangents and saved gradients."""

import importlib
import math

import pytest
import torch
from torch.nn import functional as F

from mlops import head_objective
from mlops.dispatch import weight_gradients_at


def weighted(logprobs, inputs, controls):
    valid, weights = inputs
    weight = torch.where(valid, weights, 0.0)
    return -(torch.where(valid, logprobs, 0.0) * weight).sum(), -weight


def policy(logprobs, inputs, controls):
    valid, weights, old, advantages = inputs[:4]
    current = torch.where(valid, logprobs, 0.0)
    old = torch.where(valid, old, 0.0)
    advantages = torch.where(valid, advantages, 0.0)
    weights = torch.where(valid, weights, 0.0)
    ratio = (current - old).exp()
    clip = controls[0]
    surrogate = torch.minimum(
        ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages
    )
    clipped = torch.where(advantages >= 0, ratio > 1 + clip, ratio < 1 - clip)
    term = -surrogate
    gradient = torch.where(clipped, 0.0, -ratio * advantages)
    if len(inputs) == 5:
        reference = torch.where(valid, inputs[4], 0.0)
        delta, beta = reference - current, controls[1]
        term = term + beta * (delta.expm1() - delta)
        gradient = gradient - beta * delta.expm1()
    loss, gradient = (weights * term).sum(), weights * gradient
    return loss, torch.where(valid, gradient, 0.0)


def _case(device="cpu", dtype=torch.float32, mode="full", objective="weighted"):
    torch.manual_seed(704)
    x = (torch.randn(2, 7, 16, device=device, dtype=dtype) / 4).requires_grad_()
    w = (torch.randn(31, 16, device=device, dtype=dtype) / 4).requires_grad_(
        mode == "full"
    )
    a = (
        (torch.randn(3, 16, device=device, dtype=dtype) / 4).requires_grad_()
        if mode == "lora"
        else None
    )
    b = (
        (torch.randn(31, 3, device=device, dtype=dtype) / 4).requires_grad_()
        if mode == "lora"
        else None
    )
    target = torch.randint(0, 31, (2, 7), device=device)
    target[0, 0] = -100
    valid = torch.rand(2, 7, device=device) > 0.2
    weights = torch.randn(2, 7, device=device)
    weights[~valid] = float("nan")
    inputs, controls = [valid, weights], []
    if objective in ("policy", "clipped_policy", "clipped_policy_no_kl"):
        inputs += [
            torch.full((2, 7), -3.4, device=device),
            torch.randn(2, 7, device=device),
            torch.full((2, 7), -3.2, device=device),
        ]
        controls = [torch.tensor(0.2, device=device), torch.tensor(0.3, device=device)]
        if objective == "clipped_policy_no_kl":
            inputs.pop()
            controls.pop()
    return x, w, a, b, target, inputs, controls


CASES = (
    "weighted",
    "policy",
    "cross_entropy",
    "clipped_policy",
    "clipped_policy_no_kl",
)


def _functions(name):
    if name == "cross_entropy":
        return name, weighted
    if name.startswith("clipped_policy"):
        return "clipped_policy", policy
    return globals()[name], globals()[name]


@pytest.mark.parametrize("mode", ["full", "frozen", "lora"])
@pytest.mark.parametrize("objective", CASES)
@pytest.mark.parametrize("chunk", [1, 3, 100])
def test_saved_vjps_match_independent_full_projection(mode, objective, chunk):
    x, w, a, b, target, inputs, controls = _case(mode=mode, objective=objective)
    selected, reference = _functions(objective)
    actual, values = head_objective(
        x,
        w,
        target,
        objective=selected,
        inputs=inputs,
        controls=controls,
        lora_a=a,
        lora_b=b,
        scale=0.7,
        chunk_size=chunk,
    )
    logits = F.linear(x, w)
    if a is not None:
        logits = logits + 0.7 * F.linear(F.linear(x, a), b)
    expected_values = -F.cross_entropy(
        logits.flatten(0, 1), target.flatten(), reduction="none"
    ).view_as(target)
    expected = reference(expected_values, inputs, controls)[0]
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(values, expected_values)
    assert not values.requires_grad
    parameters = tuple(t for t in (x, w, a, b) if t is not None and t.requires_grad)
    for scale in (0.6, -0.3):
        grads = torch.autograd.grad(actual * scale, parameters, retain_graph=True)
        reference = torch.autograd.grad(expected * scale, parameters, retain_graph=True)
        for current, other in zip(grads, reference, strict=True):
            torch.testing.assert_close(current, other, atol=3e-6, rtol=1e-5)


def test_forward_projects_each_chunk_once_and_backward_only_consumes_seeds(monkeypatch):
    module = importlib.import_module("mlops.providers.builtin.head_objective")
    original, calls = module._logprobs, []

    def counted(logits, targets):
        calls.append(tuple(logits.shape))
        return original(logits, targets)

    monkeypatch.setattr(module, "_logprobs", counted)
    x, w, _a, _b, target, inputs, controls = _case()
    loss, _ = head_objective(
        x,
        w,
        target,
        objective=weighted,
        inputs=inputs,
        controls=controls,
        chunk_size=3,
    )
    saved = [
        tuple(value.shape) for value in loss.grad_fn.saved_tensors if value is not None
    ]
    assert len(calls) == math.ceil(target.numel() / 3)
    assert max(shape[0] for shape in calls) <= 3
    count = len(calls)
    loss.backward()
    assert len(calls) == count
    # Only the hidden and parameter unit VJPs survive the head's forward.
    assert saved == [tuple(x.shape), tuple(w.shape)]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["full", "frozen", "lora"])
@pytest.mark.parametrize("objective", CASES)
def test_cuda_against_compiled_reference(dtype, mode, objective):
    torch._dynamo.reset()  # Independent dtype/trainability capture cases.
    x, w, a, b, target, inputs, controls = _case("cuda", dtype, mode, objective)

    selected, callback = _functions(objective)

    def run(x, w, a, b):
        return head_objective(
            x,
            w,
            target,
            objective=selected,
            inputs=inputs,
            controls=controls,
            lora_a=a,
            lora_b=b,
            scale=0.7,
            chunk_size=3,
        )

    def oracle(x, w, a, b):
        logits = F.linear(x, w)
        if a is not None:
            logits = logits + 0.7 * F.linear(F.linear(x, a), b)
        values = -F.cross_entropy(
            logits.flatten(0, 1).float(), target.flatten(), reduction="none"
        ).view_as(target)
        return callback(values, inputs, controls)[0], values.detach()

    reference = torch.compile(oracle, fullgraph=True)(x, w, a, b)
    actual = torch.compile(run, fullgraph=True)(x, w, a, b)
    tolerance = (
        {"atol": 3e-5, "rtol": 3e-5}
        if dtype == torch.float32
        else {"atol": 0.02, "rtol": 0.02}
    )
    for value, other in zip(actual, reference, strict=True):
        torch.testing.assert_close(value, other, **tolerance)
    parameters = tuple(t for t in (x, w, a, b) if t is not None and t.requires_grad)
    gradients = torch.autograd.grad(actual[0], parameters)
    expected = torch.autograd.grad(reference[0], parameters)
    for value, other in zip(gradients, expected, strict=True):
        torch.testing.assert_close(value, other, **tolerance)


@pytest.mark.parametrize(
    "needs",
    [
        (True, False, True, True),
        (False, False, True, True),
        (True, True, True, True),
        (True, False, False, False),
        (False, False, True, False),
        (False, False, False, True),
        (False, False, False, False),
    ],
)
def test_only_requested_gradient_buffers_are_saved(needs):
    x, w, a, b, target, fields, _ = _case(mode="lora")
    tensors = (x, w, a, b)
    for value, need in zip(tensors, needs, strict=True):
        value.requires_grad_(need)
    loss, logprobs = head_objective(
        x,
        w,
        target,
        objective="cross_entropy",
        inputs=fields,
        lora_a=a,
        lora_b=b,
        chunk_size=3,
    )
    assert not logprobs.requires_grad
    if not any(needs):
        assert not loss.requires_grad
        return
    saved = loss.grad_fn.saved_tensors
    assert tuple(value is not None for value in saved) == needs
    for seed, value, need in zip(saved, tensors, needs, strict=True):
        if need:
            assert seed.shape == value.shape
            assert not seed.requires_grad
    loss.backward()
    assert tuple(value.grad is not None for value in tensors) == needs


@pytest.mark.parametrize("transposed", [False, True])
def test_schema_fake_autograd_and_dynamic_capture(transposed):
    implementation = importlib.import_module("mlops.providers.builtin.head_objective")
    x, w, a, b, target, fields, controls = _case(mode="lora", objective="policy")
    if transposed:
        x = x.transpose(0, 1).detach().requires_grad_()
        target = target.T
        fields = [value.T for value in fields]
    arguments = (
        x,
        w,
        target,
        a,
        b,
        0.7,
        "clipped_policy",
        fields,
        controls,
        3,
        None,
        [True, False, True, True],
    )
    checks = torch.library.opcheck(implementation._forward_op, arguments)
    assert set(checks.values()) == {"SUCCESS"}


def test_parameter_seed_dtype_and_factor_storage_are_independent():
    x, w, a, b, target, fields, _ = _case(mode="lora", dtype=torch.bfloat16)
    a, b = (value.detach().float().requires_grad_() for value in (a, b))
    with weight_gradients_at(torch.float32):
        loss, _ = head_objective(
            x,
            w,
            target,
            objective="cross_entropy",
            inputs=fields,
            lora_a=a,
            lora_b=b,
            chunk_size=3,
        )
    dx, dw, da, db = loss.grad_fn.saved_tensors
    assert dx.dtype == torch.bfloat16 and dw is None
    assert da.dtype == db.dtype == torch.float32
    loss.backward()
    assert a.grad.dtype == b.grad.dtype == torch.float32 and w.grad is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("objective", ["cross_entropy", "clipped_policy"])
def test_builtin_cuda_uses_only_fused_chunk_formulas(monkeypatch, objective):
    implementation = importlib.import_module("mlops.providers.builtin.head_objective")
    original, chunks = implementation.fused_head_objective, []

    def record(logits, *args, **kwargs):
        chunks.append(logits.shape)
        return original(logits, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("built-in GPU objective used a Python formula")

    monkeypatch.setattr(implementation, "fused_head_objective", record)
    monkeypatch.setattr(implementation, "evaluate", forbidden)
    monkeypatch.setattr(implementation, "_logprobs", forbidden)
    x, w, a, b, target, fields, controls = _case(
        "cuda", mode="lora", objective=objective
    )
    loss, _ = head_objective(
        x,
        w,
        target,
        objective=objective,
        inputs=fields,
        controls=controls,
        lora_a=a,
        lora_b=b,
        chunk_size=3,
    )
    expected = [(3, 31)] * 4 + [(2, 31)]
    assert chunks == expected
    loss.backward()
    assert chunks == expected
    assert w.grad is None
