"""Correctness and one-pass execution tests for bounded-logits head loss."""

from __future__ import annotations

import importlib
import math

import torch
import torch.nn.functional as functional

from mlops import explicit, head_loss
from mlops.dispatch import use_implementation


def _clone_parameters(*values):
    return tuple(value.detach().clone().requires_grad_() for value in values)


def test_chunked_head_computes_each_logits_chunk_once(monkeypatch):
    implementation = importlib.import_module("mlops.providers.builtin.head")
    calls = 0
    original = implementation.cross_entropy_fwd_bwd

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(implementation, "cross_entropy_fwd_bwd", counted)
    torch.manual_seed(9101)
    hidden, weight = _clone_parameters(
        torch.randn(11, 16),
        torch.randn(23, 16),
    )
    targets = torch.randint(0, 23, (11,))

    loss = head_loss(hidden, weight, targets, chunk_size=4)
    assert calls == math.ceil(targets.numel() / 4)
    calls_after_forward = calls
    loss.backward()
    assert calls == calls_after_forward


def test_chunked_head_matches_full_logits_forward_and_gradients():
    torch.manual_seed(9102)
    raw = _clone_parameters(torch.randn(9, 12), torch.randn(19, 12))
    chunked = _clone_parameters(*raw)
    targets = torch.randint(0, 19, (9,))
    raw_loss = functional.cross_entropy(raw[0] @ raw[1].T, targets)
    chunked_loss = head_loss(*chunked, targets, chunk_size=4)
    scale = 0.61
    (raw_loss * scale).backward()
    (chunked_loss * scale).backward()

    torch.testing.assert_close(chunked_loss, raw_loss)
    for raw_parameter, chunked_parameter in zip(raw, chunked, strict=True):
        torch.testing.assert_close(
            chunked_parameter.grad,
            raw_parameter.grad,
            atol=2e-6,
            rtol=2e-6,
        )


def test_native_torch_head_is_an_independent_full_logits_implementation():
    torch.manual_seed(9103)
    hidden = torch.randn(7, 12, requires_grad=True)
    weight = torch.randn(19, 12, requires_grad=True)
    targets = torch.randint(0, 19, (7,))
    with use_implementation("head_loss", "native_torch.head_loss"):
        actual = head_loss(hidden, weight, targets)
    expected = functional.cross_entropy(hidden @ weight.T, targets)
    actual_gradients = torch.autograd.grad(actual, (hidden, weight))
    expected_gradients = torch.autograd.grad(expected, (hidden, weight))
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_chunked_head_saved_gradients_are_immutable_across_repeated_vjps():
    torch.manual_seed(9104)
    hidden = torch.randn(8, 12, requires_grad=True)
    weight = torch.randn(17, 12, requires_grad=True)
    targets = torch.randint(0, 17, (8,))
    loss = head_loss(hidden, weight, targets, chunk_size=3)
    grad_output = loss.new_tensor(-0.29)
    first = torch.autograd.grad(
        loss,
        (hidden, weight),
        grad_outputs=grad_output,
        retain_graph=True,
    )
    second = torch.autograd.grad(
        loss,
        (hidden, weight),
        grad_outputs=grad_output,
        retain_graph=True,
    )
    for first_gradient, second_gradient in zip(first, second, strict=True):
        torch.testing.assert_close(second_gradient, first_gradient)


def test_head_valid_rows_has_provider_invariant_loss_and_gradients():
    torch.manual_seed(9105)
    targets = torch.randint(0, 17, (8,))
    chunked = _clone_parameters(torch.randn(8, 12), torch.randn(17, 12))
    native = _clone_parameters(*chunked)
    chunked_loss = head_loss(*chunked, targets, chunk_size=3, valid_rows=6)
    with use_implementation("head_loss", "native_torch.head_loss"):
        native_loss = head_loss(*native, targets, valid_rows=6)
    chunked_gradients = torch.autograd.grad(chunked_loss, chunked)
    native_gradients = torch.autograd.grad(native_loss, native)
    torch.testing.assert_close(chunked_loss, native_loss)
    for chunked_gradient, native_gradient in zip(
        chunked_gradients, native_gradients, strict=True
    ):
        torch.testing.assert_close(
            chunked_gradient,
            native_gradient,
            atol=2e-6,
            rtol=2e-6,
        )


def test_explicit_head_matches_semantic_vjp_without_autograd_state():
    torch.manual_seed(9106)
    hidden = torch.randn(8, 12, requires_grad=True)
    weight = torch.randn(17, 12, requires_grad=True)
    targets = torch.randint(0, 17, (8,))
    loss = head_loss(hidden, weight, targets, chunk_size=3)
    cotangent = loss.new_tensor(0.37)
    expected = torch.autograd.grad(loss, (hidden, weight), cotangent)
    explicit_loss, grad_hidden_seed, grad_weight_seed = explicit.head_loss.forward(
        hidden,
        weight,
        targets,
        chunk_size=3,
    )
    actual = explicit.head_loss.backward(
        cotangent,
        grad_hidden_seed,
        grad_weight_seed,
    )
    torch.testing.assert_close(explicit_loss, loss.detach())
    for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert actual_gradient.grad_fn is None
    assert hidden.grad is None and weight.grad is None
