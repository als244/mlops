"""A zero-centered RMSNorm scales the FP32 normalized value by 1 + weight
before its one cast, forward and backward, in every implementation."""

import pytest
import torch

import mlops
from mlops import explicit
from mlops.providers.native_torch import rms_norm as native

EPSILON = 1e-6


def reference(x, weight):
    """Qwen 3.5's zero-centered norm, as plain PyTorch computes it."""
    source = x.float()
    scale = torch.rsqrt(source.square().mean(-1, keepdim=True) + EPSILON)
    return (source * scale * (1 + weight.float())).to(x.dtype)


def inputs(dtype, device, rows=7, width=48, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, width, generator=generator).to(device=device, dtype=dtype)
    weight = (0.1 * torch.randn(width, generator=generator)).to(device)
    grad = torch.randn(rows, width, generator=generator).to(device=device, dtype=dtype)
    return x, weight, grad


def autograd_values(function, x, weight, grad):
    x = x.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    output = function(x, weight)
    output.backward(grad)
    return output.detach(), x.grad, weight.grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_native_implementation_is_the_reference_arithmetic(dtype):
    x, weight, grad = inputs(dtype, "cpu")
    expected = autograd_values(reference, x, weight, grad)
    actual = autograd_values(
        lambda a, w: native.apply(a, w, EPSILON, zero_centered=True), x, weight, grad
    )
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_explicit_vjp_is_the_reference_gradient(dtype):
    x, weight, grad = inputs(dtype, "cpu")
    output, grad_x, grad_weight = autograd_values(reference, x, weight, grad)
    forward_output, rstd = native.forward(x, weight, EPSILON, zero_centered=True)
    torch.testing.assert_close(forward_output, output, rtol=0, atol=0)
    explicit_grad_x, explicit_grad_weight = native.backward(
        grad, x, weight, rstd, zero_centered=True
    )
    tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(explicit_grad_x, grad_x, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(explicit_grad_weight, grad_weight, rtol=1e-5, atol=1e-5)


def test_the_native_vjp_passes_gradcheck_in_float64():
    x, weight, _grad = inputs(torch.float64, "cpu", rows=3, width=8)
    torch.autograd.gradcheck(
        lambda a, w: native.apply(a, w, EPSILON, zero_centered=True),
        (x.requires_grad_(True), weight.double().requires_grad_(True)),
    )


def test_zero_centered_is_the_plain_norm_of_one_plus_weight_in_float32():
    # In float32 nothing rounds between the two forms, so they agree.
    x, weight, _grad = inputs(torch.float32, "cpu")
    torch.testing.assert_close(
        native.apply(x, weight, EPSILON, zero_centered=True),
        native.apply(x, 1 + weight, EPSILON),
        rtol=1e-6,
        atol=1e-6,
    )


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_fused_kernels_match_the_reference(dtype):
    x, weight, grad = inputs(dtype, "cuda", rows=33, width=2048 + 16)
    output, grad_x, grad_weight = autograd_values(reference, x, weight, grad)
    fused = autograd_values(
        lambda a, w: mlops.rms_norm(a, w, EPSILON, zero_centered=True), x, weight, grad
    )
    # One rounding of the same FP32 value: at most one unit in the last place.
    output_tolerance = 1e-6 if dtype == torch.float32 else 8e-3
    torch.testing.assert_close(
        fused[0].float(), output.float(), rtol=output_tolerance, atol=1e-6
    )
    gradient_tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(
        fused[1].float(), grad_x.float(), rtol=gradient_tolerance, atol=1e-3
    )
    torch.testing.assert_close(fused[2], grad_weight, rtol=1e-4, atol=1e-4)


@cuda
def test_the_explicit_fused_forward_and_vjp_take_the_mode():
    x, weight, grad = inputs(torch.bfloat16, "cuda", rows=9, width=256)
    output, grad_x, grad_weight = autograd_values(reference, x, weight, grad)
    forward_output, rstd = explicit.rms_norm.forward(
        x, weight, EPSILON, zero_centered=True
    )
    torch.testing.assert_close(
        forward_output.float(), output.float(), rtol=8e-3, atol=1e-6
    )
    explicit_grad_x, explicit_grad_weight = explicit.rms_norm.backward(
        grad, x, weight, rstd, zero_centered=True
    )
    torch.testing.assert_close(
        explicit_grad_x.float(), grad_x.float(), rtol=1e-2, atol=1e-3
    )
    torch.testing.assert_close(explicit_grad_weight, grad_weight, rtol=1e-4, atol=1e-4)


@cuda
def test_the_default_mode_is_unchanged():
    x, weight, grad = inputs(torch.bfloat16, "cuda", rows=9, width=256)
    plain = autograd_values(
        lambda a, w: mlops.rms_norm(a, w, EPSILON), x, 1 + weight, grad
    )
    expected = autograd_values(
        lambda a, w: native.apply(a, w, EPSILON), x, 1 + weight, grad
    )
    torch.testing.assert_close(
        plain[0].float(), expected[0].float(), rtol=8e-3, atol=1e-6
    )
