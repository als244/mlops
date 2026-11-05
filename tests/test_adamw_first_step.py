"""First Adam updates normalize their moments using the declared beta values."""

import pytest
import torch

from mlops.optim import functional_adamw, functional_master_adamw


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("betas", [(0.9, 0.999), (0.9, 0.9999), (0.8, 0.95)])
@pytest.mark.parametrize("counter_device", ["cpu", "cuda"])
@pytest.mark.parametrize("master", [False, True])
def test_first_update_matches_independent_closed_form(betas, counter_device, master):
    # A zero initial parameter/moments gives -lr*g/(abs(g)+eps), regardless
    # of beta. Cover signs, zero, magnitudes near epsilon and a partial tile.
    magnitude = torch.logspace(-6, 2, 2049, dtype=torch.float32)
    cpu_gradient = torch.cat((-magnitude.flip(0), torch.zeros(1), magnitude))
    gradient = cpu_gradient.cuda()
    parameter = torch.zeros_like(gradient)
    mean, variance = torch.zeros_like(gradient), torch.zeros_like(gradient)
    counter = torch.tensor(0, dtype=torch.int64, device=counter_device)
    lr, eps = 1e-4, 1e-8
    arguments = (parameter, gradient, mean, variance, counter)
    if master:
        arguments = (parameter, parameter.clone(), gradient, mean, variance, counter)
    call = functional_master_adamw if master else functional_adamw
    outputs = call(*arguments, lr=lr, betas=betas, eps=eps, weight_decay=0.01)
    actual = outputs[1 if master else 0].cpu()
    expected = (-torch.tensor(lr, dtype=torch.float32).double() * cpu_gradient.double()
                / (cpu_gradient.double().abs() + eps)).float()
    # A few FP32 rounding units accommodate the independent sqrt/division
    # path; the former systematic ~2e-5 relative scale error is much larger.
    torch.testing.assert_close(actual, expected, rtol=4e-7,
                               atol=2 * torch.finfo(torch.float32).eps * lr)
    assert int(outputs[-1]) == 1 and int(counter) == 0
    assert not parameter.count_nonzero() and not mean.count_nonzero() and not variance.count_nonzero()
    if master:
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
