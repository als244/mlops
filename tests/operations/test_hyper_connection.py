from types import SimpleNamespace

import pytest
import torch

from mlops import (
    mhc_coefficients,
    mhc_combine,
    mhc_normalize_streams,
)

from .reference import official


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("parameter_dtype", [torch.float32, torch.bfloat16])
def test_hyper_connection(dtype, compiled, parameter_dtype):
    _, HC = official("Glm5NextTextUnweightedRMSNorm", "Glm5NextTextHyperConnection")
    cfg = SimpleNamespace(
        hc_mult=4, hidden_size=32, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-5
    )
    model = HC(cfg).cuda().to(parameter_dtype)
    torch.manual_seed(29)
    with torch.no_grad():
        model.fn.normal_(std=0.05)
        model.base.normal_(std=0.1)
        model.scale.fill_(0.1)
    x = torch.randn(2, 7, 4, 32, dtype=dtype, device="cuda", requires_grad=True)
    ours_x = x.detach().clone().requires_grad_()
    params = [
        p.detach().clone().requires_grad_() for p in (model.fn, model.base, model.scale)
    ]
    ref = model(x)

    def composition(x, projection, bias, scale):
        with torch.autocast(device_type="cuda", enabled=False):
            projected = torch.nn.functional.linear(
                mhc_normalize_streams(x), projection.float()
            )
        return mhc_coefficients(x, projected, bias, scale)

    fn = torch.compile(composition, fullgraph=True) if compiled else composition
    got = fn(ours_x, *params)
    upstream = [torch.randn_like(value) for value in ref]
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
    ref_grads = torch.autograd.grad(
        ref, (x, model.fn, model.base, model.scale), upstream
    )
    got_grads = torch.autograd.grad(got, (ours_x, *params), upstream)
    for a, b in zip(got_grads, ref_grads):
        torch.testing.assert_close(
            a,
            b,
            rtol=0.02 if dtype == torch.bfloat16 else 2e-4,
            atol=0.003 if dtype == torch.bfloat16 else 2e-5,
        )
    branch = torch.randn(2, 7, 32, dtype=dtype, device="cuda")
    expected = (
        ref[0].to(dtype).unsqueeze(-1) * branch.unsqueeze(-2)
        + ref[1].to(dtype).transpose(-1, -2) @ x
    )
    torch.testing.assert_close(mhc_combine(branch, x, *ref[:2]), expected)


pytestmark = pytest.mark.gpu
