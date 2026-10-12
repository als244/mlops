from types import SimpleNamespace

import pytest
import torch

from mlops import (
    mhc_coefficients,
    mhc_combine,
    mhc_normalize_streams,
)

from .reference import official

# The streams' dtype, the projection weight's, then the bias and scale's. mHC
# trains with the streams in the activation dtype and every coefficient in FP32.
LAYOUTS = {
    # Full precision holds the implementation to FP32 tolerances throughout.
    "fp32": (torch.float32, torch.float32, torch.float32),
    # DeepSeek-V4 and Megatron-Core keep every mHC parameter in FP32.
    "fp32_parameters": (torch.bfloat16, torch.float32, torch.float32),
    # GLM-5.3-Flash's checkpoint stores only the projection in BF16.
    "glm_checkpoint": (torch.bfloat16, torch.bfloat16, torch.float32),
    # transformers loads every parameter in the model's BF16.
    "bf16": (torch.bfloat16, torch.bfloat16, torch.bfloat16),
}


def assert_close(actual, expected):
    """Hold each tensor to its own precision.

    A BF16 result is rounded from FP32 math, so agreeing FP32 values can still
    round one BF16 ulp apart. The streams' gradient is rounded once per path
    into the streams in eager mode, but once in total where compilation fuses
    the paths, so the two differ by ulps of the gradient's scale rather than of
    each element; FP32 results keep the tight bound.
    """
    if actual.dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.003)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_hyper_connection(layout, compiled):
    dtype, projection_dtype, vector_dtype = LAYOUTS[layout]
    _, HC = official("Glm5NextTextUnweightedRMSNorm", "Glm5NextTextHyperConnection")
    cfg = SimpleNamespace(
        hc_mult=4, hidden_size=32, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-5
    )
    model = HC(cfg).cuda()
    torch.manual_seed(29)
    with torch.no_grad():
        model.fn.normal_(std=0.05)
        model.base.normal_(std=0.1)
        model.scale.fill_(0.1)
    for name, parameter_dtype in (
        ("fn", projection_dtype),
        ("base", vector_dtype),
        ("scale", vector_dtype),
    ):
        value = getattr(model, name).detach().to(parameter_dtype)
        setattr(model, name, torch.nn.Parameter(value))
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
    # Whatever the streams and parameters are, the coefficients are FP32.
    assert got[0].dtype == got[1].dtype == torch.float32
    upstream = [torch.randn_like(value) for value in ref]
    for a, b in zip(got, ref):
        assert_close(a, b)
    ref_grads = torch.autograd.grad(
        ref, (x, model.fn, model.base, model.scale), upstream
    )
    got_grads = torch.autograd.grad(got, (ours_x, *params), upstream)
    for a, b in zip(got_grads, ref_grads):
        assert_close(a, b)
    branch = torch.randn(2, 7, 32, dtype=dtype, device="cuda")
    expected = (
        ref[0].to(dtype).unsqueeze(-1) * branch.unsqueeze(-2)
        + ref[1].to(dtype).transpose(-1, -2) @ x
    )
    torch.testing.assert_close(mhc_combine(branch, x, *ref[:2]), expected)


pytestmark = pytest.mark.gpu
