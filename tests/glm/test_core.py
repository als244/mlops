from types import SimpleNamespace

import pytest
import torch

from mlops.glm.activation import clipped_swiglu
from mlops.glm.hyper_connection import coefficients, combine, normalize_streams
from mlops.glm.routing import route

from .reference import official


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("limit", [0.4, 10.0])
def test_clipped_activation(dtype, compiled, limit):
    torch.manual_seed(18)
    x = (torch.randn(37, 66, device="cuda", dtype=dtype) * 12).requires_grad_()
    with torch.no_grad():
        x[0, :6] = torch.tensor([-10, 10, -11, 11, 0, 1], device="cuda", dtype=dtype)
    with torch.no_grad():
        x[0, :6] *= limit / 10
    ref = x.detach().clone().requires_grad_()
    fn = torch.compile(clipped_swiglu, fullgraph=True) if compiled else clipped_swiglu
    got = fn(x, limit)
    a, b = ref.chunk(2, -1)
    expected = torch.nn.functional.silu(a.clamp(max=limit)) * b.clamp(-limit, limit)
    dy = torch.randn_like(got)
    (actual_grad,) = torch.autograd.grad(got, x, dy)
    (ref_grad,) = torch.autograd.grad(expected, ref, dy)
    tol = 0.01 if dtype == torch.bfloat16 else 0.002 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(got, expected, rtol=tol, atol=tol)
    torch.testing.assert_close(actual_grad, ref_grad, rtol=tol, atol=tol)


@pytest.mark.parametrize("groups,selected_groups", [(1, 1), (4, 2)])
@pytest.mark.parametrize("normalize", [False, True])
def test_router_matches_official(groups, selected_groups, normalize):
    (Router,) = official("Glm5NextTextTopkRouter")
    cfg = SimpleNamespace(
        num_experts_per_tok=4,
        num_local_experts=16,
        hidden_size=32,
        routed_scaling_factor=2.5,
        n_group=groups,
        topk_group=selected_groups,
        norm_topk_prob=normalize,
    )
    model = Router(cfg).cuda()
    torch.manual_seed(22)
    with torch.no_grad():
        model.weight.normal_(std=0.2)
        model.e_score_correction_bias.normal_(std=0.1)
    x = torch.randn(19, 32, device="cuda", requires_grad=True)
    logits, weights, ids = model(x)
    ours_ids, ours_weights = route(
        logits,
        model.e_score_correction_bias,
        4,
        groups=groups,
        selected_groups=selected_groups,
        normalize=normalize,
    )
    # Selection ordering can differ while the weighted expert assignment is identical.
    target = torch.zeros_like(logits).scatter(-1, ids, weights)
    got = torch.zeros_like(logits).scatter(-1, ours_ids, ours_weights)
    torch.testing.assert_close(got, target)
    dy = torch.randn_like(got)
    got_grads = torch.autograd.grad(got, (x, model.weight), dy, retain_graph=True)
    ref_grads = torch.autograd.grad(target, (x, model.weight), dy)
    for a, b in zip(got_grads, ref_grads):
        torch.testing.assert_close(a, b)


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
                normalize_streams(x), projection.float()
            )
        return coefficients(x, projected, bias, scale)

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
    torch.testing.assert_close(combine(branch, x, *ref[:2]), expected)


pytestmark = pytest.mark.gpu
