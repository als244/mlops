from types import SimpleNamespace

import pytest
import torch

from mlops import (
    sigmoid_topk,
)

from .reference import official


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
    ours_ids, ours_weights = sigmoid_topk(
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


pytestmark = pytest.mark.gpu
