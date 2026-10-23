"""Selecting every expert group must preserve stable routing without ranking groups."""

from unittest.mock import patch

import pytest
import torch

from mlops.kernels.moe_router import route, route_torch


@pytest.mark.parametrize("groups,selected_groups", [(1, 1), (4, 4), (4, 2)])
@pytest.mark.parametrize("tied", [False, True])
def test_all_groups_selection_matches_reference(groups, selected_groups, tied):
    torch.manual_seed(993)
    logits = torch.zeros(7, 288) if tied else torch.randn(7, 288)
    bias = torch.zeros(288) if tied else torch.randn(288) * 0.1
    kwargs = {
        "bias": bias,
        "n_group": groups,
        "topk_group": selected_groups,
        "routed_scaling": 2.5,
    }
    expected = route_torch(logits, 8, "sigmoid_noaux_tc", **kwargs)
    original = torch.sort
    shapes = []

    def count(t, *args, **kw):
        shapes.append(t.shape)
        return original(t, *args, **kw)

    with patch("torch.sort", count):
        actual = route(logits, 8, "sigmoid_noaux_tc", **kwargs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert len(shapes) == (1 if groups == selected_groups else 3)
