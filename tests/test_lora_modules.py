"""Independent dense/grouped math, trainability, aliasing and gradient demand."""

import importlib
from itertools import product

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mlops import head_loss
from mlops.lora import LoRAConfig, apply_lora, parameter_report
from mlops.lora.grouped import grouped_linear


@pytest.mark.parametrize("compiled", [False, True])
def test_dense_matches_merged_weight_gradients(compiled):
    torch.manual_seed(603)
    layer = apply_lora(nn.Linear(19, 13), LoRAConfig(5, 7), targets=[""])
    with torch.no_grad():
        layer.lora_b.normal_(std=0.07)
    x = torch.randn(11, 19, requires_grad=True)
    call = (
        torch.compile(layer, fullgraph=True, backend="aot_eager") if compiled else layer
    )
    y = call(x)
    oracle = F.linear(
        x, layer.weight + (7 / 5) * layer.lora_b @ layer.lora_a, layer.bias
    )
    seed = torch.randn_like(y)
    actual = torch.autograd.grad(
        y, (x, layer.lora_a, layer.lora_b), seed, retain_graph=True
    )
    expected = torch.autograd.grad(oracle, (x, layer.lora_a, layer.lora_b), seed)
    torch.testing.assert_close(y, oracle)
    torch.testing.assert_close(actual, expected)
    assert not layer.weight.requires_grad and not layer.bias.requires_grad


def test_aliases_original_parameters_and_selection_errors():
    model = nn.Module()
    model.first = nn.Linear(9, 7)
    model.alias = model.first
    original = model.first.weight
    apply_lora(
        model, LoRAConfig(3, 3), targets=["first"], trainable_base=["alias.bias"]
    )
    assert model.first is model.alias and model.first.weight is original
    assert model.first.bias.requires_grad and not original.requires_grad
    assert parameter_report(model)["trainable_parameters"] == 3 * 9 + 7 * 3 + 7
    with pytest.raises(ValueError, match="already"):
        apply_lora(model, LoRAConfig(), targets=["first"])
    fresh = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
    with pytest.raises(TypeError, match="no LoRA conversion"):
        apply_lora(fresh, LoRAConfig(), targets=["0", "1"])
    assert isinstance(fresh[0], nn.Linear) and fresh[0].weight.requires_grad
    with pytest.raises(ValueError, match="matched no"):
        apply_lora(fresh, LoRAConfig(), targets=["missing"])


@pytest.mark.parametrize("need_x,need_weight", list(product([False, True], repeat=2)))
def test_grouped_projection_skips_frozen_weight_work(need_x, need_weight, monkeypatch):
    module = importlib.import_module("mlops.lora.grouped")
    torch.manual_seed(735)
    x = torch.randn(13, 9, requires_grad=need_x)
    w = torch.randn(4, 9, 7, requires_grad=need_weight)
    offsets = torch.tensor(
        [0, 3, 3, 11, 13], dtype=torch.int32
    )  # includes empty expert
    calls = []
    real = module.grouped_mm_wgrad

    def wgrad(*args):
        calls.append(args[3])
        return real(*args)

    monkeypatch.setattr(module, "grouped_mm_wgrad", wgrad)
    y = grouped_linear(x, w, offsets)
    expected = torch.cat(
        [x[int(offsets[i]) : int(offsets[i + 1])] @ w[i] for i in range(4)]
    )
    torch.testing.assert_close(y, expected)
    if need_x or need_weight:
        selected = [t for t in (x, w) if t.requires_grad]
        seed = torch.randn_like(y)
        actual = torch.autograd.grad(y, selected, seed)
        target = torch.autograd.grad(expected, selected, seed)
        torch.testing.assert_close(actual, target)
    assert len(calls) == int(need_weight)


@pytest.mark.parametrize(
    "need_hidden,need_head", list(product([False, True], repeat=2))
)
def test_frozen_head_saves_only_requested_vjp_seeds(need_hidden, need_head):
    torch.manual_seed(173)
    x = torch.randn(7, 11, requires_grad=need_hidden)
    weight = torch.randn(29, 11, requires_grad=need_head)
    targets = torch.randint(29, (7,))
    saved = []

    def pack(t):
        saved.append(tuple(t.shape))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        actual = head_loss(x, weight, targets, chunk_size=3)
    expected = F.cross_entropy(x @ weight.T, targets)
    torch.testing.assert_close(actual, expected)
    assert ((29, 11) in saved) == need_head
    assert ((7, 11) in saved) == need_hidden
    if need_hidden or need_head:
        selected = [t for t in (x, weight) if t.requires_grad]
        torch.testing.assert_close(
            torch.autograd.grad(actual, selected),
            torch.autograd.grad(expected, selected),
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(17, 9), (768, 32), (32, 1024), (768, 1024)])
def test_grouped_cuda_projections_and_gradients(dtype, shape):
    torch.manual_seed(186)
    d, h = shape
    x = (torch.randn(173, d, device="cuda", dtype=dtype) * 0.1).requires_grad_()
    w = (torch.randn(4, d, h, device="cuda", dtype=dtype) * 0.1).requires_grad_()
    offsets = torch.tensor([0, 57, 57, 129, 173], device="cuda", dtype=torch.int32)
    y = grouped_linear(x, w, offsets)
    ref = torch.cat(
        [
            x[a:b] @ w[i]
            for i, (a, b) in enumerate([(0, 57), (57, 57), (57, 129), (129, 173)])
        ]
    )
    seed = torch.randn_like(y) * 0.1
    grads = torch.autograd.grad(y, (x, w), seed)
    expected = torch.autograd.grad(ref, (x, w), seed)
    tolerance = (
        {"atol": 2e-3, "rtol": 2e-2}
        if dtype == torch.bfloat16
        else {"atol": 2e-4, "rtol": 2e-3}
    )
    torch.testing.assert_close(y, ref, **tolerance)
    torch.testing.assert_close(grads, expected, **tolerance)
