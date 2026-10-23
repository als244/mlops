"""Contracts for the SM90 pooled-indexer provider."""

import pytest
import torch

from mlops import pooled_topk
from mlops.dispatch import explain_implementation, use_implementation
from mlops.kernels import pooled_indexer_cudnn as cudnn_kernel

from .test_pooled_indexer import _assert_same_selection, _inputs, _oracle

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("compiled", [False, True])
def test_sm90_priority_preserves_scores_and_packed_selection(
    weight_dtype, compiled, monkeypatch
):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 provider")
    pytest.importorskip("cudnn")
    args, ends = _inputs((1025, 777, 9), heads=32, width=128)
    args = (*args[:3], args[3].to(weight_dtype), *args[4:])
    if weight_dtype == torch.float32:
        # Exercise FP32 weights that cannot be represented exactly in BF16.
        args[3].add_(torch.randn_like(args[3]) * 0.01)
    assert (
        explain_implementation("pooled_topk", *args, top_k=128).selected
        == "cudnn.pooled_topk.sm90"
    )
    fn = torch.compile(pooled_topk, fullgraph=True) if compiled else pooled_topk
    selected = fn(*args, top_k=128)
    _assert_same_selection(
        selected, _oracle(*args, 128), args[:5], ends, 0.01 * sum((1025, 777, 9))
    )
    # Force multiple query chunks and exercise nonzero causal offsets.
    monkeypatch.setattr(cudnn_kernel, "WORKSPACE_BYTES", 1 << 17)
    repeated = fn(*args, top_k=128)
    assert torch.equal(selected.sort(-1).values, repeated.sort(-1).values)
    assert not selected.requires_grad


def test_sm90_scores_keep_fp32_head_weights():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 provider")
    torch.manual_seed(81)
    q = torch.randn(1, 64, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1, 64, 32, device="cuda", dtype=torch.float32)
    offset = torch.tensor([64], device="cuda", dtype=torch.int32)
    got = cudnn_kernel._score(q, k, w, offset, 4)
    ref = torch.einsum("bqhd,bkd->bqhk", q.float(), k.squeeze(2).float()).relu()
    ref = (ref * w[..., None]).sum(2) / (32 * 128) ** 0.5
    visible = torch.arange(32, device="cuda")[None] < (
        torch.arange(65, 129, device="cuda")[:, None] // 4
    )
    ref = ref.squeeze(0).masked_fill(~visible, -torch.inf)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=2e-6)


def test_sm90_explicit_fallback_override():
    args, _ = _inputs((73,), heads=32, width=128)
    with use_implementation("pooled_topk", "builtin.pooled_topk.triton"):
        assert (
            explain_implementation("pooled_topk", *args, top_k=32).selected
            == "builtin.pooled_topk.triton"
        )
        assert pooled_topk(*args, top_k=32).shape == (73, 35)
