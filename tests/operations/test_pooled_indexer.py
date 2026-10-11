from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch

from mlops import pooled_topk
from mlops.dispatch import explain_implementation
from mlops.kernels import pooled_indexer as kernels

from .reference import official


def _scores(q, key, gates, weights, bias, start, stop):
    """One sequence's visible pool scores in plain FP32 PyTorch arithmetic."""
    heads, width = q.shape[1], q.shape[2]
    pool = bias.shape[0]
    n_pools = (stop - start) // pool
    pooled = kernels._pooled_keys(key, gates, bias, start, n_pools)
    scores = (q[start:stop].float() @ pooled.float().T * width**-0.5).relu()
    scores = ((weights[start:stop].float() * heads**-0.5)[:, None] @ scores).squeeze(1)
    position = torch.arange(stop - start, device=q.device)
    visible = (torch.arange(n_pools, device=q.device) + 1) * pool - 1 <= position[
        :, None
    ]
    return scores.masked_fill(~visible, float("-inf"))


def _oracle(q, key, gates, weights, bias, boundaries, top_k):
    """The selection in plain PyTorch: FP32 scores, torch.topk, the tail."""
    pool = bias.shape[0]
    device = q.device
    result = torch.full(
        (q.shape[0], top_k + pool - 1), -1, device=device, dtype=torch.int32
    )
    for start, stop in pairwise(boundaries.tolist()):
        length = stop - start
        rows = torch.arange(length, device=device)
        if length <= top_k:
            keys = torch.arange(start, stop, device=device, dtype=torch.int32)
            result[start:stop, :length] = (
                keys[None]
                .expand(length, -1)
                .masked_fill(keys[None] > rows[:, None] + start, -1)
            )
            continue
        keep = min(top_k // pool, length // pool)
        values, pools = _scores(q, key, gates, weights, bias, start, stop).topk(
            keep, dim=-1
        )
        indices = pools[..., None] * pool + torch.arange(pool, device=device) + start
        indices = indices.masked_fill(~values.isfinite()[..., None], -1)
        result[start:stop, : keep * pool] = indices.flatten(-2).to(torch.int32)
        if pool > 1:
            tail_count = (rows + 1) % pool
            offset = torch.arange(pool - 1, device=device)
            tail = start + (rows + 1 - tail_count)[:, None] + offset
            result[start:stop, keep * pool : keep * pool + pool - 1] = tail.masked_fill(
                offset >= tail_count[:, None], -1
            ).to(torch.int32)
    return result


def _pools(row, start, pool):
    return {(index - start) // pool for index in row.tolist() if index >= 0}


def _assert_cutoff_tie(scores, omitted, extra):
    """Differing selections are complete pools whose scores tie at the cutoff
    within FP32 rounding: the kernel sums the heads in another order."""
    assert len(omitted) == len(extra)
    involved = torch.cat((scores[omitted], scores[extra]))
    assert (
        involved.max() - involved.min()
    ).item() <= 1e-5 * involved.abs().max().item()


def _assert_same_selection(actual, expected, inputs, ends, allowed_rows):
    q, key, gates, weights, bias = inputs
    pool = bias.shape[0]
    assert actual.shape == expected.shape and actual.dtype == torch.int32
    assert torch.equal((actual >= 0).sum(1), (expected >= 0).sum(1))
    differing = (
        (actual.sort(1).values != expected.sort(1).values).any(1).nonzero().flatten()
    )
    assert differing.numel() <= allowed_rows
    scores = {}
    for row in differing.tolist():
        start, stop = next((a, b) for a, b in pairwise(ends) if a <= row < b)
        if start not in scores:
            scores[start] = _scores(q, key, gates, weights, bias, start, stop)
        a, b = _pools(expected[row], start, pool), _pools(actual[row], start, pool)
        _assert_cutoff_tie(scores[start][row - start], sorted(a - b), sorted(b - a))


@pytest.mark.parametrize(
    "lengths,top_k", [((3, 19), 32), ((3, 19, 41), 16), ((87,), 8), ((64, 69), 32)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_pooled_indexer_matches_official(lengths, top_k, dtype, compiled):
    (Indexer,) = official("Glm5NextTextIndexer")
    cfg = SimpleNamespace(
        hidden_size=32,
        index_n_heads=4,
        index_head_dim=16,
        qk_rope_head_dim=0,
        index_topk=top_k,
        q_lora_rank=16,
        index_kpool=4,
        index_kpool_always_select_tail=True,
    )
    model = Indexer(cfg, 0).to(device="cuda", dtype=dtype)
    torch.manual_seed(61)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.2)
    x = torch.randn(sum(lengths), 32, device="cuda", dtype=dtype)
    latent = torch.randn(sum(lengths), 16, device="cuda", dtype=dtype)
    with torch.no_grad():
        q = model.wq_b(latent).reshape(-1, 4, 16)
        key = model.k_norm(model.wk(x))
        gates = torch.nn.functional.linear(x, model.index_kpool_compress_gate)
        weights = model.weights_proj(x)
        ends = [0]
        for n in lengths:
            ends.append(ends[-1] + n)
        boundaries = torch.tensor(ends, dtype=torch.int64)
        fn = torch.compile(pooled_topk, fullgraph=True) if compiled else pooled_topk
        got = fn(
            q,
            key,
            gates,
            weights,
            model.index_kpool_compress_ape,
            boundaries,
            top_k=top_k,
        )
        for start, stop in pairwise(ends):
            ref = model(
                x[start:stop].unsqueeze(0),
                latent[start:stop].unsqueeze(0),
                torch.ones(1, stop - start, device="cuda", dtype=torch.bool),
                None,
            )[0]
            ref = torch.where(ref >= 0, ref + start, ref)
            actual = got[start:stop]
            # Exact attended sets except tied cutoffs. Compute reference
            # scores independently from the official pooled states.
            packed = torch.cat(
                (
                    key[start:stop],
                    gates[start:stop],
                    torch.ones(stop - start, 1, device="cuda", dtype=dtype),
                ),
                -1,
            )[None]
            pooled, _, _ = model.get_pooled_states(packed)
            scores = (q[start:stop].float() @ pooled[0].float().T * 16**-0.5).relu()
            scores = (
                (weights[start:stop].float() * 4**-0.5)[:, None] @ scores
            ).squeeze(1)
            for row in range(stop - start):
                a = set(actual[row][actual[row] >= 0].tolist())
                b = set(ref[row][ref[row] >= 0].tolist())
                assert len(a) == len(b)
                assert len(a) == int((actual[row] >= 0).sum())
                assert all(start <= index <= start + row for index in a)
                if a != b:
                    omitted = sorted({(index - start) // 4 for index in b - a})
                    extra = sorted({(index - start) // 4 for index in a - b})
                    _assert_cutoff_tie(scores[row], omitted, extra)
    assert not got.requires_grad


def _inputs(lengths, heads=4, width=64, pool=4, dtype=torch.bfloat16, key_dtype=None):
    torch.manual_seed(7)
    tokens = sum(lengths)

    def make(*shape, dtype=dtype):
        return torch.randn(*shape, device="cuda", dtype=dtype)

    ends = [0]
    for n in lengths:
        ends.append(ends[-1] + n)
    return (
        make(tokens, heads, width),
        make(tokens, width, dtype=key_dtype or dtype),
        make(tokens, width),
        make(tokens, heads),
        make(pool, width),
        torch.tensor(ends, dtype=torch.int64),
    ), ends


@pytest.mark.parametrize("lengths,top_k", [((4100, 2600, 9), 256), ((5000,), 512)])
def test_long_sequences_match_fp32_pytorch_however_chunked(lengths, top_k, monkeypatch):
    arguments, ends = _inputs(lengths)
    allowed = 0.01 * sum(lengths)
    selected = pooled_topk(*arguments, top_k=top_k)
    _assert_same_selection(
        selected, _oracle(*arguments, top_k), arguments[:5], ends, allowed
    )
    # Selection is per query row, so a smaller score chunk changes nothing.
    monkeypatch.setattr(kernels, "SCORE_CHUNK_BYTES", 1 << 16)
    assert torch.equal(pooled_topk(*arguments, top_k=top_k), selected)
    # Rows too wide for the select kernel go through torch.topk instead.
    monkeypatch.setattr(kernels, "_SELECT_POOLS_LIMIT", 0)
    _assert_same_selection(
        pooled_topk(*arguments, top_k=top_k), selected, arguments[:5], ends, allowed
    )


@pytest.mark.parametrize(
    "dtype,key_dtype,width",
    [
        (torch.float32, None, 64),
        (torch.bfloat16, None, 48),
        (torch.float16, None, 16),
        (torch.bfloat16, torch.float16, 256),
    ],
)
def test_every_payload_dtype_and_width_is_scored_exactly(dtype, key_dtype, width):
    arguments, ends = _inputs(
        (700, 301), heads=3, width=width, dtype=dtype, key_dtype=key_dtype
    )
    chosen = explain_implementation("pooled_topk", *arguments, top_k=64)
    assert chosen.selected == "builtin.pooled_topk.triton"
    selected = pooled_topk(*arguments, top_k=64)
    _assert_same_selection(
        selected, _oracle(*arguments, 64), arguments[:5], ends, 0.01 * 1001
    )


pytestmark = pytest.mark.gpu
