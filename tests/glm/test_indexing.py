from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch

from mlops.glm.indexing import pooled_topk

from .reference import official


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
            query_chunk=7,
            key_chunk=3,
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
            # Exact attended sets except mathematically tied cutoffs. Compute
            # reference scores independently from the official pooled states.
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
                    # Only complete pools can differ, and only at equal scores.
                    assert len(omitted) == len(extra)
                    missing_scores = scores[row, omitted]
                    extra_scores = scores[row, extra]
                    torch.testing.assert_close(
                        extra_scores, missing_scores, rtol=0, atol=0
                    )
                    assert torch.all(missing_scores == missing_scores[0])
    assert not got.requires_grad


pytestmark = pytest.mark.gpu
