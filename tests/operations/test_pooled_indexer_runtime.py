"""Packed sequence contents may change without recompilation or recapture."""

import pytest
import torch

from mlops import pooled_topk
from mlops.kernels import pooled_indexer as kernels

from .test_pooled_indexer import _assert_same_selection, _inputs, _oracle


@pytest.mark.parametrize(
    "lengths,top_k,pool",
    [
        ((3, 19), 32, 4),
        ((0, 3, 19, 0, 41, 0), 16, 4),
        ((0, 91, 0, 4, 33), 8, 4),
        ((70, 31), 16, 1),
        ((2, 3, 1, 2, 3), 4, 4),  # No complete pools, although T > top_k.
        ((137, 5, 160), 64, 16),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_runtime_boundaries_match_reference(lengths, top_k, pool, dtype):
    args, ends = _inputs(lengths, width=48, pool=pool, dtype=dtype)
    actual = pooled_topk(*args[:5], args[-1].cuda(), top_k=top_k)
    _assert_same_selection(
        actual, _oracle(*args, top_k), args[:5], ends, 0.01 * sum(lengths)
    )


@pytest.mark.parametrize("pool", [1, 4])
def test_runtime_scoring_is_chunk_independent_and_supports_wide_rows(pool, monkeypatch):
    args, ends = _inputs((700, 0, 301), pool=pool)
    runtime = (*args[:5], args[-1].cuda())
    expected = _oracle(*args, 64)
    actual = pooled_topk(*runtime, top_k=64)
    _assert_same_selection(actual, expected, args[:5], ends, 10)
    monkeypatch.setattr(kernels, "SCORE_CHUNK_BYTES", 1 << 16)
    assert torch.equal(actual, pooled_topk(*runtime, top_k=64))
    monkeypatch.setattr(kernels, "_SELECT_POOLS_LIMIT", 0)
    _assert_same_selection(
        pooled_topk(*runtime, top_k=64), expected, args[:5], ends, 10
    )


def test_compiled_selection_uses_changed_boundary_contents():
    from torch._dynamo.testing import CompileCounterWithBackend

    args, _ = _inputs((0, 52, 204, 0), width=16)
    boundaries = args[-1].cuda()
    counter = CompileCounterWithBackend("inductor")
    fn = torch.compile(pooled_topk, backend=counter, fullgraph=True)
    first = fn(*args[:5], boundaries, top_k=16)
    changed = torch.tensor([0, 9, 9, 173, 256], dtype=boundaries.dtype)
    boundaries.copy_(changed)
    second = fn(*args[:5], boundaries, top_k=16)
    _assert_same_selection(first, _oracle(*args, 16), args[:5], args[-1].tolist(), 2)
    _assert_same_selection(
        second, _oracle(*args[:5], changed, 16), args[:5], changed.tolist(), 2
    )
    assert not torch.equal(first, second)
    assert counter.frame_count == 1


def test_cuda_graph_replay_uses_changed_boundary_contents():
    args, _ = _inputs((0, 52, 204, 0), width=16)
    boundaries = args[-1].cuda()
    runtime = (*args[:5], boundaries)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            pooled_topk(*runtime, top_k=16)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = pooled_topk(*runtime, top_k=16)
    for ends in ([0, 0, 52, 256, 256], [0, 9, 9, 173, 256], [0, 256, 256, 256, 256]):
        host = torch.tensor(ends, dtype=boundaries.dtype)
        boundaries.copy_(host)
        graph.replay()
        _assert_same_selection(output, _oracle(*args[:5], host, 16), args[:5], ends, 2)


pytestmark = pytest.mark.gpu
