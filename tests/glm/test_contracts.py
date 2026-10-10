import pytest
import torch

from mlops.glm import activation, attention, indexing, kda


def test_activation_schema():
    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.library.opcheck(activation._op, (x, 10.0))


def test_sparse_attention_schema():
    q = torch.randn(7, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ids = torch.arange(7, device="cuda", dtype=torch.int32).expand(7, -1).clone()
    ids.masked_fill_(ids > torch.arange(7, device="cuda")[:, None], -1)
    torch.library.opcheck(attention._forward, (q, kv, ids, 0.125))


def test_kda_schema():
    q = torch.randn(13, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = (
        torch.nn.functional.normalize(q.detach().clone().float(), dim=-1)
        .bfloat16()
        .requires_grad_()
    )
    v = torch.randn_like(q, requires_grad=True)
    g = (-torch.rand(q.shape, device="cuda") * 0.2 - 0.01).requires_grad_()
    beta = torch.rand(
        q.shape[:2], device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    boundaries = torch.tensor([0, 13], device="cuda", dtype=torch.int32)
    chunks = torch.tensor([[0, 0]], device="cuda", dtype=torch.int32)
    torch.library.opcheck(kda._forward, (q, k, v, g, beta, boundaries, chunks, 0.125))


def test_cpu_metadata_stays_runtime_data():
    torch.manual_seed(97)
    q = torch.randn(25, 2, 16, device="cuda")
    key = torch.randn(25, 16, device="cuda")
    gates = torch.randn_like(key)
    weights = torch.randn(25, 2, device="cuda")
    ape = torch.randn(4, 16, device="cuda")
    compiled_graphs = []
    from torch._dynamo.backends.registry import lookup_backend

    inductor = lookup_backend("inductor")

    def backend(graph, inputs):
        compiled_graphs.append(graph)
        return inductor(graph, inputs)

    fn = torch.compile(indexing.pooled_topk, backend=backend, fullgraph=True)
    for lengths in ((0, 7, 25), (0, 13, 25), (0, 7, 25)):
        metadata = torch.tensor(lengths, dtype=torch.int64)
        actual = fn(q, key, gates, weights, ape, metadata, top_k=8)
        expected = indexing.pooled_topk(q, key, gates, weights, ape, metadata, top_k=8)
        torch.testing.assert_close(actual, expected)
    assert len(compiled_graphs) == 1


def test_kda_metadata_reuse():
    torch.manual_seed(101)
    values = [
        torch.randn(65, 2, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    values[:2] = [
        torch.nn.functional.normalize(x.float(), dim=-1).bfloat16() for x in values[:2]
    ]
    g = -torch.rand(65, 2, 64, device="cuda") * 0.1 - 0.01
    beta = torch.rand(65, 2, device="cuda", dtype=torch.bfloat16)
    chunks = torch.tensor([[0, 0], [1, 0]], device="cuda", dtype=torch.int32)
    graphs = []
    from torch._dynamo.backends.registry import lookup_backend

    inductor = lookup_backend("inductor")

    def backend(graph, inputs):
        graphs.append(graph)
        return inductor(graph, inputs)

    fn = torch.compile(kda.kimi_delta_attention, backend=backend, fullgraph=True)
    for end in (31, 33, 31):
        cumulative = torch.tensor([0, end, 65], device="cuda", dtype=torch.int32)
        expected = kda.kimi_delta_attention(*values, g, beta, cumulative, chunks)
        actual = fn(*values, g, beta, cumulative, chunks)
        torch.testing.assert_close(actual, expected)
    assert len(graphs) == 1


@pytest.mark.parametrize(
    "invalid",
    [
        "decay_dtype",
        "beta_dtype",
        "boundary_dtype",
        "chunk_dtype",
        "chunk_shape",
        "boundary_stride",
    ],
)
def test_kda_rejects_invalid_contracts(invalid):
    from mlops.glm.kda import kimi_delta_attention

    q = torch.zeros(64, 2, 64, device="cuda", dtype=torch.bfloat16)
    g = torch.zeros_like(q, dtype=torch.float32)
    beta = torch.ones(64, 2, device="cuda", dtype=torch.bfloat16)
    cumulative = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    chunks = torch.tensor([[0, 0]], device="cuda", dtype=torch.int32)
    if invalid == "decay_dtype":
        g = g.bfloat16()
    elif invalid == "beta_dtype":
        beta = beta.int()
    elif invalid == "boundary_dtype":
        cumulative = cumulative.float()
    elif invalid == "chunk_dtype":
        chunks = chunks.float()
    elif invalid == "chunk_shape":
        chunks = chunks.flatten()
    elif invalid == "boundary_stride":
        cumulative = torch.tensor([0, 0, 64, 0], device="cuda", dtype=torch.int32)[::2]
    with pytest.raises(ValueError):
        kimi_delta_attention(q, q, q, g, beta, cumulative, chunks)


pytestmark = pytest.mark.gpu
