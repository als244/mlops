"""Caller-owned LoRA around the same operation, with frozen original matrices."""

import pytest
import torch
from torch.nn import functional as F

from mlops.glm.attention import sparse_latent_attention


def absorbed_lora(query, latent, a, b, frozen, indices):
    heads, dim = frozen.shape[0], query.shape[-1]
    wk, wv = frozen.split(dim, dim=1)
    bk, bv = b.reshape(heads, 2 * dim, -1).split(dim, dim=1)
    latent_q = torch.einsum("thd,hdl->thl", query, wk)
    latent_q = latent_q + torch.einsum("thd,hdr->thr", query, bk) @ a
    mixed = sparse_latent_attention(latent_q, latent, indices, scale=dim**-0.5)
    return torch.einsum("thl,hdl->thd", mixed, wv) + torch.einsum(
        "thr,hdr->thd", mixed @ a.T, bv
    )


def expanded_lora(query, latent, a, b, frozen, indices):
    tokens, heads, dim = query.shape
    projected = F.linear(latent, frozen.flatten(0, 1)) + F.linear(
        F.linear(latent, a), b
    )
    keys, values = projected.reshape(tokens, heads, 2 * dim).chunk(2, -1)
    mask = torch.zeros(tokens, tokens + 1, device=query.device, dtype=torch.bool)
    mask.scatter_(1, torch.where(indices >= 0, indices, tokens).long(), True)
    scores = torch.einsum("thd,shd->hts", query.float(), keys.float()) * dim**-0.5
    probabilities = scores.masked_fill(~mask[:, :-1], float("-inf")).softmax(-1)
    return torch.einsum("hts,shd->thd", probabilities, values.float()).to(query.dtype)


@pytest.mark.parametrize("compiled", [False, True])
def test_mla_with_external_joint_kv_lora(compiled):
    torch.manual_seed(129)
    tokens, heads, dim, width, rank = 13, 16, 32, 64, 8
    frozen = (
        torch.randn(heads, 2 * dim, width, device="cuda", dtype=torch.bfloat16)
        * width**-0.5
    )
    shapes = (
        (tokens, heads, dim),
        (tokens, width),
        (rank, width),
        (heads * 2 * dim, rank),
    )
    args = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for shape in shapes]
    args[2] *= 0.1
    args[3] *= 0.1
    args = [x.requires_grad_() for x in args]
    refs = [x.detach().clone().requires_grad_() for x in args]
    ids = (
        torch.arange(tokens, device="cuda", dtype=torch.int32)
        .expand(tokens, -1)
        .clone()
    )
    ids.masked_fill_(ids > torch.arange(tokens, device="cuda")[:, None], -1)
    fn = torch.compile(absorbed_lora, fullgraph=True) if compiled else absorbed_lora
    output = fn(*args, frozen, ids)
    target = expanded_lora(*refs, frozen, ids)
    dy = torch.randn_like(output)
    gradients = torch.autograd.grad(output, args, dy)
    expected_gradients = torch.autograd.grad(target, refs, dy)
    errors = []
    for actual, expected in zip((output, *gradients), (target, *expected_gradients)):
        error = float(
            (
                (actual.float() - expected.float()).norm() / expected.float().norm()
            ).detach()
        )
        errors.append(error)
        assert error < 0.025, errors
    assert frozen.grad is None and not frozen.requires_grad
    assert all(torch.count_nonzero(grad) > 0 for grad in gradients[2:])
    print("MLA external LoRA output/input/factor gradient errors:", errors, flush=True)


pytestmark = pytest.mark.gpu
