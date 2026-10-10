import pytest
import torch

from mlops.glm.mla import sparse_mla


def expanded(query, latent, wk, wv, ids, scale):
    keys = torch.einsum("tl,hdl->thd", latent, wk)
    values = torch.einsum("tl,hdl->thd", latent, wv)
    mask = torch.zeros(
        query.shape[0], query.shape[0] + 1, device=query.device, dtype=torch.bool
    )
    mask.scatter_(1, torch.where(ids >= 0, ids, query.shape[0]).long(), True)
    scores = torch.einsum("thd,shd->hts", query.float(), keys.float()) * scale
    probability = scores.masked_fill(~mask[:, :-1], float("-inf")).softmax(-1)
    return torch.einsum("hts,shd->thd", probability, values.float()).to(query.dtype)


@pytest.mark.parametrize("heads,dim,latent", [(16, 32, 64), (64, 256, 512)])
@pytest.mark.parametrize("compiled", [False, True])
def test_projection_absorption(heads, dim, latent, compiled):
    torch.manual_seed(91)
    t = 11
    shapes = ((t, heads, dim), (t, latent), (heads, dim, latent), (heads, dim, latent))
    tensors = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for shape in shapes
    ]
    tensors[2] *= latent**-0.5
    tensors[3] *= latent**-0.5
    args = [v.requires_grad_() for v in tensors]
    refs = [v.detach().clone().requires_grad_() for v in tensors]
    ids = torch.arange(t, device="cuda", dtype=torch.int32).expand(t, -1).clone()
    ids = ids.masked_fill(ids > torch.arange(t, device="cuda")[:, None], -1)
    fn = torch.compile(sparse_mla, fullgraph=True) if compiled else sparse_mla
    got = fn(*args, ids, scale=dim**-0.5)
    expected = expanded(*refs, ids, dim**-0.5)
    dy = torch.randn_like(got)
    gradients = torch.autograd.grad(got, args, dy)
    reference_gradients = torch.autograd.grad(expected, refs, dy)
    errors = []
    for actual, target in zip((got, *gradients), (expected, *reference_gradients)):
        err = float(
            ((actual.float() - target.float()).norm() / target.float().norm()).detach()
        )
        errors.append(err)
        assert err < 0.015, errors
    print(
        "Absorbed/expanded MLA output and four gradients relative L2:",
        errors,
        flush=True,
    )


pytestmark = pytest.mark.gpu
