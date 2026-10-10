import pytest
import torch

from mlops.glm.attention import sparse_latent_attention


def reference(q, kv, ids, scale):
    mask = torch.zeros(q.shape[0], q.shape[0] + 1, device=q.device, dtype=torch.bool)
    mask.scatter_(1, torch.where(ids >= 0, ids, q.shape[0]).long(), True)
    logits = torch.einsum("thd,sd->hts", q.float(), kv.float()) * scale
    probabilities = logits.masked_fill(~mask[:, :-1], float("-inf")).softmax(-1)
    return torch.einsum("hts,sd->thd", probabilities, kv.float()).to(q.dtype)


@pytest.mark.parametrize("heads,width", [(16, 64), (64, 512)])
@pytest.mark.parametrize("compiled", [False, True])
def test_attention_forward_backward(heads, width, compiled):
    torch.manual_seed(83)
    tokens = 47
    q = torch.randn(
        tokens, heads, width, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    kv = torch.randn(
        tokens, width, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    qr = q.detach().clone().requires_grad_()
    kr = kv.detach().clone().requires_grad_()
    # Two packed sequences; only every other past key plus the current key.
    ids = torch.full((tokens, 33), -1, device="cuda", dtype=torch.int32)
    for t in range(tokens):
        start = 0 if t < 19 else 19
        selected = torch.arange(start, t, 2, device="cuda")
        selected = torch.cat((selected, torch.tensor([t], device="cuda")))
        ids[t, : selected.numel()] = selected.to(torch.int32)
    fn = (
        torch.compile(sparse_latent_attention, fullgraph=True)
        if compiled
        else sparse_latent_attention
    )
    got = fn(q, kv, ids, scale=256**-0.5)
    target = reference(qr, kr, ids, 256**-0.5)
    dy = torch.randn_like(got)
    gradients = torch.autograd.grad(got, (q, kv), dy)
    expected = torch.autograd.grad(target, (qr, kr), dy)
    errors = []
    for actual, ref in zip((got, *gradients), (target, *expected)):
        error = float(
            ((actual.float() - ref.float()).norm() / ref.float().norm()).detach()
        )
        errors.append(error)
        assert error < 0.012, errors
    print("Sparse MLA output/dQ/dKV relative L2:", errors, flush=True)


pytestmark = pytest.mark.gpu
