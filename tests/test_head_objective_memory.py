"""Opt-in wide-vocabulary checks for the bounded head's real CUDA footprint."""

import json
import math
import os
from time import perf_counter

import pytest
import torch

from mlops import head_objective
from mlops.kernels.head import default_head_chunk_size


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MLOPS_LARGE_HEAD_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set MLOPS_LARGE_HEAD_TESTS=1 on a GPU with at least 10 GiB free",
)
@pytest.mark.parametrize(
    "rows, mode, objective",
    [
        (8192, "full", "cross_entropy"),
        (16384, "full", "cross_entropy"),
        (32768, "full", "cross_entropy"),
        (32768, "frozen", "cross_entropy"),
        (32768, "lora", "cross_entropy"),
        (32768, "lora", "clipped_policy"),
    ],
)
def test_wide_head_memory_tracks_saved_gradients_and_chunk_not_full_logits(
    rows, mode, objective
):
    torch.cuda.empty_cache()
    vocab, width, rank = 248320, 4096, 16

    def tensor(shape, need):
        return (
            torch.empty(shape, device="cuda", dtype=torch.bfloat16)
            .normal_(std=0.02)
            .requires_grad_(need)
        )

    x, weight = tensor((rows, width), True), tensor((vocab, width), mode == "full")
    a, b = (
        (tensor((rank, width), True), tensor((vocab, rank), True))
        if mode == "lora"
        else (None, None)
    )
    targets = torch.arange(rows, device="cuda") % vocab
    mask = torch.arange(rows, device="cuda") % 7 != 0
    fields = [mask, torch.full((rows,), 1 / rows, device="cuda")]
    controls = []
    if objective == "clipped_policy":
        fields.extend(
            (
                torch.full((rows,), -math.log(vocab), device="cuda"),
                torch.linspace(-1, 1, rows, device="cuda"),
                torch.full((rows,), -math.log(vocab), device="cuda"),
            )
        )
        controls = [torch.tensor(0.2, device="cuda"), torch.tensor(0.05, device="cuda")]
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    loss, values = head_objective(
        x,
        weight,
        targets,
        objective=objective,
        inputs=fields,
        controls=controls,
        lora_a=a,
        lora_b=b,
    )
    seeds = loss.grad_fn.saved_tensors
    assert (seeds[1] is None) == (mode != "full")
    seed_bytes = sum(
        value.numel() * value.element_size() for value in seeds if value is not None
    )
    selected = tuple(
        value
        for value in (x, weight, a, b)
        if value is not None and value.requires_grad
    )
    gradients = torch.autograd.grad(loss, selected)
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    extra = torch.cuda.max_memory_allocated() - baseline
    chunk_bytes = default_head_chunk_size(vocab) * vocab * x.element_size()
    # Includes simultaneous saved/output gradients, LoRA temporary projections,
    # row metrics and slack for library workspace. Full logits are not allowed.
    limit = 2 * seed_bytes + 4 * chunk_bytes + (64 << 20)
    assert extra < limit, (extra, limit)
    assert torch.isfinite(loss) and torch.isfinite(values).all()
    for gradient in gradients:
        assert torch.isfinite(gradient.reshape(-1)[::4096]).all()
    if objective == "cross_entropy":
        expected = math.log(vocab) * mask.float().mean()
        torch.testing.assert_close(loss, expected, atol=0.02, rtol=0.02)
    print(
        json.dumps(
            {
                "rows": rows,
                "vocab": vocab,
                "width": width,
                "mode": mode,
                "objective": objective,
                "extra_peak_bytes": extra,
                "saved_gradient_bytes": seed_bytes,
                "chunk_logit_bytes": chunk_bytes,
                "bound_bytes": limit,
                "seconds": seconds,
            }
        ),
        flush=True,
    )
