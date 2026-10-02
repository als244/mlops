"""Independent derivation tests; these do not validate CUDA kernel execution."""

import torch
from torch.nn import functional as F


def test_sonic_hidden_space_backward_matches_autograd_with_zero_probabilities():
    torch.manual_seed(82)
    x = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    w1 = torch.randn(8, 5, dtype=torch.float64, requires_grad=True)
    w2 = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    p = torch.tensor(
        [0.0, 0.1, 0.2, 1.0, 0.5, 0.0, 0.9], dtype=torch.float64, requires_grad=True
    )
    dy = torch.randn_like(x)
    pre = F.linear(x, w1)
    g, u = pre[:, 0::2], pre[:, 1::2]
    a = F.silu(g) * u
    y = p[:, None] * F.linear(a, w2)
    expected = torch.autograd.grad(y, (x, w1, w2, p), dy)
    r = dy @ w2
    da = p[:, None] * r
    s = g.sigmoid()
    dpre = torch.stack((da * u * s * (1 + g * (1 - s)), da * g * s), dim=-1).flatten(-2)
    actual = (dpre @ w1, dpre.T @ x, dy.T @ (p[:, None] * a), (r * a).sum(-1))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


def test_prefetch_slot_mapping_preserves_owner_and_expert_for_arbitrary_ep_sizes():
    for ranks, q in ((2, 32), (4, 16), (8, 8)):
        ids = torch.arange(-1, ranks * q, dtype=torch.int32)
        mapped = torch.where(ids >= 0, (ids // q) * (2 * q) + ids % q, -1)
        assert mapped[0] == -1
        assert torch.equal(mapped[1:] // (2 * q), ids[1:] // q)
        assert torch.equal(mapped[1:] % (2 * q), ids[1:] % q)
        assert bool((mapped[1:] % (2 * q) < q).all())


def test_expert_banks_align_the_rank_allocation_without_wasting_per_expert_pages():
    from mlops.expert_parallel.quack.weights import _aligned_rows

    granularity = 2**21
    # 96 experts/rank, D=1024, H=1280: each complete rank bank already aligns.
    for slots, rows, columns, itemsize in (
        (192, 2560, 1024, 2),
        (192, 1024, 1280, 2),
        (96, 2560, 1024, 4),
        (96, 1024, 1280, 4),
    ):
        assert _aligned_rows(slots, rows, columns, itemsize, granularity) == rows
    # Small and odd expert counts can still need padding. Preserve both the
    # virtual-memory allocation requirement and MoonEP's per-expert tile shape.
    for slots in (1, 3, 4, 32, 96, 192):
        for rows, columns in ((128, 128), (1024, 1280), (2560, 1024)):
            for itemsize in (2, 4):
                aligned = _aligned_rows(slots, rows, columns, itemsize, granularity)
                assert aligned >= rows and aligned % 128 == 0
                assert slots * aligned * columns * itemsize % granularity == 0
                if aligned > rows:
                    assert (
                        slots * (aligned - 128) * columns * itemsize % granularity != 0
                    )
