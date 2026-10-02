"""LoRA reference gradients, frozen weights, packing and CPU configuration checks."""

import unittest

import torch
from torch.nn import functional as F

from mlops.expert_parallel.lora import (
    LoRAConfig,
    packed_pitch,
    parse_signature,
    signature,
)
from mlops.expert_parallel.reference.lora import expert_computation_lora


class LoRAReferenceTests(unittest.TestCase):
    def test_separate_branches_match_merged_weight_gradients(self):
        torch.manual_seed(3)
        for interleaved in (False, True):
            x = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
            p = torch.rand(7, 2, dtype=torch.float64, requires_grad=True)
            ids = torch.tensor([[0, 1], [2, 0], [0, 2], [1, 0], [1, 2], [0, 1], [2, 0]])
            w1, w2 = (
                torch.randn(4, 10, 8, dtype=torch.float64),
                torch.randn(4, 8, 5, dtype=torch.float64),
            )
            factors = [
                torch.randn(shape, dtype=torch.float64, requires_grad=True)
                for shape in ((4, 3, 8), (4, 10, 3), (4, 3, 5), (4, 8, 3))
            ]
            actual = expert_computation_lora(
                x,
                ids,
                p,
                w1,
                w2,
                *factors,
                scale=0.7,
                interleaved=interleaved,
                compute_dtype=torch.float64,
            )
            a1, b1, a2, b2 = factors
            merged1, merged2 = w1 + 0.7 * b1 @ a1, w2 + 0.7 * b2 @ a2
            expected = torch.zeros_like(x)
            for e in range(4):
                t, s = (ids == e).nonzero(as_tuple=True)
                pre = F.linear(x[t], merged1[e])
                g, u = (pre[:, 0::2], pre[:, 1::2]) if interleaved else pre.chunk(2, -1)
                expected = expected.index_add(
                    0, t, F.linear(F.silu(g) * u, merged2[e]) * p[t, s, None]
                )
            dy = torch.randn_like(x)
            torch.testing.assert_close(actual, expected)
            ga = torch.autograd.grad(actual, (x, p, *factors), dy, retain_graph=True)
            gb = torch.autograd.grad(expected, (x, p, *factors), dy)
            for left, right in zip(ga, gb):
                torch.testing.assert_close(left, right)
            self.assertIsNone(w1.grad)
            self.assertTrue(all(torch.count_nonzero(g[3]) == 0 for g in ga[2:]))

    def test_zero_b_still_trains_b(self):
        torch.manual_seed(7)
        x = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
        ids = torch.zeros((5, 1), dtype=torch.int64)
        weights = [torch.randn(s, dtype=torch.float64) for s in ((1, 10, 8), (1, 8, 5))]
        factors = [
            torch.randn(s, dtype=torch.float64, requires_grad=True)
            for s in ((1, 3, 8), (1, 10, 3), (1, 3, 5), (1, 8, 3))
        ]
        with torch.no_grad():
            factors[1].zero_()
            factors[3].zero_()
        out = expert_computation_lora(
            x,
            ids,
            torch.ones(5, 1, dtype=torch.float64),
            *weights,
            *factors,
            compute_dtype=torch.float64,
        )
        grads = torch.autograd.grad(out.sum(), (x, *factors))
        self.assertGreater(torch.count_nonzero(grads[0]), 0)
        for index in (1, 3):
            self.assertEqual(torch.count_nonzero(grads[index]), 0)
        for index in (2, 4):
            self.assertGreater(torch.count_nonzero(grads[index]), 0)

    def test_config_and_packed_alignment(self):
        config = LoRAConfig(rank=48, alpha=24)
        self.assertEqual(parse_signature(signature("base", config)), ("base", config))
        self.assertEqual(config.scale, 0.5)
        for rank in (0, 15, 31, True):
            with self.assertRaises(ValueError):
                LoRAConfig(rank=rank)
        for q in (1, 4, 32):
            for width in (32 * (7168 + 4096), 32 * (2048 + 7168)):
                pitch = packed_pitch(
                    width,
                    element_size=2,
                    experts=2 * q,
                    granularity=2**21,
                    tile_elements=8192,
                )
                self.assertGreaterEqual(pitch, width)
                self.assertEqual(pitch * 2 * 2 * q % (2**21), 0)
                self.assertEqual(pitch % 8192, 0)


if __name__ == "__main__":
    unittest.main()
