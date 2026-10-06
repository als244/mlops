"""Check eager and compiled FP8 parameter publication against backend quantizers."""

import argparse
import json
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--backend", choices=["quack", "te", "both"], default="both")
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    torch.manual_seed(183 + rank)
    results = []
    source = torch.randn(256, 512, device="cuda", dtype=torch.float32)
    if args.backend in ("quack", "both"):
        from mlops.expert_parallel.quack.parameters.fp8 import QuackFP8Weight

        scale = (source.abs().amax(-1) / 448).clamp_min(1e-12)
        transposed_scale = (source.t().abs().amax(-1) / 448).clamp_min(1e-12)
        weight = QuackFP8Weight(
            (source / scale[:, None]).to(torch.float8_e4m3fn),
            (source.t() / transposed_scale[:, None]).to(torch.float8_e4m3fn),
            scale,
            transposed_scale,
        )
        cases = [("quack", weight, None)]
    else:
        cases = []
    if args.backend in ("te", "both"):
        from mlops.expert_parallel.transformer_engine.experts import _make_quantizer
        from mlops.expert_parallel.transformer_engine.parameters.publication import (
            weight_quantizer,
        )

        for precision in ("fp8_current", "fp8_block"):
            q = weight_quantizer(precision)
            weight = q.make_empty(source.shape, dtype=torch.float32, device="cuda")
            q.update_quantized(source, weight)
            cases.append((precision, weight, _make_quantizer(precision)))
    for name, weight, quantizer in cases:

        def update(destination, values):
            destination.copy_(values)

        compiled = torch.compile(update, fullgraph=True)
        for iteration in range(3):
            values = source * (iteration + 1.25)
            expected = weight.detach().clone()
            if quantizer is None:
                from quack.gemm_w4 import quantize_act_per_token_fp8

                rows, scales = quantize_act_per_token_fp8(values)
                columns, column_scales = quantize_act_per_token_fp8(
                    values.t().contiguous()
                )
                expected = QuackFP8Weight(rows, columns, scales, column_scales)
            else:
                quantizer.update_quantized(values, expected)
            compiled(weight, values)
            names, _ = weight.__tensor_flatten__()
            for key in names:
                actual_component, expected_component = (
                    getattr(weight, key),
                    getattr(expected, key),
                )
                torch.testing.assert_close(
                    actual_component.view(torch.uint8),
                    expected_component.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
            # Check the representation used for CPU checkpoint restore too.
            cpu = weight.detach().cpu()
            cpu.copy_(values.cpu())
            dense = cpu.dequantize()
            expected_dense = weight.dequantize().cpu()
            torch.testing.assert_close(dense, expected_dense, rtol=0.07, atol=1e-5)
        results.append({"backend": name, "passed": True})
        print(
            f"PASS rank {rank}: {name} eager/compiled publication and CPU restore",
            flush=True,
        )
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / f"rank-{rank:05d}.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
