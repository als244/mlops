"""Check compact FP8 quantization scratch across changing expert distributions."""

import argparse
import itertools
import json
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    from mlops.expert_parallel.quack.pipeline.quantize import quantize_groups

    torch.manual_seed(817 + rank)
    source = torch.randn(2048, 512, device="cuda", dtype=torch.bfloat16)
    row_scales = (source.float().abs().amax(-1) / 448).clamp_min(1e-12)
    row_data = (source.float() / row_scales[:, None]).to(torch.float8_e4m3fn)
    distributions = [
        (512, 512, 512, 512),
        (16, 112, 1904, 16),
        (0, 2048, 0, 0),
        (128, 0, 256, 512),
    ]
    for lengths in distributions:
        offsets = [0, *itertools.accumulate(lengths)]
        cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        for values, scales in ((source, None), (row_data, row_scales)):
            actual, actual_scales = quantize_groups(
                values, cu, offsets, row_scales=scales
            )
            dense = (
                source.float()
                if scales is None
                else (row_data.float() * row_scales[:, None]).bfloat16().float()
            )
            for group, (begin, end) in enumerate(itertools.pairwise(offsets)):
                if begin == end:
                    continue
                block = dense[begin:end].t()
                expected_scales = (block.abs().amax(-1) / 448).clamp_min(1e-12)
                expected = (
                    (block / expected_scales[:, None])
                    .clamp(-448, 448)
                    .to(torch.float8_e4m3fn)
                )
                result = actual[begin * 512 : end * 512].view(512, end - begin)
                torch.testing.assert_close(
                    actual_scales[group], expected_scales, rtol=2e-6, atol=0
                )
                torch.testing.assert_close(
                    result.float(), expected.float(), rtol=0, atol=0
                )
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / f"rank-{rank:05d}.json").write_text(
        json.dumps({"passed": True, "distributions": distributions}) + "\n"
    )
    print(
        f"PASS rank {rank}: BF16 and FP8 grouped quantization across four distributions",
        flush=True,
    )


if __name__ == "__main__":
    main()
