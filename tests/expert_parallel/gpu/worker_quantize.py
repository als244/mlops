"""Guarded FP8 row-quantization regression across the signed 32-bit offset limit."""

import argparse
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
    from mlops.expert_parallel.quack.kernels.quantize_rows import _quantize_rows

    width, prefix = 2048, 1 << 31
    rows = prefix // width + 1
    elements = rows * width
    # Guards keep the old kernel's negative offsets within our own allocations.
    # This uses about 12 GiB on each GPU in the opt-in Hopper correctness gate.
    source_owner = torch.empty(prefix + elements, device="cuda", dtype=torch.bfloat16)
    source_owner[:width].fill_(0.5)
    source = source_owner[prefix:].view(rows, width)
    source.fill_(0.25)
    output_owner = torch.empty(prefix + elements, device="cuda", dtype=torch.uint8)
    output_owner[:width].fill_(0xA5)
    output = output_owner[prefix:].view(torch.float8_e4m3fn).view(rows, width)
    output.view(torch.uint8).fill_(0x7F)
    scales = torch.empty(rows, device="cuda", dtype=torch.float32)
    _quantize_rows[(rows,)](
        source, output, scales, width, width, width, width, num_warps=4
    )
    selected = output[[0, rows - 2, rows - 1]].float().cpu()
    selected_scales = scales[[0, rows - 2, rows - 1]].cpu()
    torch.testing.assert_close(selected, torch.full_like(selected, 448), rtol=0, atol=0)
    torch.testing.assert_close(
        selected_scales, torch.full_like(selected_scales, 0.25 / 448), rtol=0, atol=0
    )
    assert bool((output_owner[:width] == 0xA5).all().cpu()), "Out-of-range write"
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / f"rank-{rank:05d}.json").write_text(
        json.dumps({"passed": True, "rows": rows, "width": width, "elements": elements})
        + "\n"
    )
    print(f"PASS rank {rank}: FP8 quantization beyond 2**31 elements", flush=True)


if __name__ == "__main__":
    main()
