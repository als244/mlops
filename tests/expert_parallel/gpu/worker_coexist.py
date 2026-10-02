"""Coexistence smoke check using only the four public layer entrypoints."""

import argparse
import json
from pathlib import Path

from _bootstrap import initialize_group, select_rank_device

select_rank_device()
import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    initialize_group()
    from mlops.expert_parallel import (
        QuackMoE,
        QuackMoEConfig,
        QuackMoELoRA,
        TEMoE,
        TEMoEConfig,
        TEMoELoRA,
        create_buffer,
    )

    records = []
    for precision in ("bf16", "fp8_current"):
        for layer_type, config_type in (
            (QuackMoE, QuackMoEConfig),
            (QuackMoELoRA, QuackMoEConfig),
            (TEMoE, TEMoEConfig),
            (TEMoELoRA, TEMoEConfig),
        ):
            config = config_type(
                ep_size=dist.get_world_size(),
                num_experts=8,
                top_k=2,
                model_dim=128,
                expert_hidden_dim=256,
                num_shared_experts=1,
                compute_precision=precision,
                num_comm_sms=16,
            )
            buffer = create_buffer(config, 128, dist.group.WORLD)
            layer = layer_type(config, dist.group.WORLD, buffer=buffer, device="cuda:0")
            try:
                x = torch.randn(
                    128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
                )
                parameters = [p for p in layer.parameters() if p.requires_grad]
                y = layer(x)
                gradients = torch.autograd.grad(
                    y, (x, *parameters), torch.randn_like(y)
                )
                assert y.shape == x.shape and torch.isfinite(y).all()
                assert all(torch.isfinite(g).all() for g in gradients)
                assert all(g.dtype == torch.float32 for g in gradients[1:])
                record = {
                    "module": layer_type.__module__,
                    "layer": layer_type.__name__,
                    "precision": precision,
                    "trainable_tensors": len(parameters),
                    "status": "PASS",
                }
                records.append(record)
                print(json.dumps(record), flush=True)
            finally:
                layer.close()
                buffer.destroy()
            dist.barrier()
    (args.outdir / f"rank{dist.get_rank()}.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
