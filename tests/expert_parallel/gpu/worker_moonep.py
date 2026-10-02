"""Large-token MoonEP planning and repeated saved-plan communication checks."""

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
    initialize_group()
    from moonep import planning

    from mlops.expert_parallel import QuackMoEConfig, create_buffer
    from mlops.expert_parallel._compat import initialize_moonep, moonep_compiler

    initialize_moonep()
    assert planning._mlops_planner_opt_level == 2
    assert moonep_compiler.apply() is False, "patch must be idempotent"
    rank, world = dist.get_rank(), dist.get_world_size()
    config = QuackMoEConfig(
        ep_size=world,
        num_experts=192,
        top_k=4,
        model_dim=1024,
        expert_hidden_dim=1280,
    )
    torch.manual_seed(4100 + rank)
    args.outdir.mkdir(parents=True, exist_ok=True)
    records = []
    for tokens in (32768, 65536):
        buffer = create_buffer(config, tokens, dist.group.WORLD)
        try:
            for routing in ("random", "skewed", "random"):
                ids = (
                    (
                        torch.randn(tokens, 192, device="cuda").topk(4, dim=-1).indices
                        if routing == "random"
                        else torch.arange(4, device="cuda").expand(tokens, 4)
                    )
                    .int()
                    .contiguous()
                )
                histogram = torch.bincount(ids.flatten().long(), minlength=192).int()
                # Every local/cross-rank partial sum is exactly representable.
                # This checks transport independently of BF16 reduction rounding.
                x = (
                    torch.randint(-16, 17, (tokens, 1024), device="cuda").bfloat16()
                    / 16
                )
                probabilities = torch.full((tokens, 4), 0.25, device="cuda")
                received, received_p, _, plan = buffer.dispatch(
                    x, probabilities, ids, histogram
                )
                y, returned_p, _ = buffer.combine(
                    plan=plan, hidden_nvsh=received, route_weights_nvs=received_p
                )
                torch.testing.assert_close(y, x * 4, rtol=0, atol=0)
                torch.testing.assert_close(returned_p, probabilities, rtol=0, atol=0)
                del received, received_p, y, returned_p
                dy = -x
                received, _, _, _ = buffer.dispatch(dy, plan=plan)
                dx, _, _ = buffer.combine(plan=plan, hidden_nvsh=received)
                torch.testing.assert_close(dx, dy * 4, rtol=0, atol=0)
                del received, dx, dy, x, plan
                record = {
                    "tokens": tokens,
                    "routing": routing,
                    "exact": True,
                    "passed": True,
                }
                records.append(record)
                print(f"PASS rank={rank} {record}", flush=True)
                (args.outdir / f"rank-{rank}.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
        finally:
            buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
