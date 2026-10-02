"""Compare MoonEP/QuACK routed outputs and VJPs with a dense CPU reference."""

import argparse
import json
import os
from pathlib import Path

_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = (
    _visible.split(",")[_local_rank] if _visible else str(_local_rank)
)

import torch
from reference_inputs_quack import check_router
from torch import distributed as dist

from mlops.expert_parallel.quack import MoEConfig, MoELayer
from mlops.expert_parallel.quack.parameters.components import rowwise_payload
from mlops.expert_parallel.quack.registry import _runtime
from mlops.expert_parallel.reference import expert_computation


def relative_rms(actual, expected, threshold=0.025):
    actual = actual.detach().cpu().float()
    expected = expected.detach().float()
    if not bool(torch.isfinite(actual).all()):
        raise AssertionError("Nonfinite output")
    value = float(
        (actual - expected).square().mean().sqrt()
        / expected.square().mean().sqrt().clamp_min(1e-12)
    )
    if value >= threshold:
        raise AssertionError(f"Relative RMS {value} >= {threshold}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--precision", choices=["bf16", "fp8_current"], default="bf16")
    parser.add_argument(
        "--activation-transport", choices=["bf16", "fp8"], default="bf16"
    )
    parser.add_argument("--token-padding", type=int, default=128)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--num-buffers", type=int, default=1)
    parser.add_argument("--weight-grad-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--router-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--router-weight-grad-dtype", choices=["fp32", "bf16"], default="fp32"
    )
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    dist.init_process_group("nccl", device_id=device)
    world = dist.get_world_size()
    rank = dist.get_rank()
    torch.manual_seed(910 + rank)
    cfg = MoEConfig(
        ep_size=world,
        num_experts=args.experts,
        top_k=args.top_k,
        model_dim=args.dim,
        expert_hidden_dim=args.hidden,
        tokens_per_rank=args.tokens,
        num_comm_sms=16,
        compute_precision=args.precision,
        activation_transport=args.activation_transport,
        router_dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[args.router_dtype],
        router_weight_grad_dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[
            args.router_weight_grad_dtype
        ],
        token_padding=args.token_padding,
        num_chunks=args.num_chunks,
        num_buffers=args.num_buffers,
        weight_grad_dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[
            args.weight_grad_dtype
        ],
    )
    from mlops.expert_parallel.buffers import create_buffer

    buffer = create_buffer(cfg, args.tokens, dist.group.WORLD)
    layer = MoELayer(cfg, dist.group.WORLD, device=device, buffer=buffer)
    router_input = torch.randn(
        args.tokens, args.dim, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    _, _, router_record = check_router(layer, router_input)
    runtime = _runtime(layer._handle)
    original_dispatch = runtime._finish_dispatch
    padding_checks = []

    def checked_dispatch(streams, pending, **options):
        result = original_dispatch(streams, pending, **options)
        torch.cuda.current_stream().wait_event(result.ready)
        xp, pp, plan = result.received, result.probabilities, result.plan
        from mlops.expert_parallel.quack.activation_transport import row_data

        xp = row_data(xp)
        padding_rows = 0
        for start, count in plan.zero_fill_ranges.cpu().tolist():
            if not count:
                continue
            assert 0 <= start and start + count <= xp.shape[0], (
                "Padding range outside dispatched buffer"
            )
            padding_rows += count
            assert torch.count_nonzero(xp[start : start + count].float()).item() == 0, (
                "Nonzero dispatched padding"
            )
            if pp is not None:
                assert torch.count_nonzero(pp[start : start + count]).item() == 0, (
                    "Nonzero padding probability"
                )
        padding_checks.append(
            {
                "phase": "backward" if pending.saved is not None else "forward",
                "padding_rows": padding_rows,
                "zero_filled": True,
            }
        )
        return result

    runtime._finish_dispatch = checked_dispatch
    call = (
        torch.compile(layer, fullgraph=True, options={"triton.cudagraphs": False})
        if args.compiled
        else layer
    )
    local = []
    for w in layer.expert_parameters():
        data = rowwise_payload(w).detach().cpu()
        if args.precision != "bf16":
            data = (data.float() * w._scale.detach().cpu()[..., None]).bfloat16()
        local.append(data)
    peers = [None] * world
    dist.all_gather_object(peers, local)
    w1, w2 = [torch.cat([peer[i] for peer in peers]) for i in range(2)]
    records = []
    for routing in ("uniform", "skewed", "uniform", "skewed"):
        padding_checks.clear()
        x = torch.randn(
            args.tokens,
            args.dim,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        ids = (
            (
                torch.arange(args.tokens * args.top_k, device=device).reshape(
                    args.tokens, args.top_k
                )
                % args.experts
                if routing == "uniform"
                else torch.arange(args.top_k, device=device).expand(
                    args.tokens, args.top_k
                )
            )
            .int()
            .contiguous()
        )
        p = (
            torch.randn(args.tokens, args.top_k, device=device)
            .softmax(-1)
            .requires_grad_()
        )
        dy = torch.randn_like(x)
        local = [v.detach().cpu() for v in (x, p, ids, dy)]
        dist.all_gather_object(peers, local)
        all_x, all_p, all_ids, all_dy = [
            torch.cat([peer[i] for peer in peers]) for i in range(4)
        ]
        ref_args = [
            v.detach().requires_grad_() for v in (all_x, all_p, w1.float(), w2.float())
        ]
        yref = expert_computation(
            ref_args[0],
            all_ids,
            ref_args[1],
            ref_args[2][:, 0::2],
            ref_args[2][:, 1::2],
            ref_args[3],
        )
        gref = torch.autograd.grad(yref, ref_args, all_dy)
        if args.recompute:
            from torch.utils.checkpoint import checkpoint

            y = checkpoint(
                call,
                x,
                ids,
                p,
                use_reentrant=False,
                preserve_rng_state=False,
                early_stop=False,
            )
        else:
            y = call(x, ids, p)
        actual = torch.autograd.grad(y, (x, p, *layer.expert_parameters()), dy)
        # Exercise reuse after a different routing distribution: consumed
        # replicas must be cleared by MoonEP before the next invocation.
        for bank in runtime.banks:
            assert not torch.count_nonzero(bank.local_replica_grad), (
                "Replica gradient not cleared"
            )
        start, end = rank * args.tokens, (rank + 1) * args.tokens
        e0, e1 = rank * cfg.local_experts, (rank + 1) * cfg.local_experts
        expected = (
            gref[0][start:end],
            gref[1][start:end],
            gref[2][e0:e1],
            gref[3][e0:e1],
        )
        if routing == "skewed":
            for gradient in actual[2:]:
                for local_expert in range(cfg.local_experts):
                    if e0 + local_expert >= args.top_k:
                        assert not torch.count_nonzero(gradient[local_expert]), (
                            "Nonzero globally empty home gradient"
                        )
        # Full-layer FP8 is also checked against the unquantized BF16 training
        # baseline. Its approximation budget is distinct from the fixed 0.025
        # implementation check against explicit quantized expert arithmetic.
        threshold = 0.025 if args.precision == "bf16" else 0.12

        def check(a, b, threshold=threshold):
            return relative_rms(a, b, threshold)

        errors = {"output": check(y, yref[start:end])}
        errors.update(
            {
                name: check(a, b)
                for name, a, b in zip(("dx", "dp", "dw1", "dw2"), actual, expected)
            }
        )
        forward_calls = 2 if args.recompute else 1
        assert [r["phase"] for r in padding_checks] == (
            ["forward"] * cfg.num_chunks * forward_calls + ["backward"] * cfg.num_chunks
        )
        if cfg.token_padding == 128 and routing == "uniform":
            assert all(r["padding_rows"] > 0 for r in padding_checks), (
                "Padding was not exercised"
            )
        records.append(
            {
                "routing": routing,
                "status": "PASS",
                "relative_rms": errors,
                "threshold": threshold,
                "padding_checks": list(padding_checks),
                "reference": "Dense BF16 baseline with dequantized row-oriented FP8 weights"
                if args.precision != "bf16"
                else "Dense BF16",
            }
        )
        print(f"PASS rank {rank} distributed {routing}: {errors}", flush=True)
        del y, actual, x, p, dy, gref, yref
        dist.barrier()
    layer.close()
    buffer.destroy()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / f"rank{rank}.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "rank": rank,
                "precision": args.precision,
                "activation_transport": args.activation_transport,
                "token_padding": cfg.token_padding,
                "dispatched_rows": cfg.dispatched_rows,
                "routing_check": router_record,
                "cases": records,
                "configuration": vars(args) | {"outdir": str(args.outdir)},
                "ep_size": world,
            },
            indent=2,
        )
        + "\n"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
