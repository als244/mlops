"""Distributed LoRA correctness against independent PyTorch expert math.

Run with torchrun. Every process selects one GPU before importing either kernel
library. The same routes are passed to the tested layer and reference.
"""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = (
    _visible.split(",")[_local_rank] if _visible else str(_local_rank)
)

import torch
import torch.distributed as dist

from mlops.expert_parallel.buffers import create_buffer
from mlops.expert_parallel.reference import router_logits, routing_probabilities
from mlops.expert_parallel.reference.lora import expert_computation_lora
from mlops.expert_parallel.transformer_engine import LoRAConfig, MoEConfig
from mlops.expert_parallel.transformer_engine import TEMoELoRA as Layer

BACKEND = "te"


def plain(value):
    return value.detach().dequantize().to(torch.bfloat16).contiguous()


def gather(value):
    parts = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, value.contiguous())
    return torch.cat(parts)


def reference(layer, x, ids, p, dy, *, internal_router=False):
    c, q = layer.config, layer.config.local_experts
    base = list(layer.expert_parameters())
    if BACKEND == "te" and c.compute_precision != "bf16":
        base = [
            torch.stack([plain(w) for w in base[i * q : (i + 1) * q]]) for i in range(2)
        ]
    else:
        base = [plain(w) for w in base]
    base = [gather(w) for w in base]
    factors = [
        gather(plain(w)).float().requires_grad_() for w in layer.lora_parameters()
    ]
    value = x.detach().clone().requires_grad_()
    probabilities = p.detach().clone().requires_grad_()
    if internal_router:
        probabilities = routing_probabilities(
            router_logits(value, layer.router_weight.detach().dequantize().float()),
            ids,
            renormalize_topk=c.renormalize_topk,
        )
    shared = []
    if c.shared_width:
        if BACKEND == "quack":
            shared = [
                plain(w)
                for w in (
                    layer.shared_gate_weight,
                    layer.shared_up_weight,
                    layer.shared_down_weight,
                )
            ]
        else:
            first = layer.shared_gate_up_weight
            shared = [*plain(first).chunk(2, 0), plain(layer.shared_down_weight)]
    out = expert_computation_lora(
        value,
        ids,
        probabilities,
        *base,
        *factors,
        scale=layer.lora_config.scale,
        interleaved=BACKEND == "quack",
        shared_weights=shared,
    )
    inputs = (value, *factors) if internal_router else (value, probabilities, *factors)
    gradients = list(torch.autograd.grad(out, inputs, dy))
    start = 1 if internal_router else 2
    rank = dist.get_rank()
    for index in range(start, len(gradients)):
        dist.all_reduce(gradients[index])
        gradients[index] = gradients[index][rank * q : (rank + 1) * q]
    return out, gradients


def error(actual, expected):
    a, b = actual.detach().float(), expected.detach().float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("Non-finite output or gradient")
    rms = float((a - b).square().mean().sqrt())
    return rms / max(float(b.square().mean().sqrt()), 1e-8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--precision",
        choices=["bf16", "fp8_current"] + (["fp8_block"] if BACKEND == "te" else []),
        default="bf16",
    )
    p.add_argument("--activation-transport", choices=["bf16", "fp8"], default="bf16")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--gradient-dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--num-chunks", type=int, default=1)
    p.add_argument("--num-buffers", type=int, default=1)
    p.add_argument("--shared", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--execution", choices=["eager", "compile", "both"], default="both")
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(721 + rank)
    extra = (
        {
            "num_chunks": args.num_chunks,
            "num_buffers": args.num_buffers,
            "activation_transport": args.activation_transport,
        }
        if BACKEND == "quack"
        else {}
    )
    if BACKEND == "te" and (
        args.num_chunks != 1
        or args.num_buffers != 1
        or args.activation_transport != "bf16"
    ):
        p.error("TE uses one BF16 dispatch buffer")
    config = MoEConfig(
        ep_size=world,
        num_experts=args.experts,
        top_k=args.top_k,
        model_dim=args.dim,
        expert_hidden_dim=args.hidden,
        compute_precision=args.precision,
        num_shared_experts=int(args.shared),
        num_comm_sms=16,
        **extra,
    )
    buffer = create_buffer(config, args.tokens, dist.group.WORLD)
    layer = Layer(
        config,
        buffer=buffer,
        device=torch.device("cuda", 0),
        lora=LoRAConfig(
            rank=args.rank,
            alpha=args.alpha,
            gradient_dtype=torch.float32
            if args.gradient_dtype == "fp32"
            else torch.bfloat16,
        ),
    )
    from mlops.expert_parallel.transformer_engine.registry import _runtime

    assert all(
        not hasattr(bank, "_replica_grad_owner")
        for bank in _runtime(layer._handle).banks
    )
    assert sum(
        w.numel() for w in layer.lora_parameters()
    ) == config.local_experts * args.rank * (2 * args.dim + 3 * args.hidden)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / f"rank{rank}-config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "backend": BACKEND,
                "world_size": world,
                "device": str(torch.cuda.get_device_properties(0)),
            },
            default=str,
            indent=2,
        )
        + "\n"
    )
    records, held = [], []
    try:
        modes = (
            (False, True)
            if args.execution == "both"
            else (args.execution == "compile",)
        )
        for nonzero_b in (False, True):
            with torch.no_grad():
                for parameter in (layer.lora_gate_up_b, layer.lora_down_b):
                    target = getattr(parameter, "_data", parameter)
                    target.normal_(0, 0.015) if nonzero_b else target.zero_()
            for skewed in (False, True):
                x = torch.randn(
                    args.tokens,
                    args.dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                dy = torch.randn_like(x)
                if skewed:
                    ids = (
                        torch.arange(args.top_k, device="cuda", dtype=torch.int32)[None]
                        .expand(args.tokens, -1)
                        .contiguous()
                    )
                else:
                    ids = (
                        torch.randn(args.tokens, args.experts, device="cuda")
                        .topk(args.top_k, -1)
                        .indices.to(torch.int32)
                    )
                probabilities = (
                    torch.randn(args.tokens, args.top_k, device="cuda")
                    .softmax(-1)
                    .requires_grad_()
                )
                expected, expected_gradients = reference(
                    layer, x, ids, probabilities, dy
                )
                for use_compile in modes:
                    print(
                        f"{datetime.now(UTC).isoformat()} START backend={BACKEND} precision={args.precision} nonzero_b={nonzero_b} skewed={skewed} compile={use_compile} recompute={args.recompute}",
                        flush=True,
                    )
                    call = (
                        torch.compile(
                            layer,
                            fullgraph=True,
                            options={
                                "triton.cudagraphs": False,
                                "emulate_precision_casts": True,
                            },
                        )
                        if use_compile
                        else layer
                    )
                    if args.recompute:
                        from torch.utils.checkpoint import checkpoint

                        output = checkpoint(
                            call,
                            x,
                            ids,
                            probabilities,
                            use_reentrant=False,
                            preserve_rng_state=False,
                            early_stop=False,
                        )
                    else:
                        output = call(x, ids, probabilities)
                    gradients = torch.autograd.grad(
                        output, (x, probabilities, *layer.lora_parameters()), dy
                    )
                    names = ("dx", "dp", "gate_up_a", "gate_up_b", "down_a", "down_b")
                    errors = {
                        "output": error(output, expected),
                        **{
                            n: error(a, b)
                            for n, a, b in zip(names, gradients, expected_gradients)
                        },
                    }
                    limit = 0.045 if args.precision == "bf16" else 0.20
                    assert max(errors.values()) < limit, errors
                    assert all(
                        p.grad is None
                        for p in layer.parameters()
                        if not p.requires_grad
                    )
                    assert len([p for p in layer.parameters() if p.requires_grad]) == 4
                    assert all(
                        g.dtype == layer.lora_config.gradient_dtype
                        for g in gradients[2:]
                    )
                    for gradient, snapshot in held:
                        torch.testing.assert_close(gradient, snapshot, rtol=0, atol=0)
                    held = [(g, g.clone()) for g in gradients[2:]]
                    record = {
                        "backend": BACKEND,
                        "precision": args.precision,
                        "nonzero_b": nonzero_b,
                        "skewed": skewed,
                        "execution": "compile" if use_compile else "eager",
                        "recompute": args.recompute,
                        "status": "PASS",
                        "relative_rms": errors,
                    }
                    records.append(record)
                    (args.outdir / f"rank{rank}.json").write_text(
                        json.dumps(records, indent=2) + "\n"
                    )
                    print(
                        f"{datetime.now(UTC).isoformat()} PASS {json.dumps(record)}",
                        flush=True,
                    )
                del output, gradients, expected, expected_gradients
        # Two live forwards must own their saved state independently of buffers.
        pending = []
        for _ in range(2):
            x = torch.randn(
                args.tokens,
                args.dim,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            ids = (
                torch.randn(args.tokens, args.experts, device="cuda")
                .topk(args.top_k, -1)
                .indices.to(torch.int32)
            )
            probabilities = (
                torch.randn(args.tokens, args.top_k, device="cuda")
                .softmax(-1)
                .requires_grad_()
            )
            dy = torch.randn_like(x)
            expected, expected_gradients = reference(layer, x, ids, probabilities, dy)
            output = call(x, ids, probabilities)
            pending.append((output, x, probabilities, dy, expected, expected_gradients))
        for output, x, probabilities, dy, expected, expected_gradients in reversed(
            pending
        ):
            gradients = torch.autograd.grad(
                output, (x, probabilities, *layer.lora_parameters()), dy
            )
            errors = {
                "output": error(output, expected),
                **{
                    n: error(a, b)
                    for n, a, b in zip(names, gradients, expected_gradients)
                },
            }
            assert max(errors.values()) < limit, errors
            records.append(
                {
                    "status": "PASS",
                    "test": "two_live_forwards_reverse_backward",
                    "relative_rms": errors,
                }
            )
        del pending, output, gradients, expected, expected_gradients
        # Exercise ordinary .backward() accumulation, beyond functional grad calls.
        layer.zero_grad(set_to_none=True)
        x = torch.randn(
            args.tokens,
            args.dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        ids = (
            torch.randn(args.tokens, args.experts, device="cuda")
            .topk(args.top_k, -1)
            .indices.to(torch.int32)
        )
        probabilities = (
            torch.randn(args.tokens, args.top_k, device="cuda")
            .softmax(-1)
            .requires_grad_()
        )
        dy = torch.randn_like(x)
        _, expected_gradients = reference(layer, x, ids, probabilities, dy)
        for _ in range(2):
            call(x, ids, probabilities).backward(dy)
        gradients = (
            x.grad,
            probabilities.grad,
            *(w.grad for w in layer.lora_parameters()),
        )
        errors = {
            n: error(a, 2 * b) for n, a, b in zip(names, gradients, expected_gradients)
        }
        assert max(errors.values()) < limit, errors
        assert all(g.dtype == layer.lora_config.gradient_dtype for g in gradients[2:])
        assert all(w.grad is None for w in layer.parameters() if not w.requires_grad)
        records.append(
            {"status": "PASS", "test": "gradient_accumulation", "relative_rms": errors}
        )
        layer.zero_grad(set_to_none=True)
        # Check the frozen router still contributes to dX using its own selected IDs.
        x = torch.randn(
            args.tokens,
            args.dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        ids, probabilities, *_ = layer.route(x)
        dy = torch.randn_like(x)
        expected, expected_gradients = reference(
            layer, x, ids.detach(), probabilities, dy, internal_router=True
        )
        output = layer(x)
        gradients = torch.autograd.grad(output, (x, *layer.lora_parameters()), dy)
        errors = {
            "output": error(output, expected),
            "dx": error(gradients[0], expected_gradients[0]),
        }
        assert max(errors.values()) < (0.05 if args.precision == "bf16" else 0.20), (
            errors
        )
        records.append(
            {
                "status": "PASS",
                "test": "frozen_router_input_gradient",
                "relative_rms": errors,
            }
        )
        (args.outdir / f"rank{rank}.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        print(
            f"LORA_CORRECTNESS_PASS backend={BACKEND} rank={rank} cases={len(records)}",
            flush=True,
        )
    finally:
        layer.close()
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
