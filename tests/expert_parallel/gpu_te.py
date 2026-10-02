"""Distributed dense-oracle validation; every rank runs the same call order."""

from __future__ import annotations

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
import torch.distributed as dist
from reference_inputs_te import check_router

from mlops.expert_parallel.reference import (
    expert_computation,
    router_logits,
    routing_probabilities,
)
from mlops.expert_parallel.transformer_engine import MoEConfig


def error_metrics(actual, expected):
    a, b = actual.detach().float(), expected.detach().float()
    d = (a - b).abs()
    rms = b.square().mean().sqrt().clamp_min(1e-14)
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "relative_rms": (d.square().mean().sqrt() / rms).item(),
    }


def check(name, actual, expected, records, limit):
    result = error_metrics(actual, expected)
    records[name] = result
    finite = torch.isfinite(actual).all() and torch.isfinite(expected).all()
    okay = bool(finite) and result["relative_rms"] < limit
    status = torch.tensor(int(okay), device="cuda")
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if not status.item():
        raise AssertionError(f"rank {dist.get_rank()} {name}: {result}, limit={limit}")


def gather_experts(weights):
    result = []
    for w in weights:
        # Reference collectives consume ordinary values, not compute-weight wrappers.
        # Detach before dequantize so no diagnostic conversion enters autograd.
        w = w.detach().dequantize().contiguous()
        parts = [torch.empty_like(w) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, w)
        result.append(torch.cat(parts).detach().requires_grad_())
    return result


def make_routing(pattern, config, rank, iteration):
    s, k, e = config.tokens_per_rank, config.top_k, config.num_experts
    if pattern == "random":
        ids = torch.randn(s, e, device="cuda").topk(k, dim=-1).indices
    elif pattern == "hot0":
        ids = torch.arange(k, device="cuda").expand(s, k)
    elif pattern == "hotlast":
        ids = torch.arange(e - k, e, device="cuda").expand(s, k)
    elif pattern == "remote":
        ids = (
            (
                (rank + 1) % config.ep_size * config.local_experts
                + torch.arange(k, device="cuda")
            )
            % e
        ).expand(s, k)
    else:
        raise ValueError(pattern)
    return ids.to(torch.int32).contiguous(), torch.randn(s, k, device="cuda").softmax(
        -1
    ).detach().requires_grad_()


def run_window(
    layer,
    call,
    weights,
    config,
    patterns,
    order,
    internal,
    seed,
    reference_weights=None,
):
    rank = dist.get_rank()
    torch.manual_seed(seed + rank)
    layer.zero_grad(set_to_none=True)
    for weight in weights:
        weight.grad = None
    ref_weights = gather_experts(
        weights
        if reference_weights is None
        else [w.detach().to("cuda") for w in reference_weights]
    )
    # Plain Tensor.dequantize() can attach an unsupported autograd identity to
    # its input in the pinned PyTorch. Detach before preparing the reference.
    router = layer.router_weight.detach().dequantize().clone().requires_grad_()
    ld = (
        None
        if layer.latent_down_weight is None
        else layer.latent_down_weight.detach().dequantize().clone().requires_grad_()
    )
    lu = (
        None
        if layer.latent_up_weight is None
        else layer.latent_up_weight.detach().dequantize().clone().requires_grad_()
    )
    shared_names = ("shared_gate_weight", "shared_up_weight", "shared_down_weight")
    shared = (
        tuple(
            getattr(layer, name).detach().dequantize().clone().requires_grad_()
            for name in shared_names
        )
        if config.shared_width
        else ()
    )
    records, saved = {}, []
    limit = 0.025 if config.compute_precision == "bf16" else 0.15
    for iteration, pattern in enumerate(patterns):
        x = torch.randn(
            config.tokens_per_rank,
            config.model_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ).requires_grad_()
        xr = x.detach().clone().requires_grad_()
        ids, p = make_routing(pattern, config, rank, iteration)
        pr = p.detach().clone().requires_grad_()
        extra = {}
        y = call(x, **extra) if internal else call(x, ids, p, **extra)
        if internal:
            # Freeze the implementation's top-k choices; test router math separately.
            with torch.no_grad():
                ids, _ = layer.route(x)
            pr = routing_probabilities(
                router_logits(xr, router, dtype=config.router_dtype),
                ids,
                renormalize_topk=config.renormalize_topk,
            )
        yr = expert_computation(
            xr,
            ids,
            pr,
            *ref_weights,
            shared_weights=shared,
            latent_down=ld,
            latent_up=lu,
        )
        check(f"output/{iteration}/{pattern}", y, yr, records, limit)
        dy = torch.randn_like(y)
        saved.append((x, xr, p, pr, y, yr, dy))
    for iteration in order:
        x, xr, p, pr, y, yr, dy = saved[iteration]
        (y.float() * dy.float()).sum().backward()
        (yr.float() * dy.float()).sum().backward()
        check(f"dx/{iteration}", x.grad, xr.grad, records, limit)
        if not internal:
            check(
                f"routing_probability_grad/{iteration}", p.grad, pr.grad, records, limit
            )
    if config.compute_precision == "bf16":
        expert_grads = [w.grad for w in weights]
    else:
        q = config.local_experts
        expert_grads = [
            torch.stack([w.grad for w in weights[i * q : (i + 1) * q]])
            for i in range(3)
        ]
    for index, (grad, ref) in enumerate(zip(expert_grads, ref_weights)):
        assert grad.dtype == config.weight_grad_dtype, (
            grad.dtype,
            config.weight_grad_dtype,
        )
        dist.all_reduce(ref.grad)
        q = config.local_experts
        check(
            f"expert_wgrad/{index}",
            grad,
            ref.grad[rank * q : (rank + 1) * q],
            records,
            limit,
        )
    layer.synchronize_replicated_gradients()
    for name, ref in [
        ("router_weight", router),
        ("latent_down_weight", ld),
        ("latent_up_weight", lu),
        *zip(shared_names, shared),
    ]:
        if ref is None:
            continue
        if ref.grad is None:
            ref.grad = torch.zeros_like(ref)
        dist.all_reduce(ref.grad)
        check(name + "/grad", getattr(layer, name).grad, ref.grad, records, limit)
    snapshot = [s[4].detach().clone() for s in saved]
    snapshot.extend(grad.detach().clone() for grad in expert_grads)
    return records, snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--retain-intermediates", action="store_true")
    parser.add_argument(
        "--skip-master-update",
        action="store_true",
        help="Validate forward/backward only; skip demo CPU weight-update check",
    )
    parser.add_argument("--latent", action="store_true")
    parser.add_argument("--num-shared-experts", type=int, default=0)
    parser.add_argument(
        "--precision", default="bf16", choices=["bf16", "fp8_current", "fp8_block"]
    )
    parser.add_argument("--weight-grad-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--output", default="logs/gpu-validation.json")
    parser.add_argument("--router-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--router-weight-grad-dtype", choices=["fp32", "bf16"], default="fp32"
    )
    args = parser.parse_args()
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    torch.manual_seed(801 + dist.get_rank())
    config = MoEConfig(
        ep_size=dist.get_world_size(),
        num_experts=8,
        top_k=2,
        model_dim=256 if args.latent else 128,
        expert_hidden_dim=256,
        tokens_per_rank=args.tokens,
        latent_dim=128 if args.latent else None,
        parameter_dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[
            args.weight_grad_dtype
        ],
        overlap=False,
        retain_intermediates=args.retain_intermediates,
        num_shared_experts=args.num_shared_experts,
        compute_precision=args.precision,
        router_dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[args.router_dtype],
        router_weight_grad_dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[
            args.router_weight_grad_dtype
        ],
        weight_transport="bf16" if args.precision == "bf16" else "fp8",
        num_comm_sms=16,
        gemm_sm_margin=16,
    )
    from demo_weights_te import close_demo_layer, make_demo_layer

    layer, masters = make_demo_layer(config)
    from mlops.expert_parallel.transformer_engine.communication import _PLAN_FIELDS
    from mlops.expert_parallel.transformer_engine.operators import _fake_state
    from mlops.expert_parallel.transformer_engine.registry import _runtime

    # Compiler-visible shapes, strides and allocation extents must describe
    # MoonEP's actual padded planning outputs, including its two scalar pairs.
    fake_plan = _fake_state(
        torch.empty((), device="meta", dtype=torch.bfloat16), config
    )[:7]

    def layout(tensor):
        return (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.untyped_storage().nbytes(),
        )

    expected_layout = [layout(value) for value in fake_plan]
    runtime = _runtime(layer._handle)
    dispatch = runtime._dispatch

    def checked_dispatch(*inputs, **kwargs):
        result = dispatch(*inputs, **kwargs)
        plan = result[-1]
        actual_layout = [layout(getattr(plan, name)) for name in _PLAN_FIELDS]
        assert actual_layout == expected_layout, (actual_layout, expected_layout)
        return result

    runtime._dispatch = checked_dispatch
    weights = tuple(layer.expert_parameters())
    reference_weights = None if masters is None else masters.tensors()
    call = (
        torch.compile(layer, fullgraph=True, options={"triton.cudagraphs": False})
        if args.compiled
        else layer
    )
    if args.recompute:
        from torch.utils.checkpoint import checkpoint

        base_call = call

        def call(*inputs, **kwargs):
            return checkpoint(
                base_call,
                *inputs,
                use_reentrant=False,
                preserve_rng_state=False,
                early_stop=False,
                **kwargs,
            )

    router_input = torch.randn(
        config.tokens_per_rank,
        config.model_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    _, _, router_record = check_router(layer, router_input)
    results, baselines = (
        {"routing_check": router_record, "configuration": vars(args)},
        {},
    )
    counters = torch._dynamo.utils.counters
    try:
        for overlap in (False, True):
            layer.set_overlap_enabled(overlap)
            cases = [
                (["random"], [0], False),
                (["hot0", "hotlast", "remote"], [0, 1, 2], False),
                (["hotlast", "random", "hot0"], [2, 0, 1], False),
                (["random", "random"], [0, 1], True),
            ]
            graph_count = None
            for index, (patterns, order, internal) in enumerate(cases):
                label = f"overlap={overlap}/case={index}/router={internal}"
                records, snapshot = run_window(
                    layer,
                    call,
                    weights,
                    config,
                    patterns,
                    order,
                    internal,
                    1801 + index,
                    reference_weights,
                )
                results[label] = records
                if args.compiled and not internal:
                    count = counters["stats"]["unique_graphs"]
                    if graph_count is None:
                        graph_count = count
                    assert count == graph_count, (
                        "Routing caused recompilation",
                        graph_count,
                        count,
                    )
                if not overlap:
                    baselines[index] = snapshot
                else:
                    for n, (actual, expected) in enumerate(
                        zip(snapshot, baselines[index])
                    ):
                        check(
                            f"overlap_equivalence/{n}", actual, expected, records, 1e-6
                        )
                if dist.get_rank() == 0:
                    print(
                        json.dumps(
                            {
                                "PASS": label,
                                "max_relative_rms": max(
                                    r["relative_rms"] for r in records.values()
                                ),
                                "unique_graphs": counters["stats"]["unique_graphs"],
                            }
                        ),
                        flush=True,
                    )
        if masters is not None and not args.skip_master_update:
            before = layer.compute_state_report()
            assert all(w.device.type == "cpu" for w in masters.tensors())
            assert all(
                w.grad is not None and w.grad.dtype == config.weight_grad_dtype
                for w in weights
            )
            q = config.local_experts
            with torch.no_grad():
                for index, master in enumerate(masters.tensors()):
                    gradient = (
                        weights[index].grad
                        if config.compute_precision == "bf16"
                        else torch.stack(
                            [w.grad for w in weights[index * q : (index + 1) * q]]
                        )
                    )
                    master.sub_(0.001 * gradient.float().cpu())
            masters.load_into(layer)
            after = layer.compute_state_report()
            assert after["pointers"] == before["pointers"], "Compute addresses changed"
            records, updated = run_window(
                layer,
                call,
                weights,
                config,
                ["random", "random"],
                [0, 1],
                True,
                1804,
                reference_weights,
            )
            assert not torch.equal(updated[0], baselines[3][0]), (
                "Updated weights had no effect"
            )
            results["cpu_master_load"] = records
            results["compute_weight_ownership"] = after
            if dist.get_rank() == 0:
                print(
                    "PASS: CPU masters, compute-only parameters, configured gradients and stable compute addresses",
                    flush=True,
                )
        results["dynamo_counters"] = {str(k): dict(v) for k, v in counters.items()}
        if dist.get_rank() == 0:
            Path(args.output).write_text(json.dumps(results, indent=2))
            print(f"PASS: results saved to {args.output}", flush=True)
    finally:
        close_demo_layer(layer)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
