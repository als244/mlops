"""Two distinct layers share scratch; compare with independent-bank execution."""

import argparse
import json
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

from _bootstrap import initialize_group, select_rank_device

select_rank_device()

import torch
from torch import distributed as dist
from torch.utils.checkpoint import checkpoint


def copy_parameter(destination, source):
    if hasattr(source, "__tensor_flatten__"):
        for name in source.__tensor_flatten__()[0]:
            getattr(destination, name).copy_(getattr(source, name))
    else:
        destination.copy_(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    initialize_group()
    from mlops.expert_parallel import QuackMoE, QuackMoEConfig, QuackMoELoRA
    from mlops.expert_parallel.buffers import create_buffer
    from mlops.expert_parallel.quack.parameters.components import rowwise_payload
    from mlops.expert_parallel.quack.registry import _runtime

    records = []
    for precision in ("bf16", "fp8_current"):
        for layer_type in (QuackMoE, QuackMoELoRA):
            # Each case constructs four independent runtime handles. Do not
            # accumulate their guards against Dynamo's per-function cache limit.
            torch._dynamo.reset()
            with ExitStack() as stack:
                cfg = QuackMoEConfig(
                    ep_size=dist.get_world_size(),
                    num_experts=8,
                    top_k=2,
                    model_dim=128,
                    expert_hidden_dim=128,
                    compute_precision=precision,
                    init_std=0.08,
                )
                buffer = create_buffer(cfg, 128, dist.group.WORLD)
                stack.callback(buffer.destroy)
                layers = []
                for shared in (False, True):
                    pair = []
                    for _ in range(2):
                        layer = layer_type(
                            replace(cfg, share_expert_banks=shared),
                            dist.group.WORLD,
                            buffer=buffer,
                        )
                        stack.callback(layer.close)
                        pair.append(layer)
                    layers.append(pair)
                separate, shared = layers
                for a, b in zip(
                    _runtime(shared[0]._handle).banks, _runtime(shared[1]._handle).banks
                ):
                    assert a is b
                with torch.no_grad():
                    for original, target in zip(separate, shared):
                        for source, destination in zip(
                            original.parameters(), target.parameters()
                        ):
                            copy_parameter(destination, source)
                for a, b in zip(
                    shared[0].expert_parameters(), shared[1].expert_parameters()
                ):
                    assert (
                        rowwise_payload(a).data_ptr() != rowwise_payload(b).data_ptr()
                    )

                calls = [
                    [
                        torch.compile(
                            layer, fullgraph=True, options={"triton.cudagraphs": False}
                        )
                        for layer in pair
                    ]
                    for pair in layers
                ]
                for recompute in (False, True):
                    for repeat in range(2):
                        xs = [
                            torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
                            for _ in range(2)
                        ]
                        dys = [torch.randn_like(x) for x in xs]
                        results = []
                        for pair, compiled in zip(layers, calls):
                            inputs = [x.detach().requires_grad_() for x in xs]
                            outputs = [
                                checkpoint(
                                    call,
                                    x,
                                    use_reentrant=False,
                                    preserve_rng_state=False,
                                    early_stop=False,
                                )
                                if recompute
                                else call(x)
                                for call, x in zip(compiled, inputs)
                            ]
                            params = [
                                p
                                for layer in pair
                                for p in layer.parameters()
                                if p.requires_grad
                            ]
                            grads = torch.autograd.grad(
                                outputs, [*inputs, *params], dys
                            )
                            results.append([*outputs, *grads])
                        for actual, expected in zip(results[1], results[0]):
                            torch.testing.assert_close(
                                actual.float(), expected.float(), rtol=1e-5, atol=1e-6
                            )
                        record = {
                            "precision": precision,
                            "layer": layer_type.__name__,
                            "recompute": recompute,
                            "repeat": repeat,
                            "passed": True,
                        }
                        records.append(record)
                        print(f"PASS rank={dist.get_rank()} {record}", flush=True)
                # Closing one borrower must not destroy the other layer's banks.
                shared[0].close()
                shared[1](xs[1]).sum().backward()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / f"rank-{dist.get_rank()}.json").write_text(
        json.dumps(records, indent=2)
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
