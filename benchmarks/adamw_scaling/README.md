# AdamW scaling benchmark

This self-contained benchmark measures the complete scheduler-visible
`mlops.optim.AdamW.step()` across two hosts. It sweeps exact powers of two from
`2^20` (1,048,576) through `2^31` (2,147,483,648) BF16 parameters for both:

- `opt_state_strategy="sharded"` (the public default): reduce-scatter, local
  fused AdamW update, then all-gather;
- `opt_state_strategy="replicated"`: all-reduce, then a full local fused AdamW
  update.

Parameters, gradients, reduction buffers, optimizer state, and master policy
are BF16. The default benchmark input uses ordinary tensors no larger than
64 MiB. Independently, the optimizer enforces `bucket_bytes=64 MiB` as a hard
collective-payload cap and may split any parameter at element boundaries.

`--parameter-layout single` instead constructs one parameter containing the
entire requested element count. It verifies that model tensor boundaries no
longer affect optimizer bucket sizing; the default is `bucketed`.

## Timing contract

Each cell runs in a fresh two-rank process group. Setup, deterministic nonzero
initialization, flat-bucket materialization, Triton/NCCL warmup, and group
barriers are outside the timed interval.

For each repetition, both ranks:

1. align at a Gloo control barrier outside timing;
2. record a CUDA start event on the caller stream;
3. call the complete `optimizer.step()`;
4. record and synchronize a CUDA end event on that same stream;
5. retire completed handles and align outside timing.

The optimizer serializes bounded buckets on the caller stream. ProcessGroupNCCL
bridges asynchronous collective completion onto that stream; Gloo uses a
portable host completion boundary. The event duration therefore contains
gradient packing, collective execution, fused update, and (for sharded state)
parameter publication. It does not contain rank skew, a device-wide
synchronization, or setup. The reported sample is the maximum of the two
aligned rank samples; tables show its median and p95.

`achieved_link_gbit_per_second` divides one full parameter payload per rank by
complete-step time for this two-rank topology. It is a comparable end-to-end
rate, not a claim about the provider's internal collective algorithm: the
interval also contains packing and optimizer compute.

## Run the two-host sweep

From the `mlops` repository on Chicago:

```bash
conda run -n dataflow python benchmarks/adamw_scaling/run_fleet_sweep.py
```

The defaults use:

- local rank: Chicago;
- remote SSH host: `tubingen`;
- master address: `192.168.50.23`;
- local/remote 25 GbE interfaces: `enp114s0f1np1` / `enp4s0f0np0`;
- NCCL, both state strategies, 3 warmups, and 10 measured repetitions;
- a new process group and a five-minute timeout for every cell.

Common focused runs:

```bash
# One small canary for each strategy.
conda run -n dataflow python benchmarks/adamw_scaling/run_fleet_sweep.py \
  --min-exponent 20 --max-exponent 20

# Probe whether the installed cross-host Gloo backend supports CUDA tensors.
conda run -n dataflow python benchmarks/adamw_scaling/run_fleet_sweep.py \
  --backends gloo --min-exponent 20 --max-exponent 20

# Reuse an already-synchronized remote checkout.
conda run -n dataflow python benchmarks/adamw_scaling/run_fleet_sweep.py \
  --no-sync-source
```

Use `--help` for host paths, NICs, ports, bucket size, repetitions, timeout,
backend, strategy, exponent range, Conda executable, and output-directory
options.

## Outputs

The default `results/` contains:

```text
results/
├── summary.json                    complete machine-readable sweep
├── RESULTS.md                      human-readable aggregate table
└── cells/<backend>_<strategy>_2pN/
    ├── result.json                 aggregate plus both per-rank samples
    ├── chicago.log
    └── tubingen.log
```

An allocation failure, unsupported backend, process failure, or timeout is a
cell result rather than a reason to lose already completed cells. Upper sizes
may be reported as skipped/OOM on the smallest participating GPU.

## Saved reference

The first pre-refactor sweep is preserved under
[`results/reference_bucketed_64m`](results/reference_bucketed_64m/RESULTS.md).
It contains 24 passing NCCL cells collected on 2026-08-04 with an RTX 5090
and RTX 3090 connected by 25 GbE: 12 powers of two for each state strategy,
three warmups, and ten measured repetitions. Each model-visible parameter in
this reference is at most 64 MiB. The per-rank JSON records exact software,
device, timing, allocator, and execution-manifest metadata.

Two follow-up ablations are also retained:

- [`results/no_overlap_64m`](results/no_overlap_64m/RESULTS.md) serializes the
  bucket phases from `2^25` through `2^31` for both strategies;
- [`results/single_parameter_2p31`](results/single_parameter_2p31/RESULTS.md)
  represents all `2^31` elements with one model-visible parameter and hence
  one oversized bucket.

On this topology, the historical overlap path improved large-cell medians by
only 0.2–0.7% for sharded state and 0.8–1.2% for replicated state. It was
removed in favor of one backend-portable serialized algorithm and reusable
bounded workspace. Element-level splitting is retained for memory
correctness: a one-tensor 2-billion-element Gloo sharded canary exhausted the
24 GiB rank when treated as one collective, while its bounded form completed.

The current `2^12`–`2^31` NCCL results and complete latency/workspace
comparison are documented in
[`results/universal_serial_64m/COMPARISON.md`](results/universal_serial_64m/COMPARISON.md).

## Reproduce one cell manually

`benchmark.py` is the rank worker. Launch one copy per host with normal
`torchrun --nnodes=2 --nproc-per-node=1`, using node ranks 0 and 1 and identical
arguments. Rank zero writes `--output`. In ordinary use, prefer the sweep
launcher because it isolates failure and preserves logs automatically.

`probe_gloo_collectives.py` is a focused backend diagnostic. It separately
reports whether Gloo's CUDA all-reduce, reduce-scatter, and all-gather Work
objects support `block_current_stream()` and a host `wait()`; it is not part
of optimizer timing. `get_future()` is deliberately not used as a capability
query because an unsupported Gloo Work may hang instead of rejecting it.
