# SonicMoE tuning defaults on Hopper

`mlops.quack_runtime.initialize()` extends the pinned Quack SM90 candidate
list. Ordinary MLOps SonicMoE calls and an engine's lowering or profiling of
them receive the same defaults. Other architectures retain their candidate
lists.

Besides M=128, N=96 ping-pong candidates (both cluster shapes, with and
without `swap_ab` where the epilogue allows it), nine exact configurations are
added below; these rows are not expanded across swaps or cluster shapes.
`swap_ab=False`, `tile_k=None`, static persistence, and ordinary TMA (no
gather) apply to each. Cooperative candidates are omitted when the caller
disables cooperative tuning.

| M | N | Schedule | Cluster M × N |
|---:|---:|---|---|
| 64 | 32 | ping-pong | 1 × 1 |
| 128 | 128 | cooperative | 1 × 2 |
| 64 | 256 | cooperative | 2 × 1 |
| 128 | 192 | cooperative | 2 × 1 |
| 64 | 192 | ping-pong | 1 × 1 |
| 128 | 32 | ping-pong | 1 × 1 |
| 64 | 128 | ping-pong | 1 × 1 |
| 64 | 128 | ping-pong | 2 × 1 |
| 128 | 32 | cooperative | 2 × 1 |

The generic pool grows from 48 to 57 rather than adopting the 192-candidate
exploration pool. Quack's normal shape, layout, dtype, and epilogue pruning
still applies, so a particular GEMM sees fewer candidates.

The shortlist comes from a Hopper GEMM sweep over full and LoRA expert shapes.
It limits tuning cost; it does not guarantee the same winner on every workload.

## Timing and caching

Each candidate defaults to 50 ms warmup and 180 ms measured GPU time in three
windows, with bounded CUDA-graph replays. The three window results retain a
median against timing noise. Environment overrides are
`MLOPS_QUACK_TUNE_WARMUP_MS`, `MLOPS_QUACK_TUNE_TIME_MS`, and
`MLOPS_QUACK_TUNE_SAMPLES`. Under an allocator that replaces PyTorch's own
(`mlops.quack_runtime.sweeps.native_allocator()` is false), the same calls run
eagerly in place of each graph replay, on the call's own tensors without
rotation copies: PyTorch's allocator serves allocations made during a capture
from the graph's private pool, while a replacement serves them itself and its
threads may poll CUDA events, which a capture forbids process-wide.

The candidate list is part of the persistent tuning key; changed lists require
new searches. Changing the time budget alone does not invalidate an existing
winner. The compiled-kernel and winner caches are separate, though Quack's
`QUACK_CACHE_DIR` override changes both locations. Default compiler caches stay
under node-local `/tmp`; they are not experiment artifacts. A sweep timed
eagerly keeps its winner in its own process and writes nothing to the winner
cache, which holds only graph-timed sweeps; it still reads the entries there.

Distributed candidate sharing is opt-in (`MLOPS_QUACK_TUNE_DISTRIBUTED=1`).
Independent profiling workers use serial per-GPU tuning and share valid
disk-cache entries, which only graph-timed sweeps write, when their exact keys
match. Read-only operands and fully overwritten outputs are not backed up or
restored between candidates; genuine read/write outputs and aliases are
preserved.

The pinned epilogue frontend allocates reduction scratch for its candidate
sweep and passes a smaller view to the selected kernel. The `sink_storage`
compatibility patch preserves that backing extent, strides and storage offset
in the warm-call recipe. A fresh process and a warmed call therefore request
the same workspace; an allocator profile must include the backing allocation,
not just the visible slice. The normal warm launch cache and selected kernels
are retained. Each call owns its scratch, and caller-provided buffers keep their
existing behavior. This patch is source-checked and included in provider cache
identity; rebuild saved provider artifacts after changing the runtime version.

Default winner results live under `~/.quack/cache`; compiled kernels use
`/tmp/<user>/quack_cache`. `QUACK_CACHE_DIR` changes both. Use
`QUACK_FORCE_CACHE_UPDATE=1` to remeasure winners while retaining compiled code.
Cooperative searches require every rank to enter identical tuner searches with
identical inputs and ordering. Ordinary independent workers must leave
`MLOPS_QUACK_TUNE_DISTRIBUTED` disabled.

## Comparing numerical results

Independent searches can select different tiles and therefore different
floating-point reduction groupings. Even with the same selected configuration,
a process that ran the CUDA-graph tuning sweep can differ at BF16 rounding
boundaries from one that only loaded its cache. Both compute the same operation
within numerical tolerances; a cache is not a general bitwise guarantee.

For controlled comparisons, tune in a separate process using PyTorch's allocator
and point the comparison processes at the same winner cache. Changing hardware,
precision or geometry still requires an appropriate numerical comparison.
