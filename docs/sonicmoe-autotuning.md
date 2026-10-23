# SonicMoE tuning defaults on Hopper

`mlops.quack_runtime.initialize()` extends the pinned Quack SM90 candidate
list. Ordinary MLOps SonicMoE calls and ShadowSpill lowering/profiling receive
the same defaults. Other architectures retain their candidate lists.

The existing N=96 additions remain. Nine exact configurations are added below;
these rows are not expanded across swaps or cluster shapes. `swap_ab=False`,
`tile_k=None`, static persistence, and ordinary TMA (no gather) apply to each.
Cooperative candidates are omitted when the caller disables cooperative tuning.

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

## Selection evidence

The 2026-10-22 Della H100 sweep used GLM-5.3-Flash expert shapes: 288 experts,
top-8 routing, width 4096, hidden width 2048, LoRA rank/alpha 32, BF16 router,
2K/4K/8K/16K/32K/64K tokens per rank, BF16/FP8 and full/LoRA training.
Its 24 SonicMoE cases produced 162 distinct GEMM timing audits, all matched
against persistent tuning-cache entries. Small adapter projections benefited
from N=32 and single-CTA clusters; larger projections/backward GEMMs benefited
from the cooperative M=64/128 tiles.

Starting from the 48-candidate defaults, a greedy selection of nine additions
puts the best recorded retained candidate within 4.54% of the expanded-pool
best on every audited GEMM, with a 0.48% geometric-mean slowdown. This is an
in-sample replay of per-candidate timings, not an independent end-to-end
performance guarantee. It avoids adding all 34 new exact winners.

Raw audits and the reproducible selection script are stored under the Della
reservation root `h100-reservation-16011874-20261022/00_moe_layers/`, in
`autotune_investigation/results/shortlist-190140/`. The replay reads a saved
copy of the original 48 candidates rather than importing today's policy.

## Timing and caching

Each candidate defaults to 50 ms warmup and 180 ms measured GPU time in three
windows, with bounded CUDA-graph replays. The three window results retain a
median against timing noise. Environment overrides are
`MLOPS_QUACK_TUNE_WARMUP_MS`, `MLOPS_QUACK_TUNE_TIME_MS`, and
`MLOPS_QUACK_TUNE_SAMPLES`.

The candidate list is part of the persistent tuning key; changed lists require
new searches. Changing the time budget alone does not invalidate an existing
winner. The compiled-kernel and winner caches are separate, though Quack's
`QUACK_CACHE_DIR` override changes both locations. Default compiler caches stay
under node-local `/tmp`; they are not experiment artifacts.

Distributed candidate sharing remains opt-in. Independent profiling workers
use serial per-GPU tuning and share valid disk-cache entries when their exact
keys match. Read-only operands and fully overwritten outputs are not backed
up or restored between candidates; genuine read/write outputs and aliases are
preserved.
