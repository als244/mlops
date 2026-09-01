# AdamW scaling results

| Backend | Strategy | Parameters | Buckets | Median (ms) | P95 (ms) | Mparam/s | Achieved link rate (Gb/s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| nccl | replicated | 1,073,741,824 | 32 | 843.525 | 846.136 | 1272.92 | 20.367 | passed |

Times are the slower rank for each aligned repetition. Barriers, setup,
materialization, JIT compilation, and handle retirement are outside the
CUDA-event interval. Achieved link rate uses one model payload per rank
for this two-rank topology; it is a comparable end-to-end rate, not a
claim about the provider's internal collective algorithm.
