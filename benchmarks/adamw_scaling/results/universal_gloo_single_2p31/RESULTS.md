# AdamW scaling results

| Backend | Strategy | Parameters | Buckets | Median (ms) | P95 (ms) | Mparam/s | Achieved link rate (Gb/s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| gloo | sharded | 2,147,483,648 | 64 | 8065.701 | 8699.405 | 266.25 | 4.260 | passed |

Times are the slower rank for each aligned repetition. Barriers, setup,
materialization, JIT compilation, and handle retirement are outside the
CUDA-event interval. Achieved link rate uses one model payload per rank
for this two-rank topology; it is a comparable end-to-end rate, not a
claim about the provider's internal collective algorithm.
