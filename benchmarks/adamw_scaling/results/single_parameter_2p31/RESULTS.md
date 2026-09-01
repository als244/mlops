# AdamW scaling results

| Backend | Strategy | Parameters | Buckets | Median (ms) | P95 (ms) | Mparam/s | Effective collective payload (GB/s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| nccl | sharded | 2,147,483,648 | 1 | 1661.106 | 1664.268 | 1292.80 | 5.171 | passed |
| nccl | replicated | 2,147,483,648 | 1 | 1659.408 | 1664.719 | 1294.13 | 2.588 | passed |

Times are the slower rank for each aligned repetition. Barriers, setup,
materialization, JIT compilation, and handle retirement are outside the
CUDA-event interval. Effective collective payload is a workload-normalized
rate (one model payload for replicated, two for sharded), not physical link
bandwidth.
