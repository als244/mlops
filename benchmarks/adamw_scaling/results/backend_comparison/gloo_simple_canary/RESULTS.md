# AdamW scaling results

| Backend | Strategy | Parameters | Buckets | Median (ms) | P95 (ms) | Mparam/s | Effective collective payload (GB/s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| gloo | sharded | 1,048,576 | 1 | 3.548 | 5.151 | 295.58 | 1.182 | passed |
| gloo | replicated | 1,048,576 | 1 | 2.286 | 3.317 | 458.78 | 0.918 | passed |

Times are the slower rank for each aligned repetition. Barriers, setup,
materialization, JIT compilation, and handle retirement are outside the
CUDA-event interval. Effective collective payload is a workload-normalized
rate (one model payload for replicated, two for sharded), not physical link
bandwidth.
