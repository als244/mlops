# AdamW scaling results

| Backend | Strategy | Parameters | Buckets | Median (ms) | P95 (ms) | Mparam/s | Effective collective payload (GB/s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| gloo | replicated | 1,048,576 | 1 | 2.838 | 3.634 | 369.48 | 0.739 | passed |
| gloo | replicated | 2,097,152 | 1 | 3.364 | 3.860 | 623.35 | 1.247 | passed |
| gloo | replicated | 4,194,304 | 1 | 6.188 | 8.937 | 677.81 | 1.356 | passed |
| gloo | replicated | 8,388,608 | 1 | 10.817 | 12.453 | 775.47 | 1.551 | passed |
| gloo | replicated | 16,777,216 | 1 | 19.521 | 29.793 | 859.43 | 1.719 | passed |
| gloo | replicated | 33,554,432 | 1 | 38.117 | 46.031 | 880.30 | 1.761 | passed |
| gloo | replicated | 67,108,864 | 2 | 74.719 | 81.036 | 898.15 | 1.796 | passed |
| gloo | replicated | 134,217,728 | 4 | 149.083 | 151.634 | 900.29 | 1.801 | passed |

Times are the slower rank for each aligned repetition. Barriers, setup,
materialization, JIT compilation, and handle retirement are outside the
CUDA-event interval. Effective collective payload is a workload-normalized
rate (one model payload for replicated, two for sharded), not physical link
bandwidth.
