# AdamW scaling results

| Backend | Strategy | Parameters | Buckets | Median (ms) | P95 (ms) | Mparam/s | Effective collective payload (GB/s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| gloo | sharded | 1,048,576 | 1 | 3.482 | 5.196 | 301.12 | 1.204 | passed |
| gloo | sharded | 2,097,152 | 1 | 4.822 | 7.672 | 434.87 | 1.739 | passed |
| gloo | sharded | 4,194,304 | 1 | 9.734 | 15.601 | 430.91 | 1.724 | passed |
| gloo | sharded | 8,388,608 | 1 | 18.760 | 25.256 | 447.14 | 1.789 | passed |
| gloo | sharded | 16,777,216 | 1 | 77.278 | 82.114 | 217.10 | 0.868 | passed |
| gloo | sharded | 33,554,432 | 1 | 151.698 | 157.397 | 221.19 | 0.885 | passed |
| gloo | sharded | 67,108,864 | 1 | 240.524 | 267.546 | 279.01 | 1.116 | passed |
| gloo | sharded | 134,217,728 | 1 | 438.389 | 451.696 | 306.16 | 1.225 | passed |
| gloo | sharded | 268,435,456 | 1 | 858.808 | 886.382 | 312.57 | 1.250 | passed |
| gloo | sharded | 536,870,912 | 1 | 1721.327 | 1756.024 | 311.89 | 1.248 | passed |
| gloo | sharded | 1,073,741,824 | 1 | 3407.625 | 3483.480 | 315.10 | 1.260 | passed |
| gloo | sharded | 2,147,483,648 | — | — | — | — | — | failed: cell produced no rank-zero result; inspect the per-host logs |
| gloo | replicated | 1,048,576 | 1 | 1.884 | 2.824 | 556.69 | 1.113 | passed |
| gloo | replicated | 2,097,152 | 1 | 3.524 | 4.449 | 595.10 | 1.190 | passed |
| gloo | replicated | 4,194,304 | 1 | 5.451 | 7.038 | 769.43 | 1.539 | passed |
| gloo | replicated | 8,388,608 | 1 | 10.423 | 11.090 | 804.81 | 1.610 | passed |
| gloo | replicated | 16,777,216 | 1 | 24.084 | 25.616 | 696.61 | 1.393 | passed |
| gloo | replicated | 33,554,432 | 1 | 39.540 | 47.504 | 848.61 | 1.697 | passed |
| gloo | replicated | 67,108,864 | 1 | 80.041 | 87.211 | 838.43 | 1.677 | passed |
| gloo | replicated | 134,217,728 | 1 | 162.354 | 168.375 | 826.70 | 1.653 | passed |
| gloo | replicated | 268,435,456 | 1 | 342.667 | 356.425 | 783.37 | 1.567 | passed |
| gloo | replicated | 536,870,912 | 1 | 653.344 | 682.115 | 821.73 | 1.643 | passed |
| gloo | replicated | 1,073,741,824 | 1 | 1301.829 | 1333.467 | 824.79 | 1.650 | passed |
| gloo | replicated | 2,147,483,648 | 1 | 2610.713 | 2654.796 | 822.57 | 1.645 | passed |

Times are the slower rank for each aligned repetition. Barriers, setup,
materialization, JIT compilation, and handle retirement are outside the
CUDA-event interval. Effective collective payload is a workload-normalized
rate (one model payload for replicated, two for sharded), not physical link
bandwidth.
