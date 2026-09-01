# AdamW scaling results

| Backend | Strategy | Parameters | Buckets | Median (ms) | P95 (ms) | Mparam/s | Achieved link rate (Gb/s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| nccl | sharded | 4,096 | 1 | 0.341 | 1.031 | 12.00 | 0.192 | passed |
| nccl | sharded | 8,192 | 1 | 0.360 | 0.443 | 22.75 | 0.364 | passed |
| nccl | sharded | 16,384 | 1 | 0.357 | 0.451 | 45.86 | 0.734 | passed |
| nccl | sharded | 32,768 | 1 | 0.369 | 0.454 | 88.90 | 1.422 | passed |
| nccl | sharded | 65,536 | 1 | 0.442 | 0.511 | 148.22 | 2.371 | passed |
| nccl | sharded | 131,072 | 1 | 0.414 | 0.481 | 316.70 | 5.067 | passed |
| nccl | sharded | 262,144 | 1 | 0.452 | 0.544 | 580.60 | 9.290 | passed |
| nccl | sharded | 524,288 | 1 | 0.630 | 0.769 | 831.84 | 13.310 | passed |
| nccl | replicated | 4,096 | 1 | 0.300 | 0.325 | 13.66 | 0.218 | passed |
| nccl | replicated | 8,192 | 1 | 0.289 | 0.333 | 28.37 | 0.454 | passed |
| nccl | replicated | 16,384 | 1 | 0.303 | 1.165 | 54.13 | 0.866 | passed |
| nccl | replicated | 32,768 | 1 | 0.298 | 0.436 | 110.13 | 1.762 | passed |
| nccl | replicated | 65,536 | 1 | 0.294 | 0.340 | 223.00 | 3.568 | passed |
| nccl | replicated | 131,072 | 1 | 0.312 | 0.381 | 420.36 | 6.726 | passed |
| nccl | replicated | 262,144 | 1 | 0.385 | 0.511 | 680.85 | 10.894 | passed |
| nccl | replicated | 524,288 | 1 | 0.631 | 0.740 | 830.37 | 13.286 | passed |

Times are the slower rank for each aligned repetition. Barriers, setup,
materialization, JIT compilation, and handle retirement are outside the
CUDA-event interval. Achieved link rate uses one model payload per rank
for this two-rank topology; it is a comparable end-to-end rate, not a
claim about the provider's internal collective algorithm.
