# AdamW scaling results

| Backend | Strategy | Parameters | Buckets | Median (ms) | P95 (ms) | Mparam/s | Effective collective payload (GB/s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| nccl | sharded | 1,048,576 | 1 | 1.070 | 1.192 | 979.95 | 3.920 | passed |
| nccl | sharded | 2,097,152 | 1 | 1.977 | 2.431 | 1060.59 | 4.242 | passed |
| nccl | sharded | 4,194,304 | 1 | 3.708 | 4.256 | 1131.05 | 4.524 | passed |
| nccl | sharded | 8,388,608 | 1 | 7.177 | 7.496 | 1168.80 | 4.675 | passed |
| nccl | sharded | 16,777,216 | 1 | 13.762 | 14.365 | 1219.13 | 4.877 | passed |
| nccl | sharded | 33,554,432 | 1 | 26.771 | 28.242 | 1253.38 | 5.014 | passed |
| nccl | sharded | 67,108,864 | 1 | 53.192 | 54.221 | 1261.64 | 5.047 | passed |
| nccl | sharded | 134,217,728 | 1 | 104.173 | 106.161 | 1288.42 | 5.154 | passed |
| nccl | sharded | 268,435,456 | 1 | 207.575 | 208.216 | 1293.19 | 5.173 | passed |
| nccl | sharded | 536,870,912 | 1 | 415.021 | 416.754 | 1293.60 | 5.174 | passed |
| nccl | sharded | 1,073,741,824 | 1 | 830.225 | 833.191 | 1293.31 | 5.173 | passed |
| nccl | sharded | 2,147,483,648 | 1 | 1656.190 | 1664.074 | 1296.64 | 5.187 | passed |
| nccl | replicated | 1,048,576 | 1 | 1.148 | 1.445 | 913.46 | 1.827 | passed |
| nccl | replicated | 2,097,152 | 1 | 1.923 | 2.150 | 1090.56 | 2.181 | passed |
| nccl | replicated | 4,194,304 | 1 | 3.526 | 3.987 | 1189.56 | 2.379 | passed |
| nccl | replicated | 8,388,608 | 1 | 6.970 | 7.320 | 1203.47 | 2.407 | passed |
| nccl | replicated | 16,777,216 | 1 | 13.409 | 14.232 | 1251.22 | 2.502 | passed |
| nccl | replicated | 33,554,432 | 1 | 26.850 | 27.725 | 1249.69 | 2.499 | passed |
| nccl | replicated | 67,108,864 | 1 | 52.992 | 53.937 | 1266.40 | 2.533 | passed |
| nccl | replicated | 134,217,728 | 1 | 104.273 | 105.232 | 1287.18 | 2.574 | passed |
| nccl | replicated | 268,435,456 | 1 | 208.551 | 208.931 | 1287.15 | 2.574 | passed |
| nccl | replicated | 536,870,912 | 1 | 417.539 | 421.106 | 1285.80 | 2.572 | passed |
| nccl | replicated | 1,073,741,824 | 1 | 836.736 | 841.169 | 1283.25 | 2.567 | passed |
| nccl | replicated | 2,147,483,648 | 1 | 1661.468 | 1668.317 | 1292.52 | 2.585 | passed |

Times are the slower rank for each aligned repetition. Barriers, setup,
materialization, JIT compilation, and handle retirement are outside the
CUDA-event interval. Effective collective payload is a workload-normalized
rate (one model payload for replicated, two for sharded), not physical link
bandwidth.
