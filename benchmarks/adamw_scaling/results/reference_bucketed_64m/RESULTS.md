# AdamW scaling results

| Backend | Strategy | Parameters | Buckets | Median (ms) | P95 (ms) | Mparam/s | Effective collective payload (GB/s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| nccl | sharded | 1,048,576 | 1 | 1.102 | 1.571 | 951.73 | 3.807 | passed |
| nccl | sharded | 2,097,152 | 1 | 1.961 | 2.044 | 1069.22 | 4.277 | passed |
| nccl | sharded | 4,194,304 | 1 | 3.678 | 3.863 | 1140.30 | 4.561 | passed |
| nccl | sharded | 8,388,608 | 1 | 7.119 | 7.998 | 1178.28 | 4.713 | passed |
| nccl | sharded | 16,777,216 | 1 | 13.779 | 14.232 | 1217.62 | 4.870 | passed |
| nccl | sharded | 33,554,432 | 1 | 27.085 | 27.680 | 1238.86 | 4.955 | passed |
| nccl | sharded | 67,108,864 | 2 | 53.937 | 54.672 | 1244.21 | 4.977 | passed |
| nccl | sharded | 134,217,728 | 4 | 106.596 | 107.096 | 1259.13 | 5.037 | passed |
| nccl | sharded | 268,435,456 | 8 | 210.713 | 212.399 | 1273.94 | 5.096 | passed |
| nccl | sharded | 536,870,912 | 16 | 421.976 | 424.741 | 1272.28 | 5.089 | passed |
| nccl | sharded | 1,073,741,824 | 32 | 846.424 | 848.719 | 1268.56 | 5.074 | passed |
| nccl | sharded | 2,147,483,648 | 64 | 1686.550 | 1690.621 | 1273.30 | 5.093 | passed |
| nccl | replicated | 1,048,576 | 1 | 1.022 | 1.243 | 1026.12 | 2.052 | passed |
| nccl | replicated | 2,097,152 | 1 | 1.936 | 2.069 | 1083.21 | 2.166 | passed |
| nccl | replicated | 4,194,304 | 1 | 3.768 | 3.857 | 1113.19 | 2.226 | passed |
| nccl | replicated | 8,388,608 | 1 | 6.943 | 7.247 | 1208.20 | 2.416 | passed |
| nccl | replicated | 16,777,216 | 1 | 13.617 | 14.572 | 1232.06 | 2.464 | passed |
| nccl | replicated | 33,554,432 | 1 | 26.667 | 28.031 | 1258.29 | 2.517 | passed |
| nccl | replicated | 67,108,864 | 2 | 52.687 | 53.083 | 1273.73 | 2.547 | passed |
| nccl | replicated | 134,217,728 | 4 | 104.969 | 105.675 | 1278.64 | 2.557 | passed |
| nccl | replicated | 268,435,456 | 8 | 208.672 | 209.529 | 1286.40 | 2.573 | passed |
| nccl | replicated | 536,870,912 | 16 | 416.613 | 418.027 | 1288.66 | 2.577 | passed |
| nccl | replicated | 1,073,741,824 | 32 | 830.966 | 834.325 | 1292.16 | 2.584 | passed |
| nccl | replicated | 2,147,483,648 | 64 | 1659.353 | 1662.070 | 1294.17 | 2.588 | passed |

Times are the slower rank for each aligned repetition. Barriers, setup,
materialization, JIT compilation, and handle retirement are outside the
CUDA-event interval. Effective collective payload is a workload-normalized
rate (one model payload for replicated, two for sharded), not physical link
bandwidth.
