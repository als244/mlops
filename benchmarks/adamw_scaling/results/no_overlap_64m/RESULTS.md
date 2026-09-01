# AdamW scaling results

| Backend | Strategy | Parameters | Buckets | Median (ms) | P95 (ms) | Mparam/s | Effective collective payload (GB/s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| nccl | sharded | 33,554,432 | 1 | 26.719 | 28.264 | 1255.81 | 5.023 | passed |
| nccl | sharded | 67,108,864 | 2 | 54.098 | 54.548 | 1240.51 | 4.962 | passed |
| nccl | sharded | 134,217,728 | 4 | 106.802 | 107.963 | 1256.70 | 5.027 | passed |
| nccl | sharded | 268,435,456 | 8 | 212.141 | 213.149 | 1265.37 | 5.061 | passed |
| nccl | sharded | 536,870,912 | 16 | 424.852 | 427.655 | 1263.66 | 5.055 | passed |
| nccl | sharded | 1,073,741,824 | 32 | 848.644 | 851.426 | 1265.24 | 5.061 | passed |
| nccl | sharded | 2,147,483,648 | 64 | 1698.369 | 1702.585 | 1264.44 | 5.058 | passed |
| nccl | replicated | 33,554,432 | 1 | 26.652 | 27.638 | 1259.00 | 2.518 | passed |
| nccl | replicated | 67,108,864 | 2 | 53.563 | 54.129 | 1252.90 | 2.506 | passed |
| nccl | replicated | 134,217,728 | 4 | 104.919 | 105.869 | 1279.25 | 2.558 | passed |
| nccl | replicated | 268,435,456 | 8 | 210.402 | 211.516 | 1275.82 | 2.552 | passed |
| nccl | replicated | 536,870,912 | 16 | 420.138 | 422.946 | 1277.84 | 2.556 | passed |
| nccl | replicated | 1,073,741,824 | 32 | 839.887 | 842.467 | 1278.44 | 2.557 | passed |
| nccl | replicated | 2,147,483,648 | 64 | 1679.959 | 1682.661 | 1278.30 | 2.557 | passed |

Times are the slower rank for each aligned repetition. Barriers, setup,
materialization, JIT compilation, and handle retirement are outside the
CUDA-event interval. Effective collective payload is a workload-normalized
rate (one model payload for replicated, two for sharded), not physical link
bandwidth.
