# AdamW scaling results

| Backend | Strategy | Parameters | Buckets | Median (ms) | P95 (ms) | Mparam/s | Achieved link rate (Gb/s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| nccl | sharded | 1,048,576 | 1 | 1.195 | 1.834 | 877.44 | 14.039 | passed |
| nccl | sharded | 2,097,152 | 1 | 2.017 | 2.572 | 1039.59 | 16.634 | passed |
| nccl | sharded | 4,194,304 | 1 | 3.604 | 3.763 | 1163.66 | 18.619 | passed |
| nccl | sharded | 8,388,608 | 1 | 7.128 | 7.649 | 1176.84 | 18.829 | passed |
| nccl | sharded | 16,777,216 | 1 | 13.800 | 14.125 | 1215.70 | 19.451 | passed |
| nccl | sharded | 33,554,432 | 1 | 27.214 | 28.225 | 1232.96 | 19.727 | passed |
| nccl | sharded | 67,108,864 | 2 | 53.730 | 54.470 | 1249.00 | 19.984 | passed |
| nccl | sharded | 134,217,728 | 4 | 105.730 | 107.484 | 1269.44 | 20.311 | passed |
| nccl | sharded | 268,435,456 | 8 | 212.206 | 214.546 | 1264.98 | 20.240 | passed |
| nccl | sharded | 536,870,912 | 16 | 424.598 | 425.576 | 1264.42 | 20.231 | passed |
| nccl | sharded | 1,073,741,824 | 32 | 850.159 | 853.082 | 1262.99 | 20.208 | passed |
| nccl | sharded | 2,147,483,648 | 64 | 1699.560 | 1703.129 | 1263.55 | 20.217 | passed |
| nccl | replicated | 1,048,576 | 1 | 1.152 | 1.654 | 910.22 | 14.564 | passed |
| nccl | replicated | 2,097,152 | 1 | 1.935 | 1.954 | 1083.96 | 17.343 | passed |
| nccl | replicated | 4,194,304 | 1 | 3.505 | 3.616 | 1196.61 | 19.146 | passed |
| nccl | replicated | 8,388,608 | 1 | 6.876 | 7.219 | 1219.97 | 19.520 | passed |
| nccl | replicated | 16,777,216 | 1 | 13.455 | 14.274 | 1246.94 | 19.951 | passed |
| nccl | replicated | 33,554,432 | 1 | 26.931 | 27.954 | 1245.94 | 19.935 | passed |
| nccl | replicated | 67,108,864 | 2 | 53.910 | 54.961 | 1244.83 | 19.917 | passed |
| nccl | replicated | 134,217,728 | 4 | 105.415 | 106.315 | 1273.23 | 20.372 | passed |
| nccl | replicated | 268,435,456 | 8 | 211.227 | 213.148 | 1270.84 | 20.333 | passed |
| nccl | replicated | 536,870,912 | 16 | 420.615 | 422.721 | 1276.39 | 20.422 | passed |
| nccl | replicated | 1,073,741,824 | 32 | 859.159 | 927.675 | 1249.76 | 19.996 | passed |
| nccl | replicated | 2,147,483,648 | 64 | 1683.400 | 1686.781 | 1275.68 | 20.411 | passed |

Times are the slower rank for each aligned repetition. Barriers, setup,
materialization, JIT compilation, and handle retirement are outside the
CUDA-event interval. Achieved link rate uses one model payload per rank
for this two-rank topology; it is a comparable end-to-end rate, not a
claim about the provider's internal collective algorithm.
