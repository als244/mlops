# Universal serialized AdamW comparison

This report compares the current backend-portable, hard-bucket implementation
against the retained NCCL reference collected on the same Chicago/Tübingen
pair. Both use BF16 parameters, gradients, reduction, moments, and master
policy; 64 MiB buckets; three warmups; and ten measured repetitions.

The historical reference enabled cross-bucket overlap. That path was already
measured to improve large cases by only 0.2–1.2%, so a comparable serialization
penalty is expected. Positive deltas mean the current implementation is
slower.

| Parameters | Sharded old (ms) | Sharded current (ms) | Delta | Replicated old (ms) | Replicated current (ms) | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 2^20 | 1.102 | 1.195 | +8.47% | 1.022 | 1.152 | +12.73% |
| 2^21 | 1.961 | 2.017 | +2.85% | 1.936 | 1.935 | -0.07% |
| 2^22 | 3.678 | 3.604 | -2.01% | 3.768 | 3.505 | -6.97% |
| 2^23 | 7.119 | 7.128 | +0.12% | 6.943 | 6.876 | -0.96% |
| 2^24 | 13.779 | 13.800 | +0.16% | 13.617 | 13.455 | -1.19% |
| 2^25 | 27.085 | 27.214 | +0.48% | 26.667 | 26.931 | +0.99% |
| 2^26 | 53.937 | 53.730 | -0.38% | 52.687 | 53.910 | +2.32% |
| 2^27 | 106.596 | 105.730 | -0.81% | 104.969 | 105.415 | +0.42% |
| 2^28 | 210.713 | 212.206 | +0.71% | 208.672 | 211.227 | +1.22% |
| 2^29 | 421.976 | 424.598 | +0.62% | 416.613 | 420.615 | +0.96% |
| 2^30 | 846.424 | 850.159 | +0.44% | 830.966 | 859.159 | +3.39% |
| 2^31 | 1686.550 | 1699.560 | +0.77% | 1659.353 | 1683.400 | +1.45% |

The replicated `2^30` sample had a 927.7 ms p95 and was the only large-cell
outlier. A fresh 20-repetition run in
[`../universal_serial_64m_repeat_p30`](../universal_serial_64m_repeat_p30/RESULTS.md)
measured 843.525 ms median and 846.136 ms p95: +1.51% versus the overlapped
reference and +0.43% versus the historical serialized path. This is consistent
with the other large cells and identifies the original +3.39% as run noise.

## Workspace and allocator peak

At `2^31` parameters, workspace becomes independent of total model size:

| Strategy | Old declared workspace | Current declared workspace | Old live allocation | Current live allocation | Current latency delta |
|---|---:|---:|---:|---:|---:|
| Sharded | 8.00 GiB | 96 MiB | 20.00 GiB | 14.00 GiB | +0.77% |
| Replicated | 4.00 GiB | 64 MiB | 20.00 GiB | 16.00 GiB | +1.45% |

The current values are task-lifetime scratch reused across all compatible
buckets. The sharded path needs one 64 MiB reduction input plus one 32 MiB
rank-local output on a two-rank group. Persistent moments and a sharded update
copy are not mislabeled as workspace.

## Small-message regime

The fresh `2^12`–`2^19` results are stored in
[`../universal_serial_64m_small`](../universal_serial_64m_small/RESULTS.md):

| Parameters | Sharded median (ms) | Replicated median (ms) |
|---:|---:|---:|
| 2^12 | 0.341 | 0.300 |
| 2^13 | 0.360 | 0.289 |
| 2^14 | 0.357 | 0.303 |
| 2^15 | 0.369 | 0.298 |
| 2^16 | 0.442 | 0.294 |
| 2^17 | 0.414 | 0.312 |
| 2^18 | 0.452 | 0.385 |
| 2^19 | 0.630 | 0.631 |

There is no retained pre-refactor dataset below `2^20`, so these rows establish
the reusable baseline rather than claiming a before/after delta. They show a
roughly 0.3–0.4 ms fixed distributed-step floor before payload bandwidth
dominates.

## Conclusion

The simplification does not introduce a material NCCL throughput regression.
At large sizes it tracks the prior serialized implementation within about
0.5%, and the difference from the original default is the already-measured
cost of removing low-value overlap. In exchange, the public flag and custom
stream machinery disappear, Gloo and NCCL use one algorithm, and declared
workspace falls from model-sized to bucket-sized.
