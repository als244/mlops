# AdamW backend comparison

This directory contains the development evidence used to select one public
`mlops.optim.AdamW` implementation for NCCL and Gloo ProcessGroups.

## Completion behavior

| Backend / collective | CUDA operation | `block_current_stream()` | Portable completion |
|---|---|---|---|
| NCCL all-reduce, reduce-scatter, all-gather | yes | yes | current-stream bridge |
| Gloo all-reduce | yes | yes in this build | `Work.wait()` |
| Gloo all-gather | yes | yes in this build | `Work.wait()` |
| Gloo reduce-scatter | yes | no (`Work::getFuture` is unavailable) | `Work.wait()` |

The support-matrix interpretation is therefore straightforward: Gloo CUDA
reduce-scatter is supported. Its PyTorch 2.13 `Work` object simply lacks the
future needed by the nonblocking current-stream helper. Calling that helper
and then trying to recover is unsafe; a pure host wait succeeds. The optimizer
uses the NCCL stream bridge only for NCCL and the portable wait for every
other backend.

## Numerical acceptance

[`nccl_correctness_v3.json`](nccl_correctness_v3.json) and
[`gloo_correctness_v3.json`](gloo_correctness_v3.json) each cover 100 updates
for:

- replicated and sharded optimizer state;
- sum and mean reduction;
- mixed BF16/FP32 parameter, gradient, reduction, moment, and master policy;
- shared and rank-local parameter slices;
- checkpoint save and restore.

Every parameter, master, first moment, and second moment has relative L2 error
exactly zero against the independent full-state reference. The corresponding
per-host logs are retained beside the JSON files.

## Performance and workspace

The controlled one-bucket/no-overlap measurements are retained under
`nccl_single_no_overlap/` and `gloo_single_no_overlap/`. At large sizes on the
Chicago RTX 5090 / Tübingen RTX 3090 25 GbE pair, complete-step achieved link
rates were approximately:

| Backend | Sharded | Replicated |
|---|---:|---:|
| NCCL | 20.5–20.8 Gb/s | 20.5–20.7 Gb/s |
| Gloo | about 5.0 Gb/s through 1B parameters | about 13.2 Gb/s |

One unbounded 2B-parameter Gloo reduce-scatter required an additional full
input-sized provider temporary and exhausted the 24 GiB rank. The bounded
64 MiB form in `gloo_bounded_2p31/` completed at 3.89 Gb/s. This is why the
current optimizer splits individual parameters and treats `bucket_bytes` as a
hard collective-payload cap even though the earlier NCCL performance ablation
did not favor smaller buckets.

The final current-code canary in
[`../universal_gloo_single_2p31`](../universal_gloo_single_2p31/RESULTS.md)
uses one model-visible 2-billion-element parameter. It was automatically split
into 64 buckets, passed without OOM, used 96 MiB of declared scratch, and
measured 8.066 s / 4.26 Gb/s median over three repetitions.

The current universal serialized implementation and its pre-refactor NCCL
reference live one directory above in `universal_serial_64m/` and
`reference_bucketed_64m/` respectively.

## Historical directories

`gloo_replicated_scaling/` is an intentionally retained incomplete diagnostic
from before the completion adapter was corrected; it stopped when the old path
accumulated asynchronous Work objects. It is not current acceptance evidence.
Directories named `*_single_no_overlap` and `gloo_bounded_2p31/` are controlled
ablations used to isolate completion and payload-size effects. The V3
correctness files and `universal_*` scaling directories are the authoritative
current results.
