# Historical distributed optimizer measurements

The `results/` directory records experiments with the former MLOps distributed
AdamW runtime. That runtime and its launch scripts have been removed. These
measurements describe historical code and are retained as evidence, rather than
as benchmarks of the current local optimizer. The source and original commands
remain available in Git history before this removal.

Current `mlops.optim.AdamW` updates local tensors. A training engine owns
communication and sharding; see [the optimizer API](../../docs/OPTIMIZERS.md).
