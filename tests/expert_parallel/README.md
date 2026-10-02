# Expert-parallel correctness checks

These tests validate the installed `mlops.expert_parallel` package. No external
experiment checkout or training framework is needed. See the
[layer documentation](../../docs/EXPERT_PARALLEL.md) for installation and API use.

| Files | Coverage |
| --- | --- |
| `test_api.py` | Configuration validation and import isolation |
| `test_reference.py`, `test_lora_reference.py`, `test_math_cpu.py` | Independent PyTorch routing, expert and LoRA math |
| `test_setup.py` | Optional dependency/toolkit setup |
| `test_gate.py` | Case selection, device mapping, failure logs and worker timeouts |
| `test_gpu.py` | Explicitly enabled distributed GPU checks |
| `_gate.py`, `conftest.py` | GPU matrix, CLI options and process lifecycle |
| `gpu/` | Distributed workers and their reference/initialization helpers |

## Default and GPU runs

```bash
# CPU checks; GPU checks are skipped, even on a GPU machine.
python -m pytest -q tests/expert_parallel

# Two H100s, all installed backends, a fresh output directory.
python -m pytest -q -s tests/expert_parallel --run-expert-parallel \
  --ep-backend both --ep-world-size 2 --ep-output /path/to/new-results
```

GPU cases carry both `gpu` and `expert_parallel` markers. They also require the
explicit `--run-expert-parallel` flag; selecting a marker alone does not launch
workers. Use `--ep-backend quack|te|both`, `--ep-world-size 1|2|4|8`, and pytest
`-k` for narrower checks. `--collect-only` lists cases without initializing CUDA.
The 28-configuration matrix covers full/LoRA, save/recompute, all supported
precisions and Quack chunking; an additional check loads all four public classes
in one process. Configuration IDs show the selected backend, precision and mode.

Each case gets `status.json`, `console.log`, `workers/` rank logs and numerical
reports. An existing case directory is an error. If `--ep-output` is omitted,
results go into an ignored, timestamped `tests/expert_parallel/results/` directory.
Cases have a 600-second timeout, configurable with `--ep-timeout`; the harness
terminates the worker process group and reports the log tail on failure.

## Individual workers

For diagnosis, workers also run directly with torchrun:

```bash
torchrun --standalone --nproc-per-node=2 tests/expert_parallel/gpu/worker_quack.py \
  --precision bf16 --compiled --outdir /path/to/quack-save

mkdir -p /path/to/te-recompute
torchrun --standalone --nproc-per-node=2 tests/expert_parallel/gpu/worker_te.py \
  --precision fp8_block --compiled --recompute --num-shared-experts 1 \
  --output /path/to/te-recompute/result.json

torchrun --standalone --nproc-per-node=2 tests/expert_parallel/gpu/worker_lora.py \
  --backend quack --precision fp8_current --execution both --recompute \
  --num-chunks 4 --num-buffers 2 --activation-transport fp8 \
  --outdir /path/to/quack-lora-recompute
```

The common worker bootstrap selects the rank's device from
`CUDA_VISIBLE_DEVICES` before importing accelerator libraries, then creates its
NCCL group. Direct worker commands do not have the outer gate's timeout or
`status.json` bookkeeping. Prefer the pytest gate for complete validation runs.
