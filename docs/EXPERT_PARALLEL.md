# Expert-parallel MoE layers

`mlops.expert_parallel` provides four ordinary PyTorch modules. The caller owns
the process group, compute device, communication buffers, optimizer and training
loop. These modules have no planning, spilling or trainer dependency.

| Module | Expert computation | Trainable state |
| --- | --- | --- |
| `QuackMoE` | Quack GEMMs; SonicMoE routing; MoonEP transport | Router, routed experts and optional shared expert |
| `QuackMoELoRA` | Same Quack computation with per-expert low-rank factors | LoRA factors only |
| `TEMoE` | Transformer Engine grouped GEMMs; MoonEP transport | Router, routed experts and optional shared expert |
| `TEMoELoRA` | Same Transformer Engine computation with low-rank factors | LoRA factors only |

## Installation

The tested stack uses Python 3.12, PyTorch 2.13 (CUDA 13), Triton 3.7.1 and
H100/SM90. Optional GPU dependencies are not part of the default MLOps
installation or its `providers` extra. Select the intended Python environment:

```bash
./scripts/setup_expert_parallel.sh --backend quack --python /path/to/python
# Or --backend te / --backend both.
```

For TE, this same command handles the CUDA setup automatically:

- Finds a complete CUDA toolkit matching the tested cuBLAS 13.6 runtime, including
  CUDA installations outside the current `PATH`.
- If none exists, downloads NVIDIA's CUDA 13.3.1 compiler/header/library wheels
  into `<python-prefix>/share/mlops/cuda/13.3.1`; no administrator access is needed.
- Builds the pinned TE extension with that toolkit and discovers the selected
  environment's cuDNN/NCCL headers.
- Installs cuBLAS 13.6.0.2 in that Python environment, so Torch and TE load the
  compatible runtime without shell library-path settings.
- Checks the loaded library and grouped-GEMM entry point in a fresh Python
  process with CUDA/library-path overrides removed. Rebuilds an incompatible
  cached TE extension automatically.

Users do not need to set `CUDA_HOME`, `PATH` or `LD_LIBRARY_PATH` for these layers.
A compatible `CUDA_HOME` is still accepted for build-tool selection. Installation
requires network access and a host C++ compiler; it requires neither a visible
GPU nor system-wide CUDA changes. Run setup on a networked build/head node when
compute nodes are offline. The NVIDIA driver remains a system prerequisite.

The script installs this repository and pinned upstream Quack, SonicMoE, MoonEP
and/or Transformer Engine dependencies. TE's build uses the chosen environment's
Torch installation. The optional TE stack upgrades cuBLAS beyond the exact pin
in Torch's CUDA 13.2 wheel metadata; `pip check` can report that metadata conflict.
The combination is validated by the checks below. Re-running setup restores this
runtime after another installer replaces CUDA dependencies; it does not alter
Torch's package metadata or add import-time library-loading patches.

The TE build defaults to SM90 and disables TE's separate NCCL EP backend because
these layers use MoonEP for communication. `NVTE_CUDA_ARCHS`,
`NVTE_WITH_NCCL_EP`, `MAX_JOBS` and `NVTE_BUILD_THREADS_PER_JOB` can override the
build settings. The installer supplies TE's build prerequisites before invoking
its non-isolated build. Installation does not require a visible GPU.
It also adds the selected environment's NVIDIA wheel headers to the compiler
search path, so pip-installed cuDNN/NCCL need no system-wide installation.

MoonEP's pinned upstream metadata requires Cutlass DSL 4.4.2, whereas the tested
Quack stack requires 4.7.1. The setup script installs DSL 4.7.1 and installs
MoonEP with `--no-deps`; installed MoonEP source remains unchanged. `pip check`
reports this known metadata mismatch. The `ep-quack` and `ep-te` extras install
the backend libraries; MoonEP must also be installed by the script above.

Small, source-checked compatibility patches are isolated in this package:
MoonEP singleton support, its planner compiler compatibility fix, Quack's
one-stage SM90 pipeline correction, and its
extended autotuning candidates. They apply in memory and never edit dependency
files. Unsupported dependency revisions produce explicit errors.

For DSL 4.7.1, the MoonEP planner alone uses PTXAS optimization level 2.
The default level produces an illegal address in stock MoonEP for configurations
including EP2, E192, top-k 4 and 65,536 tokens/rank. Communication kernels and
Quack/TE computation retain their usual compiler settings. The fix applies
automatically when constructing a buffer or loading either implementation;
applications do not need environment variables or modified MoonEP installations.

## Basic use

Initialize one process per GPU and an NCCL group before creating buffers. Load
the selected layer after selecting its compute device. A caller that installs a
custom device allocator must install it before loading the GPU implementation.
Importing the namespace and configuration classes alone loads no GPU backend.

```python
import torch
import torch.distributed as dist
from mlops.expert_parallel import QuackMoEConfig, create_buffer

# device and ep_group were explicitly selected/initialized by the application.
config = QuackMoEConfig(
    ep_size=dist.get_world_size(ep_group),
    num_experts=64, top_k=8, model_dim=7168, expert_hidden_dim=2048,
    compute_precision="bf16", weight_grad_dtype=torch.float32,
)
from mlops.expert_parallel import QuackMoE

buffer = create_buffer(config, tokens=32768, ep_group=ep_group)
layer = QuackMoE(config, ep_group, buffer=buffer, device=device)
try:
    x = torch.randn(32768, 7168, device=device, dtype=torch.bfloat16,
                    requires_grad=True)
    y = layer(x)
    y.backward(torch.randn_like(y))
finally:
    layer.close()
    buffer.destroy()
```

`create_buffer` is a convenience function. An already-created compatible MoonEP
buffer can be passed directly. The supplied buffer fixes the local token
capacity; input shape must match that capacity. Quack supports `num_chunks` and
`num_buffers` in its configuration. A multi-buffer configuration returns a
`ChunkBufferPool`. The caller destroys it once all borrowing layers are closed.

For a stack of Quack layers sharing a token buffer, set
`share_expert_banks=True`. Matching layers then share one set of weight
publication and replica-gradient banks too. Each layer keeps its own parameters
and publishes them before both forward and backward. Saved activations and
returned gradients remain independently owned. The same option supports
QuackMoELoRA; closing one layer leaves resources used by other layers alive.
The default single-layer path keeps parameter storage in its private bank.

Use `TEMoEConfig` and `TEMoE` for Transformer Engine. TE currently uses one chunk,
one buffer and BF16 activation transport. Quack supports BF16 or opt-in FP8
activation transport (`activation_transport="fp8"`). Both support BF16 and
`fp8_current` expert computation; TE additionally supports `fp8_block`.
MXFP8 is not exposed by these implementations.

## Configuration

Required geometry is `ep_size`, `num_experts`, `top_k`, `model_dim` and
`expert_hidden_dim`. Experts divide evenly across ranks. Routed feature and
hidden widths must be multiples of 128; top-k is at most 32.

| Option | Default | Meaning |
| --- | --- | --- |
| `compute_precision` | `"bf16"` | Routed weight/activation computation; FP8 choices are listed above |
| `weight_grad_dtype` | `torch.float32` | Returned routed/shared weight gradient dtype |
| `router_dtype` | `torch.float32` | Router weight and logit computation dtype |
| `router_weight_grad_dtype` | `torch.float32` | Returned router gradient dtype |
| `num_shared_experts` | `0` | Shared expert count, packed into one wider BF16 MLP |
| `shared_expert_dim` | Routed hidden width | Width of each shared expert |
| `renormalize_topk` | `True` | Renormalize selected routing probabilities to sum to one |
| `token_padding` | `128` | Round each received expert's token group for GEMM alignment |
| `num_comm_sms` | `32` | MoonEP communication SM setting |
| `init_std` | `0.02` | Standard deviation of initial weights |
| `gradient_output_mode` | `"owned"` | Transfer completed gradient ownership; `"copy"` retains an explicit copy |
| `profile_ranges` | `True` | Emit layer/phase NVTX ranges |
| Quack `num_chunks`, `num_buffers` | `1`, `1` | Equal token chunks and reusable caller-owned buffers |
| Quack `activation_transport` | `"bf16"` | Optional `"fp8"` transport with FP8 expert computation |
| Quack `share_expert_banks` | `False` | Reuse publication/reduction scratch across matching layers on the same token buffer |
| Quack `gemm_tuned` | `False` | Opt into autotuning instead of the fixed shape-dependent GEMM policy |
| TE `gemm_sm_margin` | `32` | SM headroom requested from Transformer Engine GEMMs |

Quack's fixed GEMM choices live in
[the expert policy](../src/mlops/expert_parallel/quack/experts/policy.py).
The BF16 fallback uses a 256x128 cooperative, 1x2-cluster weight-gradient tile
when expected tokens per expert exceed 1024; shorter reductions keep the
128x192 ping-pong setting. Expected load is
`tokens_per_chunk * top_k / (num_experts / ep_size)`, using the supplied buffer's
capacity. This is a performance heuristic, not a routing constraint: skewed and
empty groups still work. Existing measured-shape overrides take precedence.
Selection happens at initialization without inspecting GPU routing counts;
`gemm_tuned=True` bypasses these fixed choices.

BF16/FP32 are supported gradient/router dtypes. An FP32 router requires FP32
gradients. There is no layer-owned master-parameter or optimizer-state dtype.
Quack's expert GEMM and cross-rank reduction scratch is FP32 even when returned
gradients are BF16. Bank alignment applies to each rank's complete allocation,
with 128-row expert tiles; individual experts need not occupy full VMM pages.
`LoRAConfig` independently describes factor initialization, compute and gradient
precision. Full configuration definitions, including diagnostic/experimental
options, are in [Quack configuration](../src/mlops/expert_parallel/quack/config.py)
and [TE configuration](../src/mlops/expert_parallel/transformer_engine/config.py).
New callers should use the buffer's token capacity and `weight_grad_dtype`;
`tokens_per_rank` and `parameter_dtype` remain compatibility aliases.

## Precision, routing and ownership

`compute_precision` selects the routed expert compute representation;
`weight_grad_dtype` selects ordinary returned expert gradients. `router_dtype`
and `router_weight_grad_dtype` configure the router separately. Shared expert
computation is BF16. `num_shared_experts` and `shared_expert_dim` control its
shape. Masters and optimizer state belong to the caller.

The layer initializes only its own expert shards. `expert_parameters()` returns
those unique weights; `replicated_parameters()` returns router/shared weights.
The MoE backward gathers contributions to each expert's home rank. Callers
reduce gradients of replicated parameters separately, once per accumulation
window; `synchronize_replicated_gradients()` provides a simple SUM helper.
Corresponding expert shards in distinct replicated EP groups need their own
replica reductions, supplied by the training system.

Call `layer(x)` for its router, or supply `expert_ids` and `routing_weights` to
compare expert computation with fixed routes. The separate
`mlops.expert_parallel.reference` module uses only PyTorch for router and expert
reference equations. It supports fixed-route output/gradient checks.

FP8 weights wrap quantized payloads and scales. BF16 compute with FP32 returned
weight gradients also uses a wrapper, separating compute storage from the logical
gradient dtype. An external state manager must preserve
`__tensor_flatten__`/`__tensor_unflatten__` components and ordinary logical
gradients; treating the wrapper as one ordinary storage is insufficient.

These interfaces expose forward/backward computation. Quack's compute wrappers
deliberately reject ordinary in-place optimizer arithmetic: the caller must
update its master values and publish the compute representation. For ordinary
BF16 parameter tensors, choose `weight_grad_dtype=torch.bfloat16` (or
`LoRAConfig(gradient_dtype=torch.bfloat16)` for the trainable factors). The GPU
checks below validate computation and returned gradients, not arbitrary
optimizer implementations.

## LoRA

```python
from mlops.expert_parallel import LoRAConfig, QuackMoELoRA

layer = QuackMoELoRA(
    config, ep_group, buffer=buffer, device=device,
    lora=LoRAConfig(rank=32, alpha=32, compute_dtype=torch.bfloat16,
                    gradient_dtype=torch.float32),
)
trainable = list(layer.lora_parameters())
```

`TEMoELoRA` takes the same arguments with a `TEMoEConfig`. Both freeze the router,
shared expert and original routed weights. Each routed expert has a joint
packed gate/up LoRA projection and a separate down projection. Rank must be a
multiple of 16. BF16 factor computation and BF16/FP32 gradients are supported;
base expert precision is configured independently. The default rank and alpha
are both 32; scaling is alpha/rank.

## Package organization

All implementation code lives in `src/mlops/expert_parallel/`:

| Directory / file | Responsibility |
| --- | --- |
| `__init__.py`, `buffers.py` | Public exports and caller-owned communication buffer construction |
| `quack/` | Quack layer, operators, routing, expert math and compute-weight representations |
| `quack/pipeline/` | Chunk scheduling, buffer resources, transport and BF16/FP8 execution |
| `transformer_engine/` | TE layer, operators, communication and grouped expert math |
| Each backend's `lora/` | LoRA layer, factor storage and forward/backward integration |
| `lora.py`, `parameters.py`, `kernels/` | Shared LoRA configuration, BF16 compute weights and gradient reduction |
| `_compat/` | Isolated, source-checked MoonEP and Quack runtime patches |
| `reference/` | Independent PyTorch routing/expert math, without optional backend imports |

Backend code owns computation; application code owns optimizers, training,
benchmarking and planning. Both implementations use the same BF16 weight
representation and gradient-reduction helper. GPU packages load lazily when a
layer class is requested. The reference and configuration APIs remain CPU-safe.

## Testing

The default run needs no optional EP libraries or GPU and skips the GPU gate:

```bash
python -m pytest -q tests/expert_parallel
```

After installation, explicitly enable the H100/SM90 gate:

```bash
python -m pytest -q -s tests/expert_parallel --run-expert-parallel \
  --ep-backend both --ep-world-size 2 --ep-output /path/to/new-results
```

The gate runs 28 model configurations plus public-API coexistence, MoonEP
planning, shared expert-bank reuse and large-offset FP8 quantizer checks. It covers
full training and LoRA, save/recompute, BF16 and FP8-current, TE FP8-block, and
Quack's four-chunk/two-buffer path (including FP8 dispatch). LoRA checks both
eager and compiled execution. Full-training checks use compiled entrypoints.
The independent PyTorch reference uses identical expert assignments and checks
routing separately. Tests also cover skewed/empty expert groups, repeated calls,
nonzero LoRA factors, frozen weights, and configured gradients.
The large-offset quantizer check uses about 12 GiB per GPU and verifies that
matrices exceeding `2**31` elements use valid 64-bit read/write addresses.

Use `--ep-backend quack` or `te` for a single installed backend,
`--ep-world-size 1` for singleton EP, and pytest `-k` to select cases. Collection
alone never starts CUDA or distributed workers:

```bash
python -m pytest --collect-only -q tests/expert_parallel
python -m pytest -q -s tests/expert_parallel --run-expert-parallel \
  --ep-backend quack -k 'fp8_current and save and chunks4' \
  --ep-output /path/to/fp8-chunks
```

Each case starts fresh torchrun workers, prints and flushes start and completion
records, and saves `status.json`, `console.log`, per-rank worker logs and numerical
results in its own subdirectory. Existing case directories are never overwritten.
`--ep-timeout` bounds each worker group (600 seconds by default); failure or timeout
preserves its logs. An explicitly enabled gate fails clearly when dependencies,
GPU architecture or the requested device count are unavailable.

See the [test map and worker commands](../tests/expert_parallel/README.md) for
individual checks. FP8 tolerances compare against BF16 reference math; a passing
check does not imply bitwise equality. CPU isolation checks block optional
backend and legacy standalone imports so validation cannot silently use an
experimental checkout.
