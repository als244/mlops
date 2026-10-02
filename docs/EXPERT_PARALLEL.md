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
MoonEP singleton support, Quack's one-stage SM90 pipeline correction, and its
extended autotuning candidates. They apply in memory and never edit dependency
files. Unsupported dependency revisions produce explicit errors.

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
| Quack `gemm_tuned` | `False` | Opt into autotuning instead of the fixed shape-dependent GEMM policy |
| TE `gemm_sm_margin` | `32` | SM headroom requested from Transformer Engine GEMMs |

BF16/FP32 are supported gradient/router dtypes. An FP32 router requires FP32
gradients. There is no layer-owned master-parameter or optimizer-state dtype.
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

## Testing and current validation

CPU reference/configuration checks:

```bash
python -m pytest tests/expert_parallel -q
```

GPU tests use one process per rank and an independent PyTorch reference. They
check the router separately, then hold routes fixed for expert output and gradient
comparisons. Run these after installing the corresponding optional backend:

```bash
# Full training: --recompute checkpoints the forward; omit it to save activations.
torchrun --standalone --nproc-per-node=2 tests/expert_parallel/gpu_quack.py \
  --precision bf16 --compiled --outdir /path/to/results/quack-save
torchrun --standalone --nproc-per-node=2 tests/expert_parallel/gpu_te.py \
  --precision fp8_current --compiled --recompute --output /path/to/results/te.json

# LoRA: check eager and torch.compile, zero/nonzero B, and frozen base weights.
torchrun --standalone --nproc-per-node=2 tests/expert_parallel/gpu_quack_lora.py \
  --precision fp8_current --execution both --outdir /path/to/results/quack-lora
torchrun --standalone --nproc-per-node=2 tests/expert_parallel/gpu_te_lora.py \
  --precision bf16 --execution both --recompute --outdir /path/to/results/te-lora
```

Create the parent directory for `gpu_te.py --output` first. TE tests also accept
`fp8_block`; both LoRA tests accept `--recompute`. Quack tests can exercise
`--num-chunks 4 --num-buffers 2` and, with FP8 compute,
`--activation-transport fp8`. Use `--nproc-per-node=1` to test singleton EP.
The scripts report per-case output/gradient relative RMS error and fail on
nonfinite values or mismatches. FP8 tolerances account for comparison with a BF16
reference; passing them does not mean bitwise equality.

The default MLOps installation and CPU tests work without these GPU dependencies.
The CPU import check explicitly blocks optional backend imports. Wheel validation
also blocks legacy standalone module imports, so tests cannot silently fall back
to an experimental checkout.
