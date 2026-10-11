# Sequential experts

`mlops.sequential_moe` provides `SequentialMoEConfig`, `SequentialMoE`, and `SequentialMoELoRA`. Each routed expert is its own module and opaque functional operation, with at most one expert's dense GEMM workspace live inside that operation.

Storage format, GEMM precision, activation dtype, activation accumulation dtype, factor dtype and gradient dtype are separate configuration choices. Default chunk capacity is 8192 rows, rounding work to 256 rows; incomplete chunks are sliced to their rounded occupied size. CPU route bounds are produced once by the routing stage.

```python
from mlops.sequential_moe import SequentialMoE, SequentialMoEConfig

model = SequentialMoE(
    SequentialMoEConfig(
        hidden_size=4096, expert_size=2048, num_experts=288, top_k=8,
        weight_storage="fp8", gemm_precision="bf16",
    ),
    device="meta",
)
```

Meta construction declares physical weight components without allocating or quantizing a full expert. An importer must initialize these components before execution. CUDA execution remains required.

## Checkpoints

`mlops.sequential_moe.precision.checkpoint.HFWeights` reads local safetensors incrementally, including explicit per-projection mixed NVFP4/FP8 recipes. `projection(..., meta=True)` validates component geometry without reading weight payloads. Gate/up packing preserves independently stored scales. NVFP4's inverse global weight scale is converted to a dequantization multiplier.

BF16/FP16 GEMMs can decode the supported dense, FP8, NVFP4, MXFP4 and symmetric INT4 layouts. FP8 GEMMs use the existing block-quantized recipe. NVFP4 GEMMs currently require 16x16 weight scales; the external 1x16 checkpoint layout must use BF16/FP16 GEMMs until its low-precision recipe is implemented. AWQ/GPTQ/NF4 encodings are not inferred from a four-bit dtype.

`FrozenLinear` uses the same physical formats for frozen dense projections. It permits input gradients and currently uses BF16/FP16 GEMMs.

## Ownership and integration

Caller-supplied initial accumulators can represent a shared expert computed
outside the sequential layer. Inputs, weights/scales, route bounds and gradient
outputs are explicit.

The functional expert operation copies its output accumulator; backward allocates
its gradient destinations. Account for these allocations when estimating memory
and transfer costs.

Model composition, architecture-specific checkpoint mappings and whole-model validation belong to the caller. Operation-level correctness does not establish numerical parity for a complete checkpoint or an untested compute recipe.

## Frozen dense projections with LoRA

`FrozenLoRALinear` adds trainable factors without retaining a dense base copy:

```python
import torch
from mlops.lora import LoRAConfig
from mlops.sequential_moe.precision.linear import FrozenLinear, FrozenLoRALinear

projection = FrozenLoRALinear(
    FrozenLinear(parts, spec, gemm_precision="bf16"),
    LoRAConfig(rank=32, alpha=32, factor_dtype=torch.float32),
)
```

Factors compute at the input activation dtype; gradients follow factor storage
dtype. A uses Kaiming initialization and B starts at zero. Base components stay
frozen and retain their checkpoint representation. BF16 and FP16 GEMMs are
supported; compressed weight storage is independent of GEMM precision.
