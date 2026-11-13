# mlops

Tensor operations, LoRA modules and AdamW for PyTorch. Use ordinary autograd,
or call stateless forward/VJP functions when managing saved tensors yourself.
Implementations include native PyTorch and optional accelerated providers.

```python
import torch
import mlops

# Model code uses tensor operations; dispatch selects a supported implementation.
x = torch.randn(4, 128, device="cuda", requires_grad=True)
weight = torch.ones(128, device="cuda", requires_grad=True)
y = mlops.rms_norm(x, weight, eps=1e-5)
y.square().mean().backward()
```

The library owns operation math, dispatch and local optimizer updates. Model
architectures, training loops, distributed state ownership and execution
schedules belong to the caller.

## Install

```bash
./scripts/setup.sh
# Or install into an existing environment:
./scripts/setup.sh --python /path/to/python
```

See [installation](docs/INSTALLATION.md) for dependencies, optional providers
and expert-parallel setup. Importing mlops does not initialize the GPU.

## Use and extend

| Task | Guide |
| --- | --- |
| Find a tensor operation or its shape/dtype rules | [Operations](docs/OPS.md) |
| Select or diagnose an implementation | [Dispatch](docs/DISPATCH.md) |
| Add LoRA or configure AdamW | [LoRA](docs/LORA.md), [optimizers](docs/OPTIMIZERS.md) |
| Manage forward/backward state directly | [Explicit VJPs](docs/EXPLICIT_OPS.md) |
| Add an implementation or operation | [Extension guide](docs/EXTENDING.md) |
| Integrate a compiler | [Operator metadata](docs/COMPILER_METADATA.md) |

[All documentation](docs/README.md) · [API index](docs/API_REFERENCE.md) ·
[Package architecture](docs/ARCHITECTURE.md)

## Validate

```bash
python -m pytest -q tests
ruff check src/mlops tests
```

Optional multi-GPU tests have a separate [expert-parallel gate](docs/EXPERT_PARALLEL.md).
