# mlops

`mlops` is a standalone operation and optimizer library for forward-only
PyTorch models and ordinary training callers. It provides three tensor-operation
layers over one set of implementation adapters:

- semantic functions such as `mlops.rms_norm(...)`, with normal autograd;
- stateless explicit `forward(...)`/`backward(...)` entry points under
  `mlops.explicit`;
- exact, per-operation implementation selection under `mlops.dispatch`.

It also provides standard `torch.optim.Optimizer` implementations under
`mlops.optim`, beginning with allocation-free mixed-precision AdamW. Optimizers
update local tensors; the caller or training engine owns gradient communication
and parameter/state sharding.

```python
optimizer = mlops.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)
```

These settings are held in host scalars, so a schedule writes into them rather
than rebuilding the optimizer -- see
[Settings a step can change](docs/OPTIMIZERS.md#settings-a-step-can-change).

The constructor follows the normal `torch.optim.AdamW` option and
parameter-group surface; [the optimizer reference](docs/OPTIMIZERS.md) lists
the admitted precision policy and unsupported semantic modes.

The package owns accelerated and native-PyTorch implementations, registered
custom-op/fake/autograd adapters, raw Triton kernels, cost hints, and gradcheck
helpers. It has no model, training-loop, planner, or execution-engine
dependency.

## Installation

```bash
./scripts/setup.sh
```

The script creates `.venv`, installs the supported PyTorch, installs mlops
editable with every implementation provider — flash-linear-attention,
liger-kernel, scattermoe, and tilelang — plus the FlashAttention-3 wheel
from the PyTorch index (activated automatically on Hopper GPUs) and the
development tools, and verifies the installation. To use an existing
virtual or Conda environment:

```bash
./scripts/setup.sh --python "$CONDA_PREFIX/bin/python"
```

A minimal install without providers remains:

```bash
python -m pip install --no-deps -e .
```

Optional implementation packages are detected lazily. Missing packages make
only their implementations unavailable.

## Documentation

- [API reference index](docs/API_REFERENCE.md)
- [Operation API reference](docs/OPS.md)
- [Package architecture and dispatch](docs/ARCHITECTURE.md)
- [Extending the package](docs/EXTENDING.md)
- [Provider implementation contract](docs/PROVIDERS.md)
- [Explicit entry points](docs/EXPLICIT_OPS.md)
- [Raw kernel boundary](docs/KERNELS.md)
- [Optimizer API reference](docs/OPTIMIZERS.md)
- [Expert-parallel layers and GPU validation](docs/EXPERT_PARALLEL.md)

## Repository layout

```text
mlops/
├── pyproject.toml
├── README.md
├── src/mlops/              importable package
│   ├── dispatch/           registry, resolution, overrides, cost/gradcheck APIs
│   ├── explicit/           public stateless forward/VJP entrypoints
│   ├── expert_parallel/    optional Quack/TE MoE and LoRA layers
│   ├── kernels/            private raw Torch/Triton mechanics
│   ├── optim/              PyTorch optimizers and functional/out=/in-place updates
│   └── providers/          exact implementation adapters
├── tests/                  standalone contract and numerical tests
├── benchmarks/             reproducible performance experiments and data
└── docs/                   API and contributor documentation
```

## Validation

```bash
python -m pytest -q tests
ruff check src/mlops tests
```

## Expert-parallel modules

`mlops.expert_parallel` contains optional QuackMoE, QuackMoELoRA, TEMoE and
TEMoELoRA modules with caller-owned groups and communication buffers.
The optional installer selects/downloads the matching CUDA build tools and sets
up runtime libraries in the selected environment:

```bash
./scripts/setup_expert_parallel.sh --python /path/to/python --backend both
```

See [expert-parallel configuration, installation and examples](docs/EXPERT_PARALLEL.md).
