# mlops

`mlops` is a standalone operation and optimizer library for forward-only
PyTorch models and ordinary training callers. It provides three tensor-operation
layers over one set of implementation adapters:

- semantic functions such as `mlops.rms_norm(...)`, with normal autograd;
- stateless explicit `forward(...)`/`backward(...)` entry points under
  `mlops.explicit`;
- exact, per-operation implementation selection under `mlops.dispatch`.

It also provides standard `torch.optim.Optimizer` implementations under
`mlops.optim`, beginning with allocation-free mixed-precision AdamW. The same
class supports local execution and optional ProcessGroup-backed replicated or
sharded-state execution.

```python
optimizer = mlops.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)
```

Supplying no group is local. Supplying a group defaults to sharded optimizer
state:

```python
optimizer = mlops.optim.AdamW(
    model.parameters(),
    replica_group=group,
    opt_state_strategy="sharded",
)
```

The same public optimizer supports NCCL and Gloo groups. Its 64 MiB default is
a hard per-collective payload cap, including for individual parameters larger
than one bucket. NCCL uses the efficient nonblocking completion path; Gloo is
a correct, slower portability path selected internally.

The constructor follows the normal `torch.optim.AdamW` option and
parameter-group surface; [the optimizer reference](docs/OPTIMIZERS.md) lists
the admitted precision policy and unsupported semantic modes.

The package owns accelerated and native-PyTorch implementations, registered
custom-op/fake/autograd adapters, raw Triton kernels, cost hints, and gradcheck
helpers. It has no model, training-loop, planner, or execution-engine
dependency.

## Installation

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

## Repository layout

```text
mlops/
├── pyproject.toml
├── README.md
├── src/mlops/              importable package
│   ├── dispatch/           registry, resolution, overrides, cost/gradcheck APIs
│   ├── explicit/           public stateless forward/VJP entrypoints
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

The real two-host optimizer canary is
`tests/distributed/adamw_worker.py`; see the
[distributed optimizer plan](docs/plans/DISTRIBUTED_ADAMW_PLAN.md).
The reproducible two-host size sweep is documented under
[`benchmarks/adamw_scaling`](benchmarks/adamw_scaling/README.md).
