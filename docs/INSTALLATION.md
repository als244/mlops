# Installation

Requires Python 3.12 or newer and PyTorch 2.13 or newer.


```bash
./scripts/setup.sh
```

The script creates `.venv`, installs the supported PyTorch, installs mlops
editable with cuDNN Frontend, FlashAttention-3, flash-linear-attention,
Quack, SonicMoE, TileLang, TileKernels and development tools, then verifies the installation.
It also builds pinned DeepGEMM for fused mHC preprocessing; `--no-deepgemm`
skips it. This build needs a C++20 compiler with `<format>` and elfutils headers;
see [provider setup](INSTALLATION.md#deepgemm-mhc-preprocessing-setup).
It also builds FlashMLA's Hopper sparse-attention forward at a pinned commit for
the Hopper sparse latent attention provider; `--no-flashmla` skips it, and that
provider then keeps its TileLang forward.
On Hopper it also builds cuLA's pinned native KDA forward and fused WY backward
provider, including patches that skip empty training sequences in forward and
unused chunk rows in backward; `--no-cula` skips that build. Both keep sequence
metadata on the GPU. Re-run `scripts/setup_cula_sm90.sh` to update
an existing cuLA installation. Set `MLOPS_BUILD_DIR` to place source
builds outside a quota-limited home directory. See [the attention guide](ATTENTION.md)
for provider geometry, precision and fallback behavior. FlashAttention-3 is
installed whatever the local GPU and activated automatically on Hopper GPUs.
To use an existing virtual or Conda environment:

```bash
./scripts/setup.sh --python "$CONDA_PREFIX/bin/python"
```

A minimal install without providers remains:

```bash
python -m pip install --no-deps -e .
```

Optional implementation packages are detected lazily. Missing packages make
only their implementations unavailable.

## Expert-parallel modules

`mlops.expert_parallel` contains optional QuackMoE, QuackMoELoRA, TEMoE and
TEMoELoRA modules with caller-owned groups and communication buffers.
Their backends are opt-in: setup installs them after everything else, which
keeps the cuBLAS that Transformer Engine needs. Running setup without the
flag restores the cuBLAS that torch pins, so pass it every time for an
environment that uses TE:

```bash
./scripts/setup.sh --python /path/to/python --expert-parallel both
```

The optional installer it runs selects/downloads the matching CUDA build tools
and sets up runtime libraries in the selected environment. It also adds the
backends to an environment that is already set up:

```bash
./scripts/setup_expert_parallel.sh --python /path/to/python --backend both
```

See [expert-parallel configuration, installation and examples](EXPERT_PARALLEL.md).

## DeepGEMM mHC preprocessing setup

The complete setup builds pinned DeepGEMM revision
`5081f8513af57e61696de4f4f04b0e707ffb503e` with its submodules against the
selected PyTorch/CUDA environment. `--no-deepgemm` skips this source build.
For an existing environment, run:

```bash
./scripts/setup_deepgemm.sh --python "$CONDA_PREFIX/bin/python"
```

The helper is idempotent; `--force` rebuilds. It requires CUDA >=12.9 and a
C++20 host compiler supporting `<format>` (for example GCC 13+). Set `CXX`
when the default compiler is older. DeepJIT also needs elfutils/libelf headers:
system development packages or `MLOPS_ELFUTILS_ROOT` are accepted. If neither is
available, the helper creates an isolated Conda build-dependency prefix.
`MLOPS_BUILD_DIR` controls source/build placement. TileKernels and TileLang are
installed by the ordinary `providers` extra, at their pinned/minimum versions.
DeepGEMM is built separately to avoid replacing the selected PyTorch dependency.
