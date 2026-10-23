# Raw operation mechanics

This private directory contains low-level Torch/Triton mechanics. It is not a
public API and contains no implementation selection, fallback, custom-op
registration, or model state.

Rules:

- one file owns one operation or one tightly coupled subsystem;
- every tensor and policy is an explicit function argument;
- raw functions never inspect the registry or choose another backend;
- caller inputs are immutable unless a public schema truthfully declares a
  mutation (all current differentiable schemas are functional);
- local outputs and invocation-local scratch may be allocated normally;
- no tensor is cached globally; reusable tables/prepacked values are caller
  inputs or module/artifact state; and
- model and block files never import this directory.

Implementation adapters in `src/mlops/providers/<source>/` call these
functions directly and own stable custom-op targets, fake functions, and
`register_autograd`. The public explicit façade calls the same raw functions
after exact implementation resolution.

To add raw mechanics, document dtype/layout/shape assumptions in the function,
return fresh outputs, expose any VJP residual explicitly, and validate it first
through its owning implementation adapter with opcheck, VJP parity, repeated
backward, immutability, and capture tests.

## KDA and tied-latent sparse attention

`src/mlops/kernels/kda/` contains the adapted FLA chunked algorithm, per-call
sequence metadata construction, and retained license/provenance. It does not
retain caller tensor views by identity. `src/mlops/kernels/sparse_mla/` contains
the attributed TileLang forward/backward kernels. Pooled index selection lives
in `src/mlops/kernels/pooled_indexer.py`: a Triton kernel that scores a chunk of
queries against every pooled key with the head reduction fused, and a selection
kernel that searches each query's FP32 scores for its cutoff and writes its
selection. The SM90 priority provider in `pooled_indexer_cudnn.py` instead
uses cuDNN 1.30's typed CuTe scorer (including FP32 head weights) and radix
top-k, with the combined score/radix workspace capped at 256 MiB.

FLA KDA uses fast dot precision by default; IEEE intra-chunk arithmetic is
an explicit implementation override. The optional cuLA SM90 training provider
replaces the fused WY backward stage while retaining call-local state metadata.
Its native-forward variant rebuilds intra-chunk matrices inside backward and
saves only the required prefix gates alongside caller inputs and metadata.

The existing SwiGLU kernels accept a compile-time clipping limit in both directions;
the default avoids the clipping branches. Gated RMSNorm threads the selected
activation into FLA's fused normalization kernels. These variations share their
existing registrations and explicit interfaces. See [ATTENTION.md](ATTENTION.md).
