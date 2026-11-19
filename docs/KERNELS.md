# Raw operation mechanics

This private directory contains low-level Torch, Triton, TileLang and CuTe DSL
mechanics. It is not a public API and contains no implementation selection,
fallback, or model state. Its only custom operators are the router GEMM's
autograd boundary (`router_op.py`, behind `router.routed_logits`) for
trace-visible MoE compositions and layers.

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

## Gated DeltaNet scratch lifetimes

`kernels/gated_delta_rule.py` composes the existing FLA 0.5.2 forward and backward
helpers for `linear_attention`. Forward releases the private WY factors `w` and
`u` after state reconstruction, before output assembly. Backward releases `u`
after state reconstruction, then `w`, `h`, `v_new` and `dh` after their last
gradient consumer, before WY backward allocates its temporaries. The same kernels
run in the same order with the same arithmetic and FLA backend selection. These
releases change neither saved forward residuals nor caller-owned inputs. The
adapted compositions include their source attribution and MIT license.

The existing contract remains packed sequences with 64-token recurrence tiles,
zero initial state and no final-state gradient. This is internal workspace
management, independent of checkpoint retention and execution-region chunking;
it introduces no public policy or model-specific path. The provider registration
source changes with the composition, so affected saved Programs and allocation
profiles must be rebuilt and reprofiled.

## KDA and tied-latent sparse attention

`src/mlops/kernels/kda/` contains the adapted FLA chunked algorithm, per-call
sequence metadata construction, and retained license/provenance. It does not
retain caller tensor views by identity. `src/mlops/kernels/sparse_mla/` contains
the attributed TileLang forward/backward kernels. Pooled index selection lives
in `src/mlops/kernels/pooled_indexer.py`: a Triton kernel that scores a chunk of
queries against every pooled key with the head reduction fused, and a selection
kernel that searches each query's FP32 scores for its cutoff and writes its
selection. `pooled_indexer_cudnn.py`, behind the SM90 priority provider
`cudnn.pooled_topk.sm90`, instead uses cuDNN 1.30's typed CuTe scorer
(including FP32 head weights) and radix top-k, with the combined score/radix
workspace capped at 256 MiB.

`pooled_indexer_runtime.py` supplies device-boundary pooling and index emission
around the shared Triton scorer/selector. Prefix sums and boundary searches stay
on the GPU; workspace sizes and launch geometry depend only on tensor shapes.
It supports reserved empty sequence slots and changing packing through compiled
and CUDA-graph calls. The cuDNN indexer currently consumes CPU boundaries.

FLA KDA uses fast dot precision by default; IEEE intra-chunk arithmetic is
an explicit implementation override. The optional cuLA SM90 training provider
replaces the fused WY backward stage while retaining call-local state metadata.
Its native-forward variant rebuilds intra-chunk matrices inside backward and
saves only the required prefix gates alongside caller inputs and metadata.

The shared backward composition releases its private WY factors, state and
intra-chunk matrices immediately after their final kernel consumer. This changes
allocation lifetimes without changing kernels or arithmetic; caller-owned inputs
remain valid. This is internal operator workspace management, separate from
compiler-managed execution-region chunking. The affected FLA and cuLA registration
sources have changed, so previously saved allocation profiles and executables
must be rebuilt and reprofiled before use with this implementation.

The final KDA intra-chunk gradient kernel accumulates in FP32 and stores `dQ`
and `dK` in their input dtypes: the cast happens in the store, avoiding full
FP32 output buffers and separate conversion launches.
The reverse gate scan and model gate derivative remain separate operations.

The existing SwiGLU kernels accept a compile-time clipping limit in both directions;
the default avoids the clipping branches. Gated RMSNorm threads the selected
activation into FLA's fused normalization kernels. These variations share their
existing registrations and explicit interfaces. See [ATTENTION.md](ATTENTION.md).


### mHC private kernels

`kernels/hyper_connection.py` contains four-stream mixing and fused post
forward/input-VJP/coefficient-VJP kernels. All allocations are invocation-local;
providers own custom-op registration. Post kernels read reduced-precision activations and small
FP32 coefficients, accumulate in FP32 and write reduced-precision activations;
there is no full FP32 activation workspace. The activation-coefficient option
uses the separate mixing primitive and a decomposable epilogue.
TileKernels calls are adapted in `providers/tilekernels/hyper_connection.py`.

`kernels/mhc_pre.py` provides the private projection/input-gradient kernel for
`providers/deepgemm/hyper_connection.py`. It combines the projection VJP,
squared-norm VJP and direct stream-collapse gradient without materializing
any full FP32 activation workspace. The public boundary remains `mhc_pre`.


`src/mlops/kernels/hybrid_layout.py` implements row-strided L2 normalization
and input-only packed causal-convolution/SiLU backward. These are internal
kernels of the FLA providers. The `causal_conv_silu_qkv` operation also uses
the input-gradient tile, with channel offsets and separate gradient tensors.
The convolution gradient preserves the FLA preactivation cast and sequence
resets while avoiding activation-sized scratch and unused weight gradients.

## Shared local expert math

`src/mlops/kernels/quack_moe/` owns local BF16/FP8 grouped GEMMs, clipped-SwiGLU
epilogues, row quantization/transpose, pointwise kernels and FP8 row payload
helpers. SonicMoE DP and Quack EP adapters import the same implementations.
`scale_rows.py` scales each row by its own FP32 factor and rounds once: routed
rows by their route scores, E4M3 rows by their descales. `swiglu_rows.py` also
holds the SwiGLU of a shared expert, whose gate and up projections are separate
tensors, and its gradient. The EP operators launch only these kernels and
library GEMMs and call no `torch.compile` when they run: a kernel Inductor
compiles inside an operator is outside the graph that captured the operator,
so a caller that freezes that graph's kernels cannot account for it, and every
process compiles it again.
It has no MoonEP or other communication dependency and does not select routed
experts. Group offsets, gather indices, scales and GEMM policy are explicit
caller inputs. The Quack runtime fingerprints this directory for its caches.
The private `expert_parallel/quack/experts/` and `kernels/` paths are aliases;
new kernels and callers belong here.
