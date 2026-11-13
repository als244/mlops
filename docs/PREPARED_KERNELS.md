# Prepared external kernels

Preparation saves selected external kernel implementations with a caller's
compiled task, then restores them without tuning or compiling during execution.
MLOps owns provider-specific caches, source checks, binaries and module lifetimes.
The consuming compiler owns its adapter, artifact store, profiling and admission.
Importing MLOps does not activate preparation or initialize CUDA.

Obtain a factory description through
[`mlops.operator_metadata`](COMPILER_METADATA.md#prepared-implementations).
Preparation freezes existing dispatch choices; it changes neither chunking nor
retention policies.

## Caller lifecycle

Import the declaration's `module:function` factory and call it with
`phase="build"` or `phase="load"`, a CUDA device and `options`.

1. **Capture during finalization.** Surround the existing task invocation with
   `builder.capture()`. Export the yielded record through
   `builder.export(record)` to obtain a JSON manifest and digest-keyed blobs.
   Do not run a mutating task again merely to collect binaries.
2. **Archive with the executable.** Include the preparation identity in profile
   compatibility. Changed implementations require fresh measurements.
3. **Load before admission.** Install the caller's allocator, then call
   `loader.bind(manifest, read_blob)`. Validation and device-module loading occur
   here; account for their external footprint before sealing the memory budget.
4. **Activate for each task.** Execute inside `binding.activate()` with the
   owning CUDA context current. Missing specializations must refuse, not tune
   or use an eager-cache fallback.
5. **Drain and close.** Quiesce streams before closing the builder/loader.
   Closing while active is refused. Device modules are released; native Python
   launcher extensions have process lifetime.

Bindings cover actual full/tail and accumulation specializations. Compatible
bindings share loaded modules, but each retains its own selected configurations
and host launch recipes. Partial loads release newly owned handles; closing an
owner invalidates its bindings. Wrong-context use and use after close refuse.

## Supported preparation

| Declaration | Saved content | Limits |
|---|---|---|
| `sonicmoe_preparation()` | BF16 Sonic Quack objects/configurations, external Triton CUBINs, native launcher and occupancy helper | No FP8, EP or unrelated attention kernels |
| `operator_preparations(dtype=torch.bfloat16)` | Availability-based Sonic preparation plus Hopper sparse-forward backend/LSE selection | Other dtypes return no preparations; build checks device support |

Availability-based Sonic preparation records an empty selection on unsupported
devices without changing dispatch. Loading follows the saved selection, never a
new default. An explicitly required Sonic implementation should use its direct
declaration.

Sparse-forward preparation records whether a task chose FlashMLA or TileLang
and FlashMLA's measured LSE conversion. Loading checks source and installed
FlashMLA Python/extension bytes, then binds that selection without a GPU probe.
A saved TileLang choice stays TileLang if FlashMLA is later installed. Tasks
without sparse calls need no optional backend. This saves selection and LSE
semantics, not attention binaries or the cuDNN backward compiler.

Prebuilt PyTorch/cuBLAS libraries may still initialize on first use. Their
footprint and first-call latency are distinct from prepared provider kernels.
A successful kernel replay does not qualify a full saved training computation
or its distributed behavior; the consuming project's tests must establish that.

## Compatibility and selection

Manifests validate provider and transitive source identities, target, host ABI,
schema, specialization and binary digests. Only the current schema is accepted.
Changed sources require rebuilding; ordinary eager caches cannot rescue an
incompatible artifact. All records and bytes validate before loading begins.
These are trusted compiled-code artifacts: digests detect corruption, not a
malicious author.

Capture observes warm cache hits as well as cold compilation. Quack benchmark
candidates are excluded from inventory; the winner's final invocation supplies
its binaries. Benchmarking during prepared replay refuses. Generated epilogues
and their outer host caches are scoped together so warm calls cannot hide a
selected configuration. The caller need not understand Quack cache layouts or
source hashes.

For controlled numerical comparisons, record Quack choices with
`mlops.quack_runtime.capture_choices()` and pin them with
`mlops.quack_runtime.pin_choices(selection)`. Nested task captures retain the
pin and record only actual calls. Missing or conflicting selections refuse
before tuning; the scope leaves mutable winner caches untouched and restores
ordinary behavior on exit. This pins Quack GEMMs only; a complete comparison
must also control any Triton autotuners.

## Triton implementation

`triton_runtime._scope.TritonCalls` observes explicit plain JITFunctions through
device-cache lookups, preserving imported aliases and thread/nested-scope state.
Each binding has its own specialization map. `_objects.TritonObjects` exports
typed signatures, constants, attributes, launch metadata and SHA-256-addressed
CUBINs. Export reproduces Triton's compilation hash from AST, options, compiler,
target and environment; stale eager binaries cannot be archived.

Restore does not evaluate cache-key strings, deserialize arbitrary Python
objects or invoke `preload()`/`compile()`. It resolves bundled libdevice locally
and validates content instead of following a manifest's absolute path.
The loader targets Triton 3.7.1 with pinned private loader/cache interfaces.
Compiler/IR overrides, instrumentation, custom linked libraries, descriptors
and arbitrary object/JIT-function constexprs are unsupported.

`TritonModulePool` shares host launchers for exact records and CUDA modules for
identical CUBIN/launch metadata. Loading is eager, including native-driver setup;
invocation still allocates launcher scratch and requires profiling. The explicit
BF16 Sonic registration covers routing, gather and activation kernel entries.
Frontend Triton autotuner choices and provider Quack choices have separate owners.

`_driver.NativeDriver` builds the shared `cuda_utils` extension using a CPU
compiler without initializing CUDA. Install it before Triton's driver first
initializes. Source/headers, Python ABI, platform and libc qualify the binary;
matching installs reuse it and an incompatible initialized driver refuses.
The extension and backing file persist for the process and own no device
allocations. Its loaded CUDA modules belong to the pool and close separately.

## Quack implementation

Under `quack_runtime`, `_cache_scope` owns task-local compile/host caches,
`_choices` owns selected configurations, `_quack_objects` validates and restores
objects, and `_occupancy` prepares the helper while keeping device/stream
occupancy queries live. `_benchmark_scope` suspends inventory during candidate
timing without disabling the tuner's mutation-preservation hooks.

`QuackModulePool` shares content-addressed objects across bindings. Export adds
preload/release entrypoints to a clone of CuTe's IR, using its pinned private
runtime and one-time initialization semantics. Load initializes the CUDA library
without launching tensor computation; close releases it without a second release
from the later wrapper destructor. Ordinary eager exports are unchanged.

The Sonic factory composes these owners with the Triton, native-launcher and
occupancy owners. Its implementation digest covers preparation/runtime sources;
consumers see only the factory lifecycle, manifest and immutable blobs.
