# Prepared external kernels

These components support the explicit `mlops.preparation.sonic:factory` adapter
for ShadowSpill's generic `OperatorPreparation` lifecycle. They are not activated
by importing MLOps. See the [opt-in build example](SHADOWSPILL.md#preparing-sonic-kernels).
They freeze the implementation already selected by dispatch and autotuning;
they do not add a search candidate or change checkpoint retention/chunk sizes.

## Triton

`triton_runtime._scope.TritonCalls` registers explicit, plain JITFunctions.
Capture observes both cold and warm actual device-cache lookups. It does not
replace function objects, so previously imported aliases still participate.
Bindings have independent specialization maps: a missing entry raises even if
the ordinary process cache contains it. Nested scopes and existing threads
retain their own dispatch state. The upstream private cache contract is pinned.

`_objects.TritonObjects` exports JSON records and SHA-256-addressed CUBIN bytes.
Each record contains a typed signature/constants/attributes description and
launch metadata. Export checks that the actual AST, options, compiler, target
and environment reproduce Triton's compilation hash; stale eager binaries
cannot become prepared artifacts. Restoration never evaluates runtime cache-key
strings, deserializes arbitrary Python objects, or calls `preload()`/`compile()`.
The bundled libdevice is resolved locally and checked by content, not by following
an absolute path from the manifest. All records and bytes validate before the
first module is loaded. Like other compiled program artifacts, these are trusted
code; content digests detect corruption, not a malicious artifact author.

`TritonModulePool` owns loaded CUDA modules for one execution context. Load it
before physical memory admission, including the native driver's lazy setup.
Exact records share host launchers across task bindings. Identical CUBIN and
launch metadata also share a CUDA module across different alignment/AST
specializations. CUDA modules are loaded eagerly; invocation still allocates
any scratch specified by the original Triton launcher and must be profiled.
The owner keeps the CUDA context current during an activation, quiesces streams
before `close()`, and never closes an active pool. Partial loads release newly
owned handles. Closing invalidates all its task bindings.

`_driver.NativeDriver` prepares the shared native `cuda_utils` extension using
a CPU compiler without initializing CUDA. Its source/headers, Python ABI,
platform and libc qualify the binary. Install it before first Triton driver
initialization. Repeated matching installations reuse the same module and Python
types; an incompatible already-initialized driver refuses. This extension and
its backing file have **process lifetime**: Python native extensions are not
unloaded per task. It owns no device allocations. The CUDA modules it launches
are separately owned by the pool above.

The initial CUDA contract uses Triton 3.7.1 and pinned loader/cache source.
Compiler/IR overrides, instrumentation, custom linked libraries, tensor
descriptors and arbitrary object/JIT-function constexprs are not admitted.
`_sonic.sonic_triton_calls()` explicitly covers the BF16 Sonic provider's
router, routing metadata, gather and activation kernel entries. Autotuning
choices remain a separate engine responsibility. FP8/EP and unrelated attention
providers are not covered by this registration.

## Quack

Private Quack preparation lives under `quack_runtime`: `_cache_scope` owns
per-task compile/host-plan caches; `_quack_objects` validates target/source/host
ABI and restores selected objects; `_occupancy` substitutes a prepared empty
kernel while retaining live device/stream occupancy queries. Generated and
lazily created epilogues need their outer plan/call caches scoped too.

`QuackModulePool` shares content-addressed objects across task bindings. CuTe's
TVM wrapper normally loads its private CUDA library on first invocation. The
v2 object format adds preload/release entry points to a clone of its exported
IR; these use the same private state and pinned runtime once semantics as the
wrapper and destructor. Loading initializes that state without launching the
tensor computation. Closing explicitly releases it; a later engine destructor
does not release the library again. Ordinary eager exports are unchanged.
The exporter source, host runtime and initialization contract are checked.

`_benchmark_scope` suspends selected-kernel inventory only while the pinned
Quack/Triton autotuners benchmark candidates. Their ordinary mutation-preserving
hooks still run. The winner's final invocation is captured once; there is no
additional replay of a mutating task. Entering a benchmark during prepared
replay refuses.

## Engine ownership

The Sonic factory composes the Quack, Triton, native-launcher and occupancy
owners into one builder or loader. It validates all components before binding
a task and shares compatible modules across its bindings. Active load/close,
wrong CUDA context and use after close refuse. Its implementation digest covers
the adapter and all preparation/runtime source modules; underlying providers
also validate their transitive dependencies.

The engine binds these assets to each unique executable,
including full/tail and accumulating forms, stores manifests/blobs, incorporates
their identity into profiling compatibility, shares loaded objects, and performs
cold admission before execution. ShadowSpill installs its allocator before
loading the modules; loading precedes Plan layout admission and the final
physical-budget seal. The measured external footprint includes device-library
memory outside PyTorch's allocator. This adapter prepares its selected external
kernels. Prebuilt PyTorch/cuBLAS libraries can still initialize lazily on their
first native operation; their footprint belongs to the measured external
reservation, and their first-use latency is distinct from steady execution.
The adapter does not export these framework libraries or override their loading.
A provider-kernel replay is not evidence that
a complete saved Plan is compilation-free or that full-model numerical/DP
qualification passes. Those integration checks remain separate acceptance gates.
