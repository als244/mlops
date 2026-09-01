# Distributed AdamW implementation plan

This is the live implementation guide, progress log, and evidence index for
`mlops.optim.AdamW`. Update it whenever an implementation choice,
test result, limitation, or lowering contract changes.

## Goal

Provide one normal PyTorch `mlops.optim.AdamW` optimizer that selects local,
replicated-gradient, or sharded-state execution from ordinary constructor
arguments while remaining compatible with stage-local dataflow lowering. The
standalone eager path serializes bounded buckets through one backend-portable
algorithm. A lowered stage update is initially one scheduler-visible atomic task whose
measured runtime and workspace include all internal communication.

The engine, PressureFit, and simulator remain communication-agnostic. They see
only local object sizes, mutations, task runtime, and task workspace.

## Decisions

- The only public optimizer class is `mlops.optim.AdamW`.
- `replica_group=None` selects local execution and ignores the optimizer-state
  strategy setting. Supplying a replica ProcessGroup defaults to
  `opt_state_strategy="sharded"`; callers may request `"replicated"`
  explicitly.
- A distributed `AdamW` owns gradient synchronization; it must not be combined
  with another synchronizing wrapper such as ordinary DDP.
- V1 optimizer-state strategies are `replicated` and `sharded`.
- Each synchronization domain uses a separate optimizer and ProcessGroup.
  Rank-local pipeline/expert parameters use an independent local optimizer.
- A lowered stage update is one synchronous `effectful_composite` task.
  Synchronous means one scheduler-visible task completion boundary. NCCL does
  not host-wait; a portability backend may do so when its Work type cannot
  express stream completion. No path performs a global device synchronization.
- Buckets execute serially on the caller stream. NCCL bridges asynchronous
  completion onto that stream; other backends use a portable host wait.
- `bucket_bytes` is a hard collective-payload cap. Compatible buckets reuse
  task-lifetime gradient/reduction scratch, including when one parameter spans
  several buckets.
- Distributed task profiling synchronizes the participating group outside the
  timed interval and charges the maximum member-rank median runtime.
- Trainability is static. Frozen parameters create no optimizer state or
  update work. Every admitted trainable parameter must have a dense gradient
  on every member rank.
- Distributed eager execution preserves Parameter identity while lazily
  rebinding storage into dtype-compatible arenas. This permits hard
  element-level bucket boundaries without duplicating steady-state model
  storage and is delayed until the first step so lowering observes the
  original model state layout.
- Sharded checkpoints are rank-local, noncollective, and same-topology only.
- PyTorch 2.13 or newer is required.

## Dtype policy

Every optimizer parameter group independently specifies:

| Field | Default | Meaning |
|---|---|---|
| model parameter dtype | existing tensor dtype | dtype consumed by model compute |
| `gradient_dtype` | `torch.bfloat16` | local/packed gradient representation |
| `reduction_dtype` | `torch.bfloat16` | collective input/output and local-update gradient dtype |
| `state_dtype` | `torch.bfloat16` | first- and second-moment storage dtype |
| `master_parameter_dtype` | `torch.bfloat16` | optimizer-owned master storage dtype |

`state_dtype` and `master_parameter_dtype` also accept `"parameter"`. If the
resolved master dtype equals the model parameter dtype, the parameter itself
is the master and no extra persistent tensor is created.

The actual eager `.grad` may use another dtype. Packing casts it to
`gradient_dtype`. If `gradient_dtype != reduction_dtype`, the value is cast to
`reduction_dtype` before all-reduce or reduce-scatter. This is necessary for a
genuine FP32 reduction: NCCL/PyTorch collective inputs and outputs have one
dtype, so casting only after a BF16 collective cannot recover lost precision.

PyTorch 2.13 `Parameter.grad_dtype` may be used by callers to request a
different autograd accumulation dtype. Lowering uses the actual AOT gradient
metadata and makes every required cast and accumulator object explicit.

## Status

| Area | Status | Current evidence / next action |
|---|---|---|
| Baseline | complete | 149 mlops tests pass in `dataflow` on 2026-08-04 |
| Living plan | complete | this document |
| Generic local AdamW dtype kernel | complete | mixed BF16/FP16/FP32 raw/custom-op surfaces; `builtin.adamw.triton` |
| Local optimizer dtype policies | complete | four independent constructor and per-group policies |
| Replicated distributed strategy | complete | bucketed asynchronous all-reduce followed by local fused update |
| Sharded-state strategy | complete | reduce-scatter, local state/master update, all-gather |
| Backend and completion | complete | one serialized algorithm; NCCL current-stream bridge, portable Gloo host wait, explicit `synchronize()` |
| State dict and cleanup | complete | rank-local same-topology bucket checkpoint and restore |
| Local distributed tests | complete | world-one NCCL parity and cleanup in ordinary pytest |
| Fleet validation | complete | Chicago RTX 5090 plus Tübingen RTX 3090, 4 × 100-step gates |
| Stage-local lowering | in progress | neutral atomic task/state/collective manifest and PyTorch adapter complete; canonical Program projection pending |
| Coordinated profiling | complete | prototype helper brackets CUDA timing with external group barriers and admits the maximum rank median |
| Documentation | complete | standalone optimizer API/architecture plus prototype task contract and frontend design updated |
| Final validation | complete | standalone Ruff + 162 tests and renewed NCCL/Gloo two-host 4 × 100-step gates pass with exact reference parity |
| Scaling benchmark | complete | 40/40 current NCCL cells pass from `2^12` through `2^31` for both strategies; retained pre-refactor and current data include per-rank timing and allocator evidence |

## Implementation checklist

### A. Generalized local AdamW

- [x] Factor common AdamW option and dtype-policy validation.
- [x] Support BF16, FP16, and FP32 parameters, gradients, moments, and
  admitted master-parameter combinations.
- [x] Make the registered kernel reproduce PyTorch's ordered dtype-visible
  round points; retain the original internal-FP32 kernel as a named raw
  development ablation.
- [x] Add a private gradient scale used for mean reduction.
- [x] Provide allocation-returning, `out=`, and last-use mutation surfaces
  with truthful schemas for the generalized call domain.
- [x] Rename the exact implementation to `builtin.adamw.triton` and migrate
  consumers without aliases.
- [x] Validate master aliasing eliminates both the object and its bytes.
- [x] Add mixed-dtype numerical tests. Multi-step distributed parity remains
  an acceptance item.

### B. Optional distributed eager execution

- [x] Extend the one AdamW constructor with optional replica ProcessGroup,
  optimizer-state strategy,
  reduction and hard bucket settings.
- [x] Add deterministic per-group bucket construction and manifest hashing.
- [x] Validate group membership and cross-rank parameter manifests once at
  setup.
- [x] Implement replicated all-reduce/update execution.
- [x] Implement sharded reduce-scatter/update/all-gather execution.
- [x] Serialize bounded bucket phases on the caller stream.
- [x] Select backend completion internally: nonblocking current-stream bridge
  for NCCL and portable host wait for Gloo/other ProcessGroup backends.
- [x] Split oversized parameters at homogeneous element boundaries and reuse
  one task-lifetime workspace per dtype class.
- [x] Retain tensor and Work lifetimes until real completion.
- [x] Add noncollective rank-local state save/load and explicit synchronize.
- [x] Confirm no steady-state tensor allocation growth after warmup.

### C. Lowering and profiling

- [x] Normalize one or several ordinary AdamW optimizer objects by their
  resolved local/replicated/sharded execution mode.
- [x] Exclude frozen parameters and require an exact cover of trainable state.
- [ ] Derive update regions from optimizer identity and final gradient
  producer rather than experiment-specific grouping.
- [x] Emit one atomic distributed optimizer task per caller-supplied stage
  task group.
- [x] Keep internal collective semantics in an executor-side composite
  manifest while omitting them from the simulator graph.
- [x] Declare parameters, gradients, masters, moments, and steps as engine
  objects; charge internal packing/communication buffers as workspace.
- [x] Add coordinated group profiling with barriers outside CUDA-event timing.
- [x] Include deployment topology, backend, completion mode, group size, and
  dtype layout in profile compatibility identity.

### D. Acceptance

- [x] Both strategies match an independent full-state reference over 100
  updates for sum and mean reductions.
- [x] Mixed BF16/FP32 parameter groups and independent gradient/reduction/state
  /master policies pass.
- [x] Frozen-base/LoRA parameters remain unchanged and allocate no state.
- [x] Reconstructed sharded moments match the full reference by cosine,
  relative L2, and sign agreement; cosine/sign agreement target is at least
  0.99.
- [x] Chicago and Tübingen use byte-identical source manifests.
- [x] Lowered task manifests remain stage-local and contain no scheduler-visible
  explicit communication tasks.
- [x] Full mlops and prototype tests plus Ruff pass.

## Evidence log

| Date | Milestone | Command / artifact | Result |
|---|---|---|---|
| 2026-08-04 | Baseline | `python -m pytest -q mlops/tests` in `dataflow` | 149 passed in 5.25 s |
| 2026-08-04 | Generalized local canary | `python -m pytest -q tests/test_optim.py` | 18 passed in 3.20 s; mixed model/gradient/state/master dtypes and all storage surfaces |
| 2026-08-04 | Generalized package gate | `python -m pytest -q` | 155 passed in 5.48 s |
| 2026-08-04 | World-one distributed gate | `python -m pytest -q tests/test_distributed_optim.py tests/test_optim.py` | 20 passed in 3.64 s; both strategies bitwise-match local semantics |
| 2026-08-04 | Two-host distributed gate | [`../evidence/distributed_adamw_fleet.json`](../evidence/distributed_adamw_fleet.json) | 100 steps × replicated/sharded × sum/mean; parameter/master/moment relative L2 exactly 0 on RTX 5090 + RTX 3090 |
| 2026-08-04 | Standalone regression gate | `ruff check src tests && python -m pytest -q` | Ruff clean; 157 passed in 5.97 s |
| 2026-08-04 | Replica API rename gate | two-host `adamw_worker.py --steps 100` | V2 `replicated`/`sharded` × sum/mean all retain parameter relative L2 exactly 0 after the clean `replica_group`/`opt_state_strategy` rename |
| 2026-08-04 | Final regression gate | standalone and `reference_models/prototypes/pytorch_lowering` Ruff + pytest | standalone 157/157 pass; complete prototype suite passes after neutral-contract/API rename |
| 2026-08-04 | Two-host scaling reference | [`../../benchmarks/adamw_scaling/results/reference_bucketed_64m/RESULTS.md`](../../benchmarks/adamw_scaling/results/reference_bucketed_64m/RESULTS.md) | 24/24 NCCL cells pass; exact powers `2^20`–`2^31`, sharded and replicated, complete-step CUDA-event medians from 1.022–1686.550 ms |
| 2026-08-04 | Overlap and layout ablations | [`../../benchmarks/adamw_scaling/README.md`](../../benchmarks/adamw_scaling/README.md) | overlap saves 0.2–1.2% at large sizes; one `2^31`-element parameter is 1.51% faster for sharded state and neutral for replicated state relative to 64 model-visible 64 MiB parameters |
| 2026-08-04 | Backend completion isolation | [`../../benchmarks/adamw_scaling/results/backend_comparison`](../../benchmarks/adamw_scaling/results/backend_comparison) | Gloo CUDA reduce-scatter is correct with `Work.wait()` but its PyTorch 2.13 Work lacks the future required by `block_current_stream()`; NCCL supports the stream bridge |
| 2026-08-04 | Universal serialized correctness | `nccl_correctness_v3.json`, `gloo_correctness_v3.json` in the backend-comparison results | both backends, replicated/sharded, sum/mean, mixed dtype policies, 100 steps: parameter/master/moment relative L2 exactly 0 |
| 2026-08-04 | Simplified package gate | `ruff check src tests benchmarks/adamw_scaling && pytest -q` | Ruff clean; 158 passed in 6.42 s |
| 2026-08-04 | Universal NCCL scaling gate | [`../../benchmarks/adamw_scaling/results/universal_serial_64m/COMPARISON.md`](../../benchmarks/adamw_scaling/results/universal_serial_64m/COMPARISON.md) | 40/40 cells pass over `2^12`–`2^31` for both strategies; large latency matches the old serialized path within about 0.5%, while `2^31` declared workspace falls from 8 GiB to 96 MiB sharded and 4 GiB to 64 MiB replicated |
| 2026-08-04 | Hard-bound Gloo canary | [`../../benchmarks/adamw_scaling/results/universal_gloo_single_2p31/RESULTS.md`](../../benchmarks/adamw_scaling/results/universal_gloo_single_2p31/RESULTS.md) | one 2B-element parameter automatically becomes 64 buckets; 96 MiB declared scratch, 14.16 GiB measured peak, no OOM, 8.066 s / 4.26 Gb/s median |
| 2026-08-04 | PyTorch arithmetic fidelity | `python -m pytest -q tests/test_optim.py` | 22 passed; default BF16 parameter and same-dtype moment updates are bitwise gated; original internal-FP32 arithmetic remains a separate raw kernel |
| 2026-08-04 | Arithmetic performance ablation | local CUDA-event probe, BF16, `2^20`/`2^24`/`2^28` elements | PyTorch-compatible versus internal-FP32: 0.0208/0.1540/2.4440 ms versus 0.0201/0.1531/2.4432 ms; no material large-tensor regression |

## Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-08-04 | Use separate local and reduced-gradient dtype policies | permits BF16 local accumulation with a true FP32 collective and update input |
| 2026-08-04 | Keep communication inside an atomic stage update task | PressureFit and the simulator can reuse the existing local-object/runtime/workspace contract |
| 2026-08-04 | Profile distributed tasks with group synchronization outside the timed interval | collective timings otherwise include rank drift or unrelated prior work |
| 2026-08-04 | Use flat parameter views for efficient eager sharded state, initialized lazily | avoids a duplicate full model copy without contaminating frontend stage identity before lowering |
| 2026-08-04 | Treat frozen parameters as absent physical optimizer members | supports LoRA while retaining ordinary model parameter registration |
| 2026-08-04 | Make ordered PyTorch round points the registered default and retain the original kernel as `internal_fp32` | same-dtype moments and the default BF16 training path become bitwise comparable without a measurable large-tensor slowdown; the alternate remains available for accuracy/performance diagnosis |
| 2026-08-04 | Do not emulate PyTorch host bias correction with FP64 inside Triton | default non-capturable PyTorch uses Python/host arithmetic, while mlops must keep the step CUDA-resident and asynchronous; GPU FP64 neither guarantees host-libm identity nor justifies its cost |
| 2026-08-04 | Use PyTorch 2.13 `reduce_scatter_single` and `all_gather_single` | avoids deprecated aliases while keeping ProcessGroupNCCL as an implementation detail below the optimizer façade |
| 2026-08-04 | Validate distributed state by reconstructing its full logical tensors only in tests | the runtime and checkpoint stay rank-local; validation does not weaken sharded ownership |
| 2026-08-04 | Keep optimizer state policies per parameter group | the fleet gate mixes BF16 and FP32 model parameters, BF16/FP32 moments, aliased/distinct masters, and BF16-local/FP32-reduced gradients in one optimizer |
| 2026-08-04 | Name the public group `replica_group` and ownership policy `opt_state_strategy` | an optimizer synchronizes replicas of the same parameter identities; explicit naming avoids confusing this group or policy with tensor/expert/pipeline parallelism |
| 2026-08-04 | Use `sharded` and `replicated` as the only state-policy values | the `opt_state_strategy` field already supplies the state context, so `sharded_state` was redundant |
| 2026-08-04 | Bump the distributed checkpoint/execution manifest to V2 | the clean field/value rename must reject V1 checkpoints explicitly rather than failing later on a missing key |
| 2026-08-04 | Preserve the original whole-parameter behavior only as historical benchmark evidence | it is the immutable performance baseline, not the current memory contract |
| 2026-08-04 | Make 64 MiB a hard per-collective payload cap | Gloo sharded state required an additional full input-sized temporary for one 2B-element collective and exhausted the 24 GiB rank; element splitting makes peak scratch independent of parameter size |
| 2026-08-04 | Remove public overlap and stream options | measured overlap benefit was only 0.2–1.2%, Gloo cannot use the same reduce-scatter stream bridge, and serialized buckets enable workspace reuse and one backend-portable implementation |
| 2026-08-04 | Default unknown ProcessGroup backends to host completion and specialize NCCL | correctness is universal while the known efficient NCCL path remains nonblocking; backend mechanics stay below the optimizer API |

## Deferred work

- dynamic unused trainable parameters;
- topology-changing checkpoint restore and state consolidation;
- multiple ProcessGroups inside one optimizer;
- engine-visible communication tasks and cross-stage overlap;
- gradient-ready backward hooks;
- quantized state, compression, AMSGrad, and elastic groups;
- complete JAX and ROCm qualification.
