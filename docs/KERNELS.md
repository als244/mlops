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
