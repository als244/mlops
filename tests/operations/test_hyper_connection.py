from types import SimpleNamespace

import pytest
import torch

from mlops import (
    mhc_combine,
    mhc_pre,
)

from .reference import official

# The streams' dtype, the projection weight's, then the bias and scale's. mHC
# trains with the streams in the activation dtype and every coefficient in FP32.
LAYOUTS = {
    # Full precision holds the implementation to FP32 tolerances throughout.
    "fp32": (torch.float32, torch.float32, torch.float32),
    # DeepSeek-V4 and Megatron-Core keep every mHC parameter in FP32.
    "fp32_parameters": (torch.bfloat16, torch.float32, torch.float32),
    # GLM-5.3-Flash's checkpoint stores only the projection in BF16.
    "glm_checkpoint": (torch.bfloat16, torch.bfloat16, torch.float32),
    # transformers loads every parameter in the model's BF16.
    "bf16": (torch.bfloat16, torch.bfloat16, torch.bfloat16),
}


def assert_close(actual, expected):
    """Hold each tensor to its own precision.

    A BF16 result is rounded from FP32 math, so agreeing FP32 values can still
    round one BF16 ulp apart. The streams' gradient is rounded once per path
    into the streams in eager mode, but once in total where compilation fuses
    the paths, so the two differ by ulps of the gradient's scale rather than of
    each element; FP32 results keep the tight bound.
    """
    if actual.dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.003)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_hyper_connection(layout, compiled):
    dtype, projection_dtype, vector_dtype = LAYOUTS[layout]
    _, HC = official("Glm5NextTextUnweightedRMSNorm", "Glm5NextTextHyperConnection")
    cfg = SimpleNamespace(
        hc_mult=4, hidden_size=32, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-5
    )
    model = HC(cfg).cuda()
    torch.manual_seed(29)
    with torch.no_grad():
        model.fn.normal_(std=0.05)
        model.base.normal_(std=0.1)
        model.scale.fill_(0.1)
    for name, parameter_dtype in (
        ("fn", projection_dtype),
        ("base", vector_dtype),
        ("scale", vector_dtype),
    ):
        value = getattr(model, name).detach().to(parameter_dtype)
        setattr(model, name, torch.nn.Parameter(value))
    x = torch.randn(2, 7, 4, 32, dtype=dtype, device="cuda", requires_grad=True)
    ours_x = x.detach().clone().requires_grad_()
    params = [
        p.detach().clone().requires_grad_() for p in (model.fn, model.base, model.scale)
    ]
    ref = model(x)

    def composition(x, projection, bias, scale):
        return mhc_pre(x, projection, bias, scale)

    fn = torch.compile(composition, fullgraph=True) if compiled else composition
    got = fn(ours_x, *params)
    # Whatever the streams and parameters are, the coefficients are FP32.
    assert got[0].dtype == got[1].dtype == torch.float32
    upstream = [torch.randn_like(value) for value in ref]
    for a, b in zip(got, ref):
        assert_close(a, b)
    ref_grads = torch.autograd.grad(
        ref, (x, model.fn, model.base, model.scale), upstream
    )
    got_grads = torch.autograd.grad(got, (ours_x, *params), upstream)
    for a, b in zip(got_grads, ref_grads):
        assert_close(a, b)
    branch = torch.randn(2, 7, 32, dtype=dtype, device="cuda")
    expected = (
        ref[0].to(dtype).unsqueeze(-1) * branch.unsqueeze(-2)
        + ref[1].to(dtype).transpose(-1, -2) @ x
    )
    torch.testing.assert_close(
        mhc_combine(branch, x, *ref[:2], coefficient_precision="activation"), expected
    )


pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [127, 4096])
@pytest.mark.parametrize("precision", ["fp32", "activation"])
def test_four_stream_combine_gradients(compiled, dtype, width, precision):
    torch.manual_seed(487)
    branch = torch.randn(2, 7, width, device="cuda", dtype=dtype)
    streams = torch.randn(2, 7, 4, width, device="cuda", dtype=dtype)
    post = torch.rand(2, 7, 4, device="cuda", dtype=torch.float32)
    mixing = torch.rand(2, 7, 4, 4, device="cuda", dtype=torch.float32)
    inputs = [v.requires_grad_() for v in (branch, streams, post, mixing)]
    ref_inputs = [v.detach().clone().requires_grad_() for v in inputs]

    def reference(b, x, p, m):
        if precision == "fp32":
            return (
                p.float().unsqueeze(-1) * b.float().unsqueeze(-2)
                + m.float().transpose(-1, -2) @ x.float()
            ).to(dtype)
        return (
            p.to(dtype).unsqueeze(-1) * b.unsqueeze(-2)
            + m.to(dtype).transpose(-1, -2) @ x
        )

    # Compare the same compilation mode: compilation already changes rounding
    # of the post-weight reduction even with the original BMM implementation.
    reference_fn = torch.compile(reference, fullgraph=True) if compiled else reference
    expected = reference_fn(*ref_inputs)

    def operation(b, x, p, m):
        return mhc_combine(b, x, p, m, coefficient_precision=precision)

    fn = torch.compile(operation, fullgraph=True) if compiled else operation
    actual = fn(*inputs)
    upstream = torch.randn_like(expected)
    reference_grads = torch.autograd.grad(expected, ref_inputs, upstream)
    actual_grads = torch.autograd.grad(actual, inputs, upstream)
    for a, b in zip((actual, *actual_grads), (expected, *reference_grads)):
        # Compiled surrounding elementwise paths round at fusion boundaries;
        # the matrix product itself retains reduced operands/FP32 accumulation.
        torch.testing.assert_close(a, b, rtol=0.02, atol=0.03)


@pytest.mark.parametrize("count", [3, 4])
def test_mhc_combine_dispatch(count):
    from mlops.dispatch import resolve_implementation, use_implementation

    b = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, count, 32, device="cuda", dtype=b.dtype)
    p = torch.rand(2, count, device="cuda")
    m = torch.rand(2, count, count, device="cuda")
    impl = resolve_implementation("mhc_combine", b, x, p, m, surface="semantic")
    expected = (
        "builtin.mhc_combine.four_stream" if count == 4 else "native_torch.mhc_combine"
    )
    assert impl.implementation_id == expected
    actual = mhc_combine(b, x, p, m)
    with use_implementation("mhc_combine", "native_torch.mhc_combine"):
        ref = mhc_combine(b, x, p, m)
    torch.testing.assert_close(actual, ref)


def test_mhc_pre_autocast_preserves_fp32_coefficients():
    x = torch.randn(2, 4, 32, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(24, 128, device="cuda", dtype=torch.bfloat16) * 0.05
    b = torch.randn(24, device="cuda") * 0.1
    s = torch.ones(3, device="cuda") * 0.1
    expected = mhc_pre(x, w, b, s)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = mhc_pre(x, w, b, s)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    assert actual[0].dtype == actual[1].dtype == torch.float32


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("iterations", [1, 20])
def test_tilekernels_mhc_pre(compiled, iterations):
    pytest.importorskip("tile_kernels")
    from mlops.dispatch import use_implementation

    x = torch.randn(2, 8, 4, 32, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(24, 128, device="cuda", dtype=torch.bfloat16) * 0.05
    b = torch.randn(24, device="cuda") * 0.1
    s = torch.ones(3, device="cuda") * 0.1
    inputs = [v.requires_grad_() for v in (x, w, b, s)]

    def fn(*args):
        return mhc_pre(*args, iterations=iterations)

    with use_implementation("mhc_pre", "native_torch.mhc_pre"):
        ref = fn(*inputs)
    upstream = [torch.randn_like(v) for v in ref]
    ref_grad = torch.autograd.grad(ref, inputs, upstream)
    with use_implementation("mhc_pre", "tilekernels.mhc_pre.sinkhorn"):
        call = torch.compile(fn, fullgraph=True) if compiled else fn
        got = call(*inputs)
        got_grad = torch.autograd.grad(got, inputs, upstream)
    for a, b in zip((*got, *got_grad), (*ref, *ref_grad)):
        assert_close(a, b)


@pytest.mark.parametrize("compiled", [False, True])
def test_tilekernels_mhc_combine(compiled):
    pytest.importorskip("tile_kernels")
    from mlops.dispatch import use_implementation

    torch.manual_seed(499)
    inputs = [
        torch.randn(*shape, device="cuda", dtype=dtype).requires_grad_()
        for shape, dtype in [
            ((17, 4096), torch.bfloat16),
            ((17, 4, 4096), torch.bfloat16),
            ((17, 4), torch.float32),
            ((17, 4, 4), torch.float32),
        ]
    ]
    with use_implementation("mhc_combine", "native_torch.mhc_combine"):
        ref = mhc_combine(*inputs)
    upstream = torch.randn_like(ref)
    ref_grad = torch.autograd.grad(ref, inputs, upstream)
    with use_implementation("mhc_combine", "tilekernels.mhc_combine"):
        fn = torch.compile(mhc_combine, fullgraph=True) if compiled else mhc_combine
        got = fn(*inputs)
        got_grad = torch.autograd.grad(got, inputs, upstream)
    for a, b in zip((got, *got_grad), (ref, *ref_grad)):
        if a.dtype == torch.float32:
            torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4)
        else:
            torch.testing.assert_close(a, b, rtol=0.01, atol=0.02)


@pytest.mark.parametrize(
    "capability,eligible",
    [((8, 0), False), ((9, 0), True), ((10, 0), True), ((12, 0), False)],
)
def test_tilekernels_architecture_gate(monkeypatch, capability, eligible):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from mlops.providers.tilekernels import hyper_connection as provider

    monkeypatch.setattr(provider, "_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    with FakeTensorMode():
        x = torch.empty(16, 4, 4096, device="cuda", dtype=torch.bfloat16)
        b = torch.empty(16, 4096, device="cuda", dtype=x.dtype)
        p = torch.empty(16, 4, device="cuda")
        m = torch.empty(16, 4, 4, device="cuda")
        w = torch.empty(24, 16384, device="cuda")
        bias = torch.empty(24, device="cuda")
        scale = torch.empty(3, device="cuda")
        assert bool(provider._supports_post(b, x, p, m, surface="semantic")) == eligible
        assert (
            bool(provider._supports(x, w, bias, scale, surface="semantic")) == eligible
        )
        assert not provider._supports_post(
            b, x, p, m, surface="semantic", coefficient_precision="activation"
        )
