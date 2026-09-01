"""Fused optimizer API, custom-op, state, and allocation contracts."""

from __future__ import annotations

import pytest
import torch
import inspect

from mlops.dispatch import implementation_registry
from mlops.optim import (
    AdamW,
    adamw,
    adamw_,
    functional_adamw,
    functional_master_adamw,
    master_adamw,
    master_adamw_,
)
from mlops.providers.builtin.adamw import (
    _functional_op,
    _in_place_op,
    _master_functional_op,
    _master_in_place_op,
    _master_out_op,
    _out_op,
)
from mlops.kernels.adamw import adamw_out_internal_fp32_raw


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def _state(elements: int = 4096):
    parameter = torch.randn(elements, device="cuda", dtype=torch.bfloat16)
    gradient = torch.randn_like(parameter)
    exp_avg = torch.zeros_like(parameter)
    exp_avg_sq = torch.zeros_like(parameter)
    step = torch.zeros((), device="cuda", dtype=torch.float32)
    return parameter, gradient, exp_avg, exp_avg_sq, step


def _scalars():
    return 1.0, 3e-4, 0.9, 0.95, 1e-8, 0.1, False


def _reference_first_update(
    parameter,
    gradient,
    exp_avg,
    exp_avg_sq,
    *,
    lr,
    betas,
    eps,
    weight_decay,
    maximize,
):
    beta1, beta2 = betas
    update_gradient = gradient.neg() if maximize else gradient
    next_parameter = parameter.clone()
    next_exp_avg = exp_avg.clone()
    next_exp_avg_sq = exp_avg_sq.clone()
    next_parameter.mul_(1.0 - lr * weight_decay)
    next_exp_avg.lerp_(update_gradient, 1.0 - beta1)
    next_exp_avg_sq.mul_(beta2).addcmul_(
        update_gradient,
        update_gradient,
        value=1.0 - beta2,
    )
    step_size = lr / (1.0 - beta1)
    denominator = (
        next_exp_avg_sq.sqrt().div_((1.0 - beta2) ** 0.5).add_(eps)
    )
    next_parameter.addcdiv_(next_exp_avg, denominator, value=-step_size)
    return next_parameter, next_exp_avg, next_exp_avg_sq


def _assert_storage_precision_close(actual, expected):
    """Require agreement appropriate to the tensor's stored precision."""
    actual_f32 = actual.float().flatten()
    expected_f32 = expected.float().flatten()
    relative_l2 = float(
        (actual_f32 - expected_f32).norm()
        / expected_f32.norm().clamp_min(1e-30)
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_f32,
            expected_f32,
            dim=0,
        )
    )
    sign_agreement = float(
        (torch.signbit(actual_f32) == torch.signbit(expected_f32))
        .float()
        .mean()
    )
    assert cosine >= 0.999
    assert relative_l2 <= 0.001
    assert sign_agreement >= 0.999

    if expected.dtype == torch.bfloat16:
        # Triton may fuse the multiply-add before BF16 storage rounding.  The
        # independent Torch expression rounds the same result within one BF16
        # ULP, while the directional metrics above guard against loose bounds.
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.016)
    elif expected.dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=0.003, atol=0.002)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=6e-6)


def test_default_bf16_adamw_matches_torch_optimizer_bitwise():
    dtype = torch.bfloat16
    torch.manual_seed(14900)
    parameter = torch.randn(65536, device="cuda", dtype=dtype)
    torch_parameter = torch.nn.Parameter(parameter.clone())
    mlops_parameter = torch.nn.Parameter(parameter.clone())
    options = {
        "lr": 3e-4,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }
    torch_optimizer = torch.optim.AdamW([torch_parameter], **options)
    mlops_optimizer = AdamW(
        [mlops_parameter],
        **options,
        gradient_dtype=dtype,
        state_dtype=dtype,
        master_parameter_dtype=dtype,
    )

    for _ in range(10):
        gradient = torch.randn_like(parameter)
        torch_parameter.grad = gradient.clone()
        mlops_parameter.grad = gradient.clone()
        torch_optimizer.step()
        mlops_optimizer.step()

    torch_state = torch_optimizer.state[torch_parameter]
    mlops_state = mlops_optimizer.state[mlops_parameter]
    assert torch.equal(mlops_parameter, torch_parameter)
    assert torch.equal(mlops_state["exp_avg"], torch_state["exp_avg"])
    assert torch.equal(mlops_state["exp_avg_sq"], torch_state["exp_avg_sq"])
    assert int(mlops_state["step"]) == int(torch_state["step"])


@pytest.mark.parametrize(
    ("dtype", "minimum_parameter_equality"),
    [(torch.float16, 0.9999), (torch.float32, 0.99)],
)
def test_other_same_dtype_adamw_matches_torch_moments_bitwise(
    dtype,
    minimum_parameter_equality,
):
    torch.manual_seed(14900)
    parameter = torch.randn(65536, device="cuda", dtype=dtype)
    torch_parameter = torch.nn.Parameter(parameter.clone())
    mlops_parameter = torch.nn.Parameter(parameter.clone())
    options = {
        "lr": 3e-4,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }
    torch_optimizer = torch.optim.AdamW([torch_parameter], **options)
    mlops_optimizer = AdamW(
        [mlops_parameter],
        **options,
        gradient_dtype=dtype,
        state_dtype=dtype,
        master_parameter_dtype=dtype,
    )
    for _ in range(10):
        gradient = torch.randn_like(parameter)
        torch_parameter.grad = gradient.clone()
        mlops_parameter.grad = gradient.clone()
        torch_optimizer.step()
        mlops_optimizer.step()

    torch_state = torch_optimizer.state[torch_parameter]
    mlops_state = mlops_optimizer.state[mlops_parameter]
    assert torch.equal(mlops_state["exp_avg"], torch_state["exp_avg"])
    assert torch.equal(mlops_state["exp_avg_sq"], torch_state["exp_avg_sq"])
    equality = float((mlops_parameter == torch_parameter).float().mean())
    assert equality >= minimum_parameter_equality


def test_original_internal_fp32_adamw_arithmetic_remains_available():
    parameter, gradient, exp_avg, exp_avg_sq, step = _state(4096)
    exp_avg.copy_(torch.randn_like(exp_avg))
    exp_avg_sq.copy_(torch.rand_like(exp_avg_sq))
    outputs = tuple(
        torch.empty_like(value) for value in (parameter, exp_avg, exp_avg_sq, step)
    )
    adamw_out_internal_fp32_raw(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        maximize=False,
        out=outputs,
    )
    assert all(torch.isfinite(value).all() for value in outputs[:-1])
    assert int(outputs[-1]) == 1


def test_adamw_custom_ops_pass_opcheck():
    parameter, gradient, exp_avg, exp_avg_sq, step = _state(128)
    scalars = _scalars()
    outputs = tuple(
        value.clone() for value in (parameter, exp_avg, exp_avg_sq, step)
    )
    cases = (
        (
            _functional_op,
            (parameter, gradient, exp_avg, exp_avg_sq, step, *scalars),
        ),
        (
            _out_op,
            (
                parameter,
                gradient,
                exp_avg,
                exp_avg_sq,
                step,
                *scalars,
                *outputs,
            ),
        ),
        (
            _in_place_op,
            (
                parameter.clone(),
                gradient,
                exp_avg.clone(),
                exp_avg_sq.clone(),
                step.clone(),
                *scalars,
            ),
        ),
        (
            _master_functional_op,
            (
                parameter,
                parameter.float(),
                gradient.float(),
                exp_avg.float(),
                exp_avg_sq.float(),
                step.to(torch.int64),
                *scalars,
            ),
        ),
        (
            _master_in_place_op,
            (
                parameter.clone(),
                parameter.float(),
                gradient.float(),
                exp_avg.float(),
                exp_avg_sq.float(),
                step.to(torch.int64),
                *scalars,
            ),
        ),
        (
            _master_out_op,
            (
                parameter,
                parameter.float(),
                gradient.float(),
                exp_avg.float(),
                exp_avg_sq.float(),
                step.to(torch.int64),
                *scalars,
                parameter.clone(),
                parameter.float(),
                exp_avg.float(),
                exp_avg_sq.float(),
                step.to(torch.int64),
            ),
        ),
    )
    for operation, arguments in cases:
        assert set(torch.library.opcheck(operation, arguments).values()) == {
            "SUCCESS"
        }


@pytest.mark.parametrize("maximize", [False, True])
def test_functional_adamw_matches_independent_rounded_bf16_reference(maximize):
    torch.manual_seed(14900 + maximize)
    parameter, gradient, exp_avg, exp_avg_sq, step = _state(4096)
    options = {
        "lr": 2e-4,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.03,
        "maximize": maximize,
    }
    expected = _reference_first_update(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        **options,
    )

    actual = functional_adamw(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        **options,
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2], rtol=0.01, atol=0.002)
    assert float(actual[3]) == 1.0


def test_functional_and_in_place_forms_match_without_allocation():
    torch.manual_seed(14901)
    parameter, gradient, exp_avg, exp_avg_sq, step = _state(1 << 20)
    options = {
        "lr": 2e-4,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.03,
    }
    expected = functional_adamw(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        **options,
    )
    actual = tuple(
        value.clone() for value in (parameter, exp_avg, exp_avg_sq, step)
    )
    adamw_(
        actual[0],
        gradient,
        actual[1],
        actual[2],
        actual[3],
        **options,
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(actual, expected, strict=True)
    )

    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    adamw_(
        actual[0],
        gradient,
        actual[1],
        actual[2],
        actual[3],
        **options,
    )
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - baseline == 0


def test_out_form_preserves_sources_and_writes_exact_destinations():
    torch.manual_seed(14903)
    sources = _state(4096)
    snapshots = tuple(value.clone() for value in sources)
    options = {
        "lr": 2e-4,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.03,
        "maximize": True,
    }
    expected = functional_adamw(*sources, **options)
    destinations = tuple(
        torch.empty_like(value)
        for value in (sources[0], sources[2], sources[3], sources[4])
    )
    returned = adamw(*sources, **options, out=destinations)

    assert returned is destinations
    assert all(
        torch.equal(value, snapshot)
        for value, snapshot in zip(sources, snapshots, strict=True)
    )
    assert all(
        torch.equal(actual, reference)
        for actual, reference in zip(destinations, expected, strict=True)
    )


def test_out_form_rejects_source_or_destination_aliases():
    sources = _state(128)
    destinations = tuple(
        torch.empty_like(value)
        for value in (sources[0], sources[2], sources[3], sources[4])
    )
    options = {
        "lr": 2e-4,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.03,
    }
    with pytest.raises(ValueError, match="storage-disjoint"):
        adamw(*sources, **options, out=(sources[0], *destinations[1:]))
    with pytest.raises(ValueError, match="storage-disjoint"):
        adamw(
            *sources,
            **options,
            out=(destinations[0], destinations[0], *destinations[2:]),
        )


def test_adamw_uses_standard_optimizer_state_protocol():
    torch.manual_seed(14902)
    parameter = torch.nn.Parameter(
        torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    )
    parameter.grad = torch.randn_like(parameter)
    optimizer = AdamW(
        [parameter],
        lr=2e-4,
        betas=(0.9, 0.95),
        weight_decay=0.03,
    )
    optimizer.step()
    state = optimizer.state[parameter]
    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
    assert state["exp_avg"].dtype == parameter.dtype
    assert state["exp_avg_sq"].dtype == parameter.dtype
    assert state["step"].dtype == torch.int64
    assert int(state["step"]) == 1

    saved = optimizer.state_dict()
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = AdamW(
        [restored_parameter],
        lr=2e-4,
        betas=(0.9, 0.95),
        weight_decay=0.03,
    )
    restored.load_state_dict(saved)
    restored_state = restored.state[restored_parameter]
    assert all(
        torch.equal(restored_state[name], state[name])
        for name in ("step", "exp_avg", "exp_avg_sq")
    )


def test_adamw_constructor_matches_torch_option_names_defaults_and_kinds():
    expected = inspect.signature(torch.optim.AdamW).parameters
    actual = inspect.signature(AdamW).parameters
    assert tuple(actual)[: len(expected)] == tuple(expected)
    for name in expected:
        assert actual[name].kind == expected[name].kind
        assert actual[name].default == expected[name].default
    assert actual["gradient_dtype"].default == torch.bfloat16
    assert actual["reduction_dtype"].default == torch.bfloat16
    assert actual["state_dtype"].default == torch.bfloat16
    assert actual["master_parameter_dtype"].default == torch.bfloat16
    assert actual["replica_group"].default is None
    assert actual["opt_state_strategy"].default == "sharded"


def test_adamw_accepts_implementation_hint_options_without_changing_identity():
    parameter = torch.nn.Parameter(
        torch.randn(32, device="cuda", dtype=torch.bfloat16)
    )
    optimizer = AdamW(
        [parameter],
        foreach=True,
        capturable=True,
        fused=False,
    )
    group = optimizer.param_groups[0]
    assert group["foreach"] is True
    assert group["capturable"] is True
    assert group["fused"] is False
    assert optimizer.implementation_id == "builtin.adamw.triton"


def test_adamw_preserves_parameter_groups_and_skips_missing_gradients():
    torch.manual_seed(14904)
    first = torch.nn.Parameter(
        torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    )
    second = torch.nn.Parameter(torch.randn_like(first))
    untouched = torch.nn.Parameter(torch.randn_like(first))
    first_before = first.detach().clone()
    second_before = second.detach().clone()
    first.grad = torch.randn_like(first)
    second.grad = first.grad.detach().clone()
    optimizer = AdamW(
        [
            {"params": [first], "lr": 0.0},
            {"params": [second, untouched], "lr": 1e-2},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    optimizer.step()

    assert torch.equal(first, first_before)
    assert not torch.equal(second, second_before)
    assert set(optimizer.state) == {first, second}
    assert optimizer.param_groups[0]["lr"] == 0.0
    assert optimizer.param_groups[1]["lr"] == 1e-2


def test_adamw_registry_identity_and_support_gate_are_explicit():
    implementation = implementation_registry()["adamw"]["builtin.adamw.triton"]
    assert implementation.apply is functional_adamw
    assert implementation.forward is None
    assert implementation.backward is None
    assert implementation.deterministic

    parameter, gradient, exp_avg, exp_avg_sq, step = _state(128)
    assert implementation.supports(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        surface="semantic",
    )
    rejected = implementation.supports(
        parameter.to(torch.int32),
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        surface="semantic",
    )
    assert not rejected
    assert "BF16, FP16, or FP32" in rejected.reason


@pytest.mark.parametrize(
    ("parameter_dtype", "gradient_dtype", "state_dtype"),
    [
        (torch.bfloat16, torch.float32, torch.float32),
        (torch.float32, torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float32, torch.float32),
    ],
)
def test_mixed_dtype_adamw_matches_explicit_storage_rounding(
    parameter_dtype,
    gradient_dtype,
    state_dtype,
):
    torch.manual_seed(14905)
    parameter = torch.randn(8192, device="cuda", dtype=parameter_dtype)
    gradient = torch.randn(8192, device="cuda", dtype=gradient_dtype)
    exp_avg = torch.randn(8192, device="cuda", dtype=state_dtype)
    exp_avg_sq = torch.rand(8192, device="cuda", dtype=state_dtype)
    step = torch.tensor(7, device="cuda", dtype=torch.int64)
    options = {
        "lr": 2e-4,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.03,
        "gradient_scale": 0.5,
    }
    beta1, beta2 = options["betas"]
    gradient_f32 = gradient.float() * options["gradient_scale"]
    expected_mean = torch.lerp(
        exp_avg.float(),
        gradient_f32,
        1.0 - beta1,
    ).to(state_dtype)
    scaled_variance = (exp_avg_sq.float() * beta2).to(state_dtype)
    expected_variance = torch.addcmul(
        scaled_variance.float(),
        gradient_f32,
        gradient_f32,
        value=1.0 - beta2,
    ).to(state_dtype)
    next_step = float(step) + 1.0
    expected_parameter = (
        parameter.float()
        * (1.0 - options["lr"] * options["weight_decay"])
    ).to(parameter_dtype)
    denominator = expected_variance.float().sqrt().to(state_dtype)
    denominator = (
        denominator.float() / (1.0 - beta2**next_step) ** 0.5
    ).to(state_dtype)
    denominator = (denominator.float() + options["eps"]).to(state_dtype)
    expected_parameter = torch.add(
        expected_parameter.float(),
        expected_mean.float() / denominator.float(),
        alpha=-options["lr"] / (1.0 - beta1**next_step),
    ).to(parameter_dtype)
    actual = functional_adamw(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        **options,
    )
    _assert_storage_precision_close(actual[0], expected_parameter)
    _assert_storage_precision_close(actual[1], expected_mean)
    _assert_storage_precision_close(actual[2], expected_variance)
    assert int(actual[3]) == 8


def test_distinct_master_and_state_dtypes_are_independent():
    torch.manual_seed(14906)
    parameter = torch.nn.Parameter(
        torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    )
    parameter.grad_dtype = torch.float32
    parameter.grad = torch.randn_like(parameter, dtype=torch.float32)
    optimizer = AdamW(
        [parameter],
        gradient_dtype=torch.float32,
        state_dtype=torch.float32,
        master_parameter_dtype=torch.float32,
        lr=2e-4,
        betas=(0.9, 0.95),
    )
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["master_parameter"].dtype == torch.float32
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    assert parameter.dtype == torch.bfloat16

    before = parameter.detach().clone()
    master_adamw_(
        parameter,
        state["master_parameter"],
        parameter.grad,
        state["exp_avg"],
        state["exp_avg_sq"],
        state["step"],
        lr=2e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    assert not torch.equal(parameter, before)


def test_master_out_form_preserves_sources():
    torch.manual_seed(14907)
    parameter = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    master = parameter.float()
    gradient = torch.randn_like(master)
    exp_avg = torch.zeros_like(master)
    exp_avg_sq = torch.zeros_like(master)
    step = torch.zeros((), device="cuda", dtype=torch.int64)
    sources = (parameter, master, gradient, exp_avg, exp_avg_sq, step)
    snapshots = tuple(value.clone() for value in sources)
    destinations = tuple(
        torch.empty_like(value)
        for value in (parameter, master, exp_avg, exp_avg_sq, step)
    )
    expected = functional_master_adamw(
        *sources,
        lr=2e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    returned = master_adamw(
        *sources,
        lr=2e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        out=destinations,
    )
    assert returned is destinations
    assert all(
        torch.equal(value, snapshot)
        for value, snapshot in zip(sources, snapshots, strict=True)
    )
    assert all(
        torch.equal(value, reference)
        for value, reference in zip(destinations, expected, strict=True)
    )


def test_master_alias_uses_no_extra_persistent_tensor():
    parameter = torch.nn.Parameter(
        torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    )
    parameter.grad = torch.randn_like(parameter)
    optimizer = AdamW(
        [parameter],
        master_parameter_dtype=torch.bfloat16,
    )
    optimizer.step()
    assert "master_parameter" not in optimizer.state[parameter]


def test_adamw_rejects_invalid_options_and_sparse_gradients():
    with pytest.raises(ValueError, match="non-negative"):
        AdamW([], lr=-1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        AdamW([], betas=(1.0, 0.95))
    with pytest.raises(ValueError, match="amsgrad"):
        AdamW([], amsgrad=True)
    with pytest.raises(ValueError, match="differentiable"):
        AdamW([], differentiable=True)

    parameter = torch.nn.Parameter(
        torch.randn(32, device="cuda", dtype=torch.bfloat16)
    )
    indices = torch.tensor([[0, 3]], device="cuda")
    values = torch.ones(2, device="cuda", dtype=torch.bfloat16)
    parameter.grad = torch.sparse_coo_tensor(
        indices,
        values,
        (32,),
        check_invariants=True,
    )
    optimizer = AdamW([parameter])
    with pytest.raises(RuntimeError, match="sparse gradients"):
        optimizer.step()
