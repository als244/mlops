"""Diagnostic AdamW performs the update without advancing tensor state."""

import pytest
import torch

from mlops.optim import (
    AdamW,
    adamw,
    adamw_,
    functional_adamw,
    functional_master_adamw,
    master_adamw,
    master_adamw_,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def assert_bits_equal(actual, expected):
    torch.testing.assert_close(
        actual.reshape(-1).view(torch.uint8),
        expected.reshape(-1).view(torch.uint8),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("master", [False, True])
@pytest.mark.parametrize("surface", ["functional", "out", "inplace"])
@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_zero_lr_preserves_all_tensor_bytes(dtype, master, surface, rounding):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 needs SM80")
    torch.manual_seed(920)
    parameter = torch.randn(2049, device="cuda", dtype=dtype)
    gradient = torch.randn_like(parameter)
    gradient[:4] = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0], device="cuda", dtype=dtype
    )
    mean = torch.randn(2049, device="cuda", dtype=torch.float32)
    variance = torch.rand_like(mean)
    step = torch.tensor(7, device="cuda", dtype=torch.int64)
    state = (parameter, mean, variance, step)
    arguments = (parameter, gradient, mean, variance, step)
    if master:
        # A compute copy need not equal a new deterministic downcast.
        master_weight = parameter.float() + 0.123
        state = (parameter, master_weight, mean, variance, step)
        arguments = (parameter, master_weight, gradient, mean, variance, step)
    originals = tuple(t.clone() for t in state)
    functions = (
        (functional_master_adamw, master_adamw, master_adamw_)
        if master
        else (functional_adamw, adamw, adamw_)
    )
    options = dict(
        lr=torch.tensor(0.0, dtype=torch.float64),
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        parameter_rounding=rounding,
        opt_state_rounding=rounding,
        rounding_salt=19,
    )
    random_before = torch.cuda.get_rng_state()
    if surface == "functional":
        outputs = functions[0](*arguments, **options)
    elif surface == "out":
        outputs = tuple(torch.empty_like(t) for t in state)
        functions[1](*arguments, **options, out=outputs)
    else:
        functions[2](*arguments, **options)
        outputs = state
    for actual, original in zip(outputs, originals, strict=True):
        assert_bits_equal(actual, original)
    for actual, original in zip(state, originals, strict=True):
        assert_bits_equal(actual, original)
    assert_bits_equal(torch.cuda.get_rng_state(), random_before)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_diagnostic_steps_leave_the_next_real_stochastic_update_unchanged(
    compiled, dtype
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 needs SM80")
    torch.manual_seed(921)
    parameter = torch.nn.Parameter(torch.randn(4097, device="cuda", dtype=dtype))
    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    arguments = dict(
        lr=1e-3,
        gradient_dtype="parameter",
        opt_state_dtype=dtype,
        parameter_rounding="stochastic",
        opt_state_rounding="stochastic",
    )
    optimizer = AdamW([parameter], **arguments)
    reference = AdamW([reference_parameter], **arguments)
    gradient = torch.randn_like(parameter)
    parameter.grad = gradient.clone()
    reference_parameter.grad = gradient.clone()
    # Initialize and populate nonzero moments before capture.
    optimizer.step()
    reference.step()
    execute = (
        torch.compile(optimizer.step, fullgraph=True) if compiled else optimizer.step
    )
    rate = optimizer.param_groups[0]["lr"]
    for _ in range(2):
        rate.fill_(0.0)
        originals = tuple(
            t.clone() for t in (parameter, *optimizer.state[parameter].values())
        )
        execute()
        for value, before in zip(
            (parameter, *optimizer.state[parameter].values()), originals, strict=True
        ):
            assert_bits_equal(value, before)
    rate.fill_(1e-3)
    execute()
    reference.step()
    assert_bits_equal(parameter, reference_parameter)
    for name, value in optimizer.state[parameter].items():
        assert_bits_equal(value, reference.state[reference_parameter][name])


def test_first_zero_lr_call_initializes_only_zero_state():
    parameter = torch.nn.Parameter(torch.ones(17, device="cuda", dtype=torch.float16))
    parameter.grad = torch.ones_like(parameter)
    optimizer = AdamW(
        [parameter], lr=0, gradient_dtype="parameter", opt_state_dtype=torch.float32
    )
    optimizer.step()
    assert_bits_equal(parameter, torch.ones_like(parameter))
    assert all(
        torch.count_nonzero(t).item() == 0 for t in optimizer.state[parameter].values()
    )
