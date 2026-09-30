"""BF16 moment defaults and independent explicit state choices."""

import pytest
import torch

from mlops.optim import AdamW


@pytest.mark.parametrize(
    "state_dtype", [None, torch.float32, torch.float16, "parameter"]
)
def test_meta_step_declares_default_and_explicit_moment_dtypes(state_dtype):
    parameter = torch.nn.Parameter(torch.empty(16, device="meta", dtype=torch.float16))
    parameter.grad = torch.empty_like(parameter)
    options = {} if state_dtype is None else {"opt_state_dtype": state_dtype}
    optimizer = AdamW([parameter], gradient_dtype="parameter", **options)
    optimizer.step()
    expected = (
        torch.bfloat16
        if state_dtype is None
        else torch.float16
        if state_dtype == "parameter"
        else state_dtype
    )
    assert optimizer.state[parameter]["exp_avg"].dtype == expected
    assert optimizer.state[parameter]["exp_avg_sq"].dtype == expected
    assert parameter.dtype == torch.float16


@pytest.mark.gpu
def test_fp16_parameter_explicit_fp32_moments_remain_finite_and_restore():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    parameter = torch.nn.Parameter(torch.ones(2, device="cuda", dtype=torch.float16))
    optimizer = AdamW([parameter], gradient_dtype="parameter", opt_state_dtype=torch.float32)
    parameter.grad = torch.tensor([0.0, 1e-5], device="cuda", dtype=torch.float16)
    optimizer.step()
    state = optimizer.state[parameter]
    assert torch.isfinite(parameter).all()
    assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    checkpoint = optimizer.state_dict()
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = AdamW([restored_parameter], gradient_dtype="parameter", opt_state_dtype=torch.float32)

    def check_restored_dtype(optimizer):
        assert optimizer.state[restored_parameter]["exp_avg"].dtype == torch.float32

    restored.register_load_state_dict_post_hook(check_restored_dtype)
    restored.load_state_dict(checkpoint)
    for name in ("exp_avg", "exp_avg_sq", "step"):
        assert torch.equal(restored.state[restored_parameter][name], state[name])
        assert restored.state[restored_parameter][name].dtype == state[name].dtype

    # A resumed update must match an uninterrupted one, including tiny moments
    # which would already have lost information if cast through FP16 on load.
    for current in (parameter, restored_parameter):
        current.grad = torch.tensor([0.01, -1e-5], device="cuda", dtype=torch.float16)
    optimizer.step()
    restored.step()
    assert torch.equal(parameter, restored_parameter)
    for name in ("exp_avg", "exp_avg_sq", "step"):
        assert torch.equal(restored.state[restored_parameter][name], state[name])


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16, "parameter"])
def test_checkpoint_preserves_explicit_moment_policy_and_load_hooks(state_dtype):
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.float16))
    optimizer = AdamW([parameter], opt_state_dtype=state_dtype)
    optimizer._initialize_parameter_state(
        parameter, optimizer.param_groups[0], optimizer.state[parameter]
    )
    optimizer.state[parameter]["exp_avg"].fill_(0.123456)
    saved = optimizer.state_dict()
    restored_parameter = torch.nn.Parameter(torch.ones_like(parameter))
    restored = AdamW([restored_parameter])
    visited = []

    def change_checkpoint(_optimizer, incoming):
        visited.append("pre")
        replacement = {
            **incoming,
            "state": {key: dict(value) for key, value in incoming["state"].items()},
        }
        replacement["state"][0]["exp_avg"] = incoming["state"][0]["exp_avg"] * 2
        return replacement

    def inspect_checkpoint(loaded):
        visited.append("post")
        actual = loaded.state[restored_parameter]["exp_avg"]
        expected = saved["state"][0]["exp_avg"] * 2
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)

    restored.register_load_state_dict_pre_hook(change_checkpoint)
    restored.register_load_state_dict_post_hook(inspect_checkpoint)
    restored.load_state_dict(saved)
    assert visited == ["pre", "post"]
    assert restored.param_groups[0]["opt_state_dtype"] == state_dtype
    assert len(restored._optimizer_load_state_dict_pre_hooks) == 1
    assert len(restored._optimizer_load_state_dict_post_hooks) == 1
