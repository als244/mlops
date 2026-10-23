"""Host counters are dynamic launch arguments, with the same update arithmetic."""

import copy

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


@pytest.mark.parametrize("capturable", [False, True])
def test_declared_counter_placement_and_checkpoint_restore(capturable):
    parameter = torch.nn.Parameter(torch.empty(17, device="meta"))
    parameter.grad = torch.empty_like(parameter)
    optimizer = AdamW([parameter], capturable=capturable)
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["step"].device.type == ("meta" if capturable else "cpu")
    assert state["step"].dtype == torch.int64
    assert state["exp_avg"].device.type == "meta"
    if not capturable:
        state["step"].fill_(23)
        restored = AdamW([parameter])
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        assert restored.state[parameter]["step"].device.type == "cpu"
        assert restored.state[parameter]["step"].item() == 23
        restored.state[parameter]["step"].add_(1)
        assert state["step"].item() == 23


@pytest.mark.parametrize("surface", ["functional", "out", "inplace"])
@pytest.mark.parametrize("master", [False, True])
@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
@pytest.mark.parametrize("step_dtype", [torch.int64, torch.float32])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_and_cuda_counters_produce_identical_updates(
    surface, master, rounding, step_dtype
):
    torch.manual_seed(372)
    weight = torch.randn(2051, device="cuda", dtype=torch.bfloat16)
    initial = [weight]
    if master:
        initial.append(weight.float() + 0.001)
    initial.extend([torch.randn_like(weight), torch.rand_like(weight)])
    states = [
        [
            *(t.clone() for t in initial),
            torch.tensor(0, device=device, dtype=step_dtype),
        ]
        for device in ("cpu", "cuda")
    ]
    functions = (
        (functional_master_adamw, master_adamw, master_adamw_)
        if master
        else (functional_adamw, adamw, adamw_)
    )
    for index, lr in enumerate([1e-3, 0.0, 0.0, 2e-3, 1e-3]):
        gradient = torch.randn_like(weight)
        for side, state in enumerate(states):
            originals = [value.clone() for value in state]
            args = (
                (*state[:2], gradient, *state[2:])
                if master
                else (state[0], gradient, *state[1:])
            )
            options = {
                "lr": lr,
                "betas": (0.8, 0.95),
                "eps": 1e-4,
                "weight_decay": 0.03,
                "parameter_rounding": rounding,
                "opt_state_rounding": rounding,
                "rounding_salt": 19,
            }
            if surface == "functional":
                states[side] = list(functions[0](*args, **options))
            elif surface == "out":
                outputs = tuple(torch.empty_like(t) for t in state)
                functions[1](*args, **options, out=outputs)
                states[side] = list(outputs)
            else:
                functions[2](*args, **options)
            if surface != "inplace" or lr == 0:
                for actual, expected in zip(state, originals, strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for cpu, cuda in zip(*states, strict=True):
            torch.testing.assert_close(cpu.cpu(), cuda.cpu(), rtol=0, atol=0)
        assert states[0][-1].item() == [1, 1, 1, 2, 3][index]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_host_counter_changes_without_retrace_and_resumes():
    from torch._dynamo.testing import CompileCounterWithBackend

    torch.manual_seed(373)
    parameter = torch.nn.Parameter(
        torch.randn(1031, device="cuda", dtype=torch.bfloat16)
    )
    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    options = {
        "lr": 0.002,
        "gradient_dtype": "parameter",
        "opt_state_dtype": torch.float32,
        "parameter_rounding": "stochastic",
    }
    optimizer, reference = (
        AdamW([parameter], **options),
        AdamW([reference_parameter], **options),
    )
    for current in (parameter, reference_parameter):
        current.grad = torch.ones_like(current)
    optimizer.step()
    reference.step()
    torch._dynamo.reset()
    counter = CompileCounterWithBackend("inductor")
    execute = torch.compile(optimizer.step, fullgraph=True, backend=counter)
    for _ in range(4):
        execute()
        reference.step()
    assert counter.frame_count == 1
    assert optimizer.state[parameter]["step"].device.type == "cpu"
    assert optimizer.state[parameter]["step"].item() == 5
    torch.testing.assert_close(parameter, reference_parameter, rtol=0, atol=0)
    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed_parameter.grad = torch.ones_like(resumed_parameter)
    resumed = AdamW([resumed_parameter], **options)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    execute()
    resumed.step()
    torch.testing.assert_close(parameter, resumed_parameter, rtol=0, atol=0)
    for name, value in optimizer.state[parameter].items():
        torch.testing.assert_close(
            value, resumed.state[resumed_parameter][name], rtol=0, atol=0
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_explicit_capturable_counter_advances_on_cuda_graph_replay():
    parameter = torch.nn.Parameter(torch.ones(1024, device="cuda"))
    parameter.grad = torch.ones_like(parameter)
    optimizer = AdamW(
        [parameter],
        capturable=True,
        gradient_dtype="parameter",
        opt_state_dtype=torch.float32,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        optimizer.step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        optimizer.step()
    before = optimizer.state[parameter]["step"].item()
    for _ in range(3):
        graph.replay()
    assert optimizer.state[parameter]["step"].device.type == "cuda"
    assert optimizer.state[parameter]["step"].item() == before + 3


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_lazy_initialization_keeps_explicit_cpu_counter_policy():
    torch._dynamo.reset()
    parameter = torch.nn.Parameter(torch.ones(17, device="cuda"))
    parameter.grad = torch.ones_like(parameter)
    optimizer = AdamW([parameter], gradient_dtype="parameter")
    torch.compile(optimizer.step, fullgraph=True)()
    assert optimizer.state[parameter]["step"].device.type == "cpu"
    assert optimizer.state[parameter]["step"].item() == 1
    assert optimizer.state_dict()["param_groups"][0]["capturable"] is False
