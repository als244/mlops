"""Full/LoRA reference correctness, compiled save/recompute and chunk transport."""

import pytest
from _gate import run, worker_command

pytestmark = [pytest.mark.gpu, pytest.mark.expert_parallel]


def test_fp8_quantizer_large_offsets(ep_environment, request):
    if request.config.getoption("--ep-backend") == "te":
        pytest.skip("Quack row quantizer")
    root, world_size, timeout = ep_environment
    output = root / "fp8-quantizer-large-offsets"
    command = worker_command("quantize", world_size, output)
    run(command + ["--outdir", str(output)], output, timeout)



def test_moonep_planner_large_token_counts(ep_environment):
    root, world_size, timeout = ep_environment
    output = root / "moonep-planner-32k-64k"
    command = worker_command("moonep", world_size, output)
    run(command + ["--outdir", str(output)], output, timeout)


def test_reference_correctness(ep_environment, ep_case):
    root, world_size, timeout = ep_environment
    output = root / ep_case.name
    run(ep_case.command(world_size, output), output, timeout)


def test_public_layers_coexist(ep_environment, request):
    if request.config.getoption("--ep-backend") != "both":
        pytest.skip("coexistence check requires both optional backends")
    root, world_size, timeout = ep_environment
    output = root / "public-layers-coexist"
    command = worker_command("coexist", world_size, output)
    run(command + ["--outdir", str(output)], output, timeout)


def test_shared_expert_banks(ep_environment, request):
    if request.config.getoption("--ep-backend") == "te":
        pytest.skip("Quack expert-bank reuse")
    root, world_size, timeout = ep_environment
    output = root / "shared-expert-banks"
    command = worker_command("shared_banks", world_size, output)
    run(command + ["--outdir", str(output)], output, timeout)
