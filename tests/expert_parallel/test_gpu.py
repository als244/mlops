"""Full/LoRA reference correctness, compiled save/recompute and chunk transport."""

import pytest
from _gate import run, worker_command

pytestmark = [pytest.mark.gpu, pytest.mark.expert_parallel]


def test_fp8_parameter_publication(ep_environment, request):
    root, world_size, timeout = ep_environment
    output = root / "fp8-parameter-publication"
    command = worker_command("publication", world_size, output)
    run(
        command
        + [
            "--outdir",
            str(output),
            "--backend",
            request.config.getoption("--ep-backend"),
        ],
        output,
        timeout,
    )


def test_fp8_quantizer_large_offsets(ep_environment, request):
    if request.config.getoption("--ep-backend") == "te":
        pytest.skip("Quack row quantizer")
    root, world_size, timeout = ep_environment
    output = root / "fp8-quantizer-large-offsets"
    command = worker_command("quantize", world_size, output)
    run(command + ["--outdir", str(output)], output, timeout)


def test_fp8_group_quantizer_distributions(ep_environment, request):
    if request.config.getoption("--ep-backend") == "te":
        pytest.skip("Quack group quantizer")
    root, world_size, timeout = ep_environment
    output = root / "fp8-group-quantizer-distributions"
    command = worker_command("group_quantize", world_size, output)
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


@pytest.mark.parametrize("precision", ["bf16", "fp8_current", "fp8_block"])
@pytest.mark.parametrize("gradient", ["bf16", "fp32"])
def test_portable_lora_resources(ep_environment, request, precision, gradient):
    if request.config.getoption("--ep-backend") == "te":
        pytest.skip("Quack portable LoRA")
    root, world_size, timeout = ep_environment
    output = root / f"portable-lora-{precision}-{gradient}"
    command = worker_command("lora", world_size, output)
    if precision == "fp8_block":
        command += ["--dim", "256", "--hidden", "384"]
    run(
        command
        + [
            "--backend",
            "quack",
            "--portable",
            "--precision",
            precision,
            "--gradient-dtype",
            gradient,
            "--execution",
            "both",
            "--outdir",
            str(output),
        ],
        output,
        timeout,
    )


@pytest.mark.parametrize("gradient", ["bf16", "fp32"])
@pytest.mark.parametrize("chunks,recompute", [(1, False), (4, True)])
def test_live_block_lora_resources(
    ep_environment, request, gradient, chunks, recompute
):
    if request.config.getoption("--ep-backend") == "te":
        pytest.skip("Quack single-copy block LoRA")
    root, world_size, timeout = ep_environment
    output = root / f"live-block-lora-{gradient}-chunks{chunks}"
    command = worker_command("lora", world_size, output)
    if recompute:
        command += ["--recompute"]
    run(
        command
        + [
            "--backend",
            "quack",
            "--precision",
            "fp8_block",
            "--gradient-dtype",
            gradient,
            "--dim",
            "256",
            "--hidden",
            "384",
            "--num-chunks",
            str(chunks),
            "--num-buffers",
            str(min(chunks, 2)),
            "--execution",
            "both",
            "--outdir",
            str(output),
        ],
        output,
        timeout,
    )
