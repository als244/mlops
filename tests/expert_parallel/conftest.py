"""Explicit opt-in for the optional Hopper/NCCL correctness matrix."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from _gate import cases, check_environment


def positive(value):
    result = int(value)
    if result <= 0:
        raise ValueError("must be positive")
    return result


def pytest_addoption(parser):
    options = parser.getgroup("expert-parallel")
    options.addoption(
        "--run-expert-parallel",
        action="store_true",
        help="Run optional H100/SM90 expert-parallel GPU correctness checks",
    )
    options.addoption("--ep-backend", choices=("quack", "te", "both"), default="both")
    options.addoption("--ep-world-size", type=int, choices=(1, 2, 4, 8), default=2)
    options.addoption(
        "--ep-output", type=Path, help="New artifact directory for this gate run"
    )
    options.addoption(
        "--ep-timeout",
        type=positive,
        default=600,
        help="Maximum seconds per torchrun case (default: 600)",
    )


def pytest_generate_tests(metafunc):
    if "ep_case" in metafunc.fixturenames:
        selected = cases(metafunc.config.getoption("--ep-backend"))
        metafunc.parametrize("ep_case", selected, ids=[case.name for case in selected])


def pytest_collection_finish(session):
    config = session.config
    if (
        config.option.collectonly
        or not config.getoption("--run-expert-parallel")
        or not any(item.get_closest_marker("expert_parallel") for item in session.items)
    ):
        return
    try:
        check_environment(
            config.getoption("--ep-backend"), config.getoption("--ep-world-size")
        )
    except RuntimeError as error:
        raise pytest.UsageError(str(error)) from error


@pytest.fixture(scope="session")
def ep_environment(request):
    config = request.config
    if not config.getoption("--run-expert-parallel"):
        pytest.skip("opt-in Hopper/NCCL check: pass --run-expert-parallel")
    output = config.getoption("--ep-output")
    if output is None:
        output = (
            Path(__file__).parent
            / "results"
            / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        )
    return (
        output.resolve(),
        config.getoption("--ep-world-size"),
        config.getoption("--ep-timeout"),
    )
