"""Bound MoonEP planner PTXAS optimization for the pinned CuTe DSL stack.

CuTe DSL 4.7.1 with default PTXAS optimization faults in stock MoonEP planning
for, among other shapes, EP2/E192/K4/65536 tokens. Level 2 avoids that failure.
Only the planner's compilation call changes; communication and GEMM kernels
retain their usual compiler settings. Installed dependency files stay intact.
"""

import hashlib
import inspect
import linecache
import textwrap
from importlib.metadata import version


def apply():
    """Install once for the tested MoonEP revision and CuTe DSL 4.7.1."""
    from moonep import planning

    if getattr(planning, "_mlops_planner_opt_level", None) == 2:
        return False
    if version("nvidia-cutlass-dsl") != "4.7.1":
        return False

    source = textwrap.dedent(inspect.getsource(planning._get_compiled))
    expected = "c9e60caa592d444666a9ba430a78a3ab2e26f271668820955a4126f2bf05ffa5"
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise RuntimeError("Unsupported MoonEP revision for planner compiler fix")
    original = planning._get_compiled
    source = source.replace(
        "Int32(0), cuda.CUstream(0))",
        'Int32(0), cuda.CUstream(0), options="--ptxas-options --opt-level=2")',
    )
    filename = "<mlops_moonep_planner_compiler>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(vars(planning))
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 -- source hash checked
    planning._get_compiled = namespace["_get_compiled"]
    original.cache_clear()
    planning._mlops_planner_opt_level = 2
    return True
