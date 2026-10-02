"""Process-local singleton fix for stock MoonEP; no installed files are edited.

Imported by the package before buffer construction. The same file ships in each
standalone implementation. Multi-rank groups retain MoonEP's multicast behavior.
"""

import functools
import hashlib
import inspect
import linecache
import textwrap


def apply():
    """Apply once per process, including when both MoE packages are imported."""
    from moonep import api, buffer, planning

    if getattr(planning, "_moe_rank1_patched", False):
        return False

    # MoonEP 0.0.1 has multiple source revisions. Check the three affected
    # functions before changing anything (tested commit 2bd860b4dd083df62b79d5e916fca71ec5742228).
    sources = []
    for function, expected in (
        (
            api._create_context,
            "bcfedc41a58be4feff098d9553046e02dd863de9969efa82cf710d9c77d8f02b",
        ),
        (
            buffer._create_nvl_multicast_view,
            "bf3676614e95c764ffe11c6615eb367552255c9cfd4f4b730a2d3b226cc7fff4",
        ),
        (
            planning.PlanningKernel.kernel,
            "a2dd3be0bcd62fa75ca177694008efa24d7593a0a95e347c85a4d8782324fd1b",
        ),
    ):
        source = textwrap.dedent(inspect.getsource(function))
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            raise RuntimeError(
                "Unsupported MoonEP revision for the rank-1 fix: "
                + function.__qualname__
            )
        sources.append(source)

    # CuTe inspects source during compilation, so give the replacement kernel
    # its own in-memory source entry. The installed source and its cache stay intact.
    store = "            multimem_st_v4(addr.ir_value(), a0, a1, a2, a3)\n"
    source = sources[-1].replace(
        store, "            if cutlass.const_expr(R > 1):\n    " + store
    )
    filename = "<patch_moonep_rank1>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(vars(planning))
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 -- source hash checked above
    kernel = namespace["kernel"]
    kernel.__qualname__ = planning.PlanningKernel.kernel.__qualname__
    inspect.unwrap(kernel).__qualname__ = kernel.__qualname__

    original_granularity = api.get_multicast_granularity
    original_view = buffer._create_nvl_multicast_view

    @functools.wraps(original_granularity)
    def granularity(world_size):
        return (
            api.get_vmm_granularity()
            if world_size == 1
            else original_granularity(world_size)
        )

    @functools.wraps(original_view)
    def multicast_view(
        meta_buf, owned_handle, local_rank, world_size, group=None, use_fabric=False
    ):
        if world_size == 1:
            return meta_buf
        return original_view(
            meta_buf, owned_handle, local_rank, world_size, group, use_fabric
        )

    api.get_multicast_granularity = granularity
    buffer._create_nvl_multicast_view = multicast_view
    planning.PlanningKernel.kernel = kernel
    planning._get_compiled.cache_clear()
    planning._moe_rank1_patched = True
    return True


apply()
