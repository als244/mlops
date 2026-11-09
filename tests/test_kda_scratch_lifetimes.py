"""Check the live scratch set between KDA kernels without requiring CUDA."""

import sys
import types
import weakref

import pytest

from mlops.kernels.kda import execution


@pytest.mark.parametrize("recompute_intra", [False, True])
def test_backward_releases_private_scratch_before_later_allocations(
    monkeypatch, recompute_intra
):
    references = {}

    class Scratch:
        pass

    def scratch(name):
        result = Scratch()
        references[name] = weakref.ref(result)
        return result

    def released(*names):
        assert all(references[name]() is None for name in names)

    def module(name, **functions):
        value = types.ModuleType(name)
        value.__dict__.update(functions)
        monkeypatch.setitem(sys.modules, name, value)

    def factors(**kwargs):
        return tuple(scratch(name) for name in ("w", "u", "qg", "kg"))

    def intra_forward(**kwargs):
        return (*factors(), scratch("aqk"), scratch("akk"))

    def state_forward(*args):
        return scratch("h"), scratch("v_new")

    def d_av(**kwargs):
        released("u")
        return scratch("daqk"), scratch("dv_first")

    def state_backward(*args):
        if recompute_intra:
            released("aqk")
        return scratch("dh"), scratch("dv_second")

    def wy(**kwargs):
        released("w", "qg", "kg", "dv_first")
        return tuple(
            scratch(name)
            for name in ("dq_part", "dk_part", "dv", "db_part", "dg_part", "dakk")
        )

    def intra_backward(*args):
        released("h", "dh", "v_new", "dv_second")
        if recompute_intra:
            released("akk")
        return tuple(scratch(name) for name in ("dq", "dk", "db", "dg_intra"))

    def scan(*args, **kwargs):
        released("daqk", "dakk", "dq_part", "dk_part", "db_part", "dg_part")
        return scratch("dg")

    module(
        "fla.ops.kda.chunk_bwd", chunk_kda_bwd_dAv=d_av, chunk_kda_bwd_wy_dqkg_fused=wy
    )
    module("fla.ops.kda.wy_fast", recompute_w_u_fwd=factors)
    module("fla.ops.utils", chunk_local_cumsum=scan)
    module("mlops.kernels.kda.intra", chunk_kda_fwd_intra=intra_forward)
    monkeypatch.setattr(execution, "state_forward", state_forward)
    monkeypatch.setattr(execution, "state_backward", state_backward)
    monkeypatch.setattr(execution, "_intra", intra_backward)

    # The non-recomputing path receives caller-owned matrices. Releasing the
    # function's private reference must not invalidate those external owners.
    matrices = (None, None) if recompute_intra else (scratch("aqk"), scratch("akk"))
    result = execution.backward(
        None,
        None,
        None,
        None,
        None,
        *matrices,
        None,
        None,
        None,
        1.0,
        recompute_intra=recompute_intra,
    )
    assert result == tuple(
        references[name]() for name in ("dq", "dk", "dv", "dg", "db")
    )
    live = {name for name, reference in references.items() if reference() is not None}
    expected = {"dq", "dk", "dv", "dg", "db"}
    if not recompute_intra:
        expected.update(("aqk", "akk"))
    assert live == expected
