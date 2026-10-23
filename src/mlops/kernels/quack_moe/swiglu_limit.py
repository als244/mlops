"""SwiGLU with a limit, as Quack GEMM epilogues.

GLM's experts clamp before the activation: the gate is capped at ``limit``,
the up projection clipped to ``[-limit, limit]``, then ``silu(gate) * up``;
the backward passes no gradient through a clipped value (the builtin
``packed_swiglu(limit=)`` semantics, ``kernels/swiglu.py``). The limit is a
run-time scalar operand of each epilogue, so one compiled kernel serves every
limit. Values reach an epilogue as Float32 scalars on SM90 and as packed
pairs on SM100; the helpers below take either.

The epilogues mirror Quack's own gated ones (``quack.epilogue.library``):

* ``swiglu_limit`` -- ``gemm_act``'s gated forward: the pre-activation pairs
  as ``D``, the clamped activation as ``postact``;
* ``swiglu_limit_add`` -- the same with ``C`` added to the accumulator first
  (a LoRA term entering before the activation);
* ``dswiglu_limit_scored`` -- ``gemm_dact``'s gated backward with a per-row
  score: ``D`` the packed pre-activation gradient of ``score * dout``,
  ``postact`` the score-scaled recomputed activation, ``dscore`` the row's
  ``<activation, dout>``;
* ``scaled_swiglu_limit``, ``scaled_swiglu_limit_add`` and
  ``scaled_dswiglu_limit_scored`` -- the FP8 forms, with the row and weight
  descales applied to the accumulator first.
"""

from __future__ import annotations

from cutlass import Boolean, Float32, const_expr, cute
from cutlass._mlir.dialects import arith
from cutlass.cutlass_dsl import dsl_user_op
from quack.activation import dswiglu, swiglu
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.math import F2, pack, unpack
from quack.epilogue.ops import ColVecLoad, ColVecReduce, RowVecLoad, Scalar


def _lanes(value):
    return value if isinstance(value, tuple) else (value, value)


@dsl_user_op
def minimum(x, y, *, loc=None, ip=None):
    """Lane-wise ``min`` (``arith.minnumf``, as Quack's ``relu`` uses ``maxnumf``)."""
    if const_expr(isinstance(x, tuple)):
        y0, y1 = _lanes(y)
        return F2(minimum(x[0], y0, loc=loc, ip=ip), minimum(x[1], y1, loc=loc, ip=ip))
    return Float32(
        arith.minnumf(
            Float32(x).ir_value(loc=loc, ip=ip),
            Float32(y).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def maximum(x, y, *, loc=None, ip=None):
    """Lane-wise ``max``."""
    if const_expr(isinstance(x, tuple)):
        y0, y1 = _lanes(y)
        return F2(maximum(x[0], y0, loc=loc, ip=ip), maximum(x[1], y1, loc=loc, ip=ip))
    return Float32(
        arith.maxnumf(
            Float32(x).ir_value(loc=loc, ip=ip),
            Float32(y).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


def clamp_gate_up(gate, up, limit):
    """``(min(gate, limit), clamp(up, -limit, limit))``."""
    return minimum(gate, limit), maximum(minimum(up, limit), -limit)


@dsl_user_op
@cute.jit
def _gate_mask(grad, raw, limit, *, loc=None, ip=None):
    """``grad`` where ``raw <= limit``, else zero."""
    if const_expr(isinstance(grad, tuple)):
        r0, r1 = raw
        return F2(
            grad[0] if Boolean(r0 <= limit) else Float32(0.0),
            grad[1] if Boolean(r1 <= limit) else Float32(0.0),
        )
    return grad if Boolean(raw <= limit) else Float32(0.0)


@dsl_user_op
@cute.jit
def _up_mask(grad, raw, limit, *, loc=None, ip=None):
    """``grad`` where ``-limit <= raw <= limit``, else zero."""
    if const_expr(isinstance(grad, tuple)):
        r0, r1 = raw
        return F2(
            grad[0] if Boolean((r0 <= limit) & (r0 >= -limit)) else Float32(0.0),
            grad[1] if Boolean((r1 <= limit) & (r1 >= -limit)) else Float32(0.0),
        )
    return grad if Boolean((raw <= limit) & (raw >= -limit)) else Float32(0.0)


def swiglu_clamped(gate, up, limit):
    """``silu(min(gate, limit)) * clamp(up, -limit, limit)``."""
    g, u = clamp_gate_up(gate, up, limit)
    return swiglu(g, u)


def dswiglu_clamped(gate, up, dout, limit):
    """``(dgate, dup, activation)`` of the clamped SwiGLU at the raw
    ``gate, up``: Quack's ``dswiglu`` at the clamped values, masked where a
    value was clipped."""
    g, u = clamp_gate_up(gate, up, limit)
    dg, du, out = dswiglu(g, u, dout)
    return _gate_mask(dg, gate, limit), _up_mask(du, up, limit), out


_LIMIT = {"limit": Scalar("limit")}


@gemm_epilogue(outputs=("postact",), mode="acc_pair", ops=dict(_LIMIT))
def swiglu_limit(acc, limit):
    gate, up = unpack(acc)
    return {"D": acc, "postact": swiglu_clamped(gate, up, limit)}


@gemm_epilogue(outputs=("postact",), mode="acc_pair", ops=dict(_LIMIT))
def swiglu_limit_add(acc, c, limit):
    value = acc + c
    gate, up = unpack(value)
    return {"D": value, "postact": swiglu_clamped(gate, up, limit)}


@gemm_epilogue(
    outputs=("postact",),
    mode="packed_cd_b16x2",
    ops={"score": ColVecLoad("score"), **_LIMIT},
    reduces={"dscore": ColVecReduce("dscore", scaled=True)},
)
def dswiglu_limit_scored(acc, c, score, limit):
    gate, up = unpack(c)
    dg, du, activation = dswiglu_clamped(gate, up, acc * score, limit)
    return {
        "D": pack(dg, du),
        "postact": activation * score,
        "dscore": (activation, acc),
    }


_SCALES = {"xs": ColVecLoad("xs"), "ws": RowVecLoad("ws")}


@gemm_epilogue(outputs=("postact",), mode="acc_pair", ops={**_SCALES, **_LIMIT})
def scaled_swiglu_limit(acc, xs, ws, limit):
    value = acc * xs * ws
    gate, up = unpack(value)
    return {"D": value, "postact": swiglu_clamped(gate, up, limit)}


@gemm_epilogue(outputs=("postact",), mode="acc_pair", ops={**_SCALES, **_LIMIT})
def scaled_swiglu_limit_add(acc, c, xs, ws, limit):
    value = acc * xs * ws + c
    gate, up = unpack(value)
    return {"D": value, "postact": swiglu_clamped(gate, up, limit)}


@gemm_epilogue(
    outputs=("postact",),
    mode="packed_cd_b16x2",
    ops={**_SCALES, "score": ColVecLoad("score"), **_LIMIT},
    reduces={"dscore": ColVecReduce("dscore", scaled=True)},
)
def scaled_dswiglu_limit_scored(acc, c, xs, ws, score, limit):
    gate, up = unpack(c)
    r = acc * xs * ws
    dg, du, activation = dswiglu_clamped(gate, up, r * score, limit)
    return {"D": pack(dg, du), "postact": activation * score, "dscore": (activation, r)}


__all__ = [
    "dswiglu_clamped",
    "dswiglu_limit_scored",
    "scaled_dswiglu_limit_scored",
    "scaled_swiglu_limit",
    "scaled_swiglu_limit_add",
    "swiglu_clamped",
    "swiglu_limit",
    "swiglu_limit_add",
]
