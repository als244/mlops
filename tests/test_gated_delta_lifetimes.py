"""Private lifetimes and upstream parity for the gated-delta composition."""

import sys
import types
import weakref

import pytest
import torch

from mlops.kernels.gated_delta_rule import backward


def test_backward_releases_replay_scratch_before_wy_allocations(monkeypatch):
    references = {}

    def scratch(name):
        value = torch.ones(1)
        references[name] = weakref.ref(value)
        return value

    def released(*names):
        assert all(references[name]() is None for name in names)

    def factors(**kwargs):
        return scratch("w"), scratch("u")

    def state_forward(**kwargs):
        assert kwargs["u"] is references["u"]()
        return scratch("h"), scratch("v_new"), None

    def local_backward(**kwargs):
        released("u")
        return scratch("dv_local")

    def state_backward(**kwargs):
        return scratch("dh"), None, scratch("dv_state")

    def qkwg(**kwargs):
        released("dv_local")
        return tuple(scratch(n) for n in ("dq", "dk", "dw", "dg"))

    def wy(**kwargs):
        released("u", "w", "h", "v_new", "dh")
        return tuple(scratch(n) for n in ("dk2", "dv", "db", "dg2"))

    def scan(value, **kwargs):
        assert value.item() == 2
        return scratch("dg_scan")

    def gate(**kwargs):
        return tuple(scratch(n) for n in ("da", "da_log", "ddt_bias"))

    module = types.ModuleType("fla.ops.gated_delta_rule")
    module.chunk = types.SimpleNamespace(
        recompute_w_u_fwd=factors,
        chunk_gated_delta_rule_fwd_h=state_forward,
        chunk_bwd_dv_local=local_backward,
        chunk_gated_delta_rule_bwd_dhu=state_backward,
        chunk_bwd_dqkwg=qkwg,
        prepare_wy_repr_bwd=wy,
        chunk_local_cumsum=scan,
        gdn_gate_bwd=gate,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = backward(*(None,) * 12, 1.0)
    assert result[1].item() == 2
    assert {n for n, ref in references.items() if ref() is not None} == {
        "dq",
        "dk",
        "dv",
        "db",
        "da",
        "da_log",
        "ddt_bias",
    }


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "dtype,lengths,heads,strided",
    [
        (torch.bfloat16, (137,), (2, 4), False),
        (torch.float16, (31, 65, 9), (2, 4), True),
        (torch.bfloat16, (31, 65, 9), (2, 2), True),
    ],
)
def test_public_backward_matches_upstream_and_preserves_inputs(
    dtype, lengths, heads, strided
):
    pytest.importorskip("fla")
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd

    import mlops
    from mlops import explicit

    torch.manual_seed(271)
    tokens, width = sum(lengths), 64
    hq, hv = heads

    def operand(*shape):
        if strided:
            return torch.randn(*shape[:-1], shape[-1] * 2, device="cuda", dtype=dtype)[
                ..., ::2
            ]
        return torch.randn(*shape, device="cuda", dtype=dtype)

    q = operand(tokens, hq, width).mul_(width**-0.5)
    k = operand(tokens, hq, width).mul_(width**-0.5)
    v = operand(tokens, hv, width)
    beta = operand(tokens, hv).sigmoid_()
    a = operand(tokens, hv).sub_(1)
    a_log = torch.zeros(hv, device="cuda")
    bias = torch.zeros_like(a_log)
    cumulative, chunks = mlops.prepare_packed_sequence_metadata(lengths, q)
    operands = (q, k, v, beta, a, a_log, bias, cumulative, chunks)
    scale = width**-0.5
    with mlops.dispatch.use_implementation(
        "linear_attention", "fla.linear_attention.gated_delta_rule"
    ):
        output, gate, matrix = explicit.linear_attention.forward(*operands, scale=scale)
        grad = operand(tokens, hv, width)
        immutable = (*operands, gate, matrix, grad)
        snapshots = tuple(t.clone() for t in immutable)
        expected = chunk_gated_delta_rule_bwd(
            q=q.contiguous().unsqueeze(0),
            k=k.contiguous().unsqueeze(0),
            v=v.contiguous().unsqueeze(0),
            g=gate,
            beta=beta.contiguous().unsqueeze(0),
            A=matrix,
            scale=scale,
            initial_state=None,
            do=grad.contiguous().unsqueeze(0),
            dht=None,
            cu_seqlens=cumulative if cumulative.numel() else None,
            chunk_indices=chunks if chunks.numel() else None,
            use_gate_in_kernel=True,
            g_input=a.contiguous().unsqueeze(0),
            A_log=a_log,
            dt_bias=bias,
        )
        expected = (*(x.squeeze(0) for x in expected[:5]), *expected[6:])
        for _ in range(2):
            actual = explicit.linear_attention.backward(
                grad,
                q,
                k,
                v,
                beta,
                a,
                a_log,
                bias,
                gate,
                matrix,
                cumulative,
                chunks,
                scale=scale,
            )
            for got, want in zip(actual, expected, strict=True):
                assert torch.isfinite(got).all()
                torch.testing.assert_close(got, want, rtol=0, atol=0)
        for tensor, snapshot in zip(immutable, snapshots, strict=True):
            torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0, equal_nan=True)
        assert torch.isfinite(output).all()
