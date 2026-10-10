"""Precision contracts independent of surrounding mixed-precision autocast."""

import pytest
import torch

from mlops.glm.gates import kda_decay
from mlops.glm.hyper_connection import coefficients, normalize_streams
from mlops.glm.routing import route


@pytest.mark.parametrize("compiled", [False, True])
def test_fp32_control_arithmetic_under_autocast(compiled):
    streams = torch.randn(7, 4, 64, device="cuda", dtype=torch.bfloat16)
    raw = torch.randn(7, 2, 64, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(128, device="cuda")
    a_log = torch.randn(2, device="cuda")
    projected = torch.randn(7, 24, device="cuda")
    base = torch.randn(24, device="cuda")
    scale = torch.randn(3, device="cuda")
    logits = torch.randn(7, 16, device="cuda")
    correction = torch.randn(16, device="cuda")

    def controls(streams, raw, bias, a_log, projected, base, scale, logits, correction):
        normalized = normalize_streams(streams)
        decay = kda_decay(raw, bias, a_log)
        post, mixing, collapsed = coefficients(streams, projected, base, scale)
        _, weights = route(logits, correction, 4)
        return normalized, decay, post, mixing, weights, collapsed

    args = (streams, raw, bias, a_log, projected, base, scale, logits, correction)
    expected = controls(*args)
    fn = torch.compile(controls, fullgraph=True) if compiled else controls
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = fn(*args)
    for a, b in zip(actual, expected):
        assert a.dtype == b.dtype
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
    assert all(value.dtype == torch.float32 for value in actual[:5])
    assert actual[5].dtype == streams.dtype


def test_low_precision_control_projection_rejected():
    logits = torch.randn(7, 16, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="FP32"):
        route(logits, logits.new_zeros(16), 4)
    streams = torch.randn(7, 4, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="FP32"):
        coefficients(
            streams,
            streams.new_zeros(7, 24),
            streams.new_zeros(24),
            streams.new_zeros(3),
        )


pytestmark = pytest.mark.gpu
