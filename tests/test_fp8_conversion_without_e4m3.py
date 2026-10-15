"""FP8 conversion on GPUs without E4M3. Below compute capability 8.9 Triton has
no E4M3 type, so decoding E4M3 checkpoint parts and quantizing 128x128 blocks
take the PyTorch equations there and agree with the same equations on the
CPU."""

import pytest
import torch

from mlops.kernels.block_fp8 import quantize_blocks
from mlops.sequential_moe.precision.storage import WeightSpec, materialize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture
def ampere(monkeypatch):
    """Every GPU reports compute capability 8.6, as an RTX 3090 does."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 6))


def _parts(storage, generator):
    """Stored parts of a 256x384 matrix: E4M3 data under 128x128 FP32 scales,
    or packed FP4 codes under 16x16 E4M3 scales and a global scale."""
    rows, cols = 256, 384
    if storage == "fp8":
        data = torch.randn(rows, cols, generator=generator).to(torch.float8_e4m3fn)
        scale = torch.rand(2, 3, generator=generator) + 0.5
        return [data, scale], WeightSpec("fp8", rows, cols, 128, 128)
    codes = torch.randint(0, 256, (rows, cols // 2), generator=generator)
    scale = torch.rand(rows // 16, cols // 16, generator=generator) + 0.5
    parts = [codes.to(torch.uint8), scale.to(torch.float8_e4m3fn), torch.tensor(0.75)]
    return parts, WeightSpec("nvfp4", rows, cols, 16, 16)


@pytest.mark.parametrize("storage", ["fp8", "nvfp4"])
def test_decoding_e4m3_parts_matches_the_cpu(ampere, storage):
    parts, spec = _parts(storage, torch.Generator().manual_seed(0))
    expected = materialize(parts, spec, torch.float32)
    decoded = materialize([part.cuda() for part in parts], spec, torch.float32)
    assert torch.equal(decoded.cpu(), expected)


def test_block_quantization_matches_the_cpu(ampere):
    weight = torch.randn(2, 256, 384, generator=torch.Generator().manual_seed(0))
    data, scale = quantize_blocks(weight.cuda())
    expected_data, expected_scale = quantize_blocks(weight)
    # CUDA divides by a scalar through its reciprocal, so a scale may differ
    # from the CPU's in its last bit; the E4M3 values come out the same here.
    torch.testing.assert_close(scale.cpu(), expected_scale, rtol=2**-22, atol=0)
    assert torch.equal(data.cpu().float(), expected_data.float())
