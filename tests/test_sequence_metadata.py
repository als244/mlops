"""Caller-owned packed-sequence metadata preparation."""

from __future__ import annotations

import pytest
import torch

from mlops import prepare_packed_sequence_metadata
from mlops.preparation.packed_sequence import _prepare_op


def test_prepare_packed_sequence_metadata_values_on_cpu():
    cumulative, chunks = prepare_packed_sequence_metadata(
        (73, 38, 17), torch.empty(0)
    )
    assert cumulative.dtype == torch.int64
    assert cumulative.tolist() == [0, 73, 111, 128]
    assert chunks.tolist() == [[0, 0], [0, 1], [1, 0], [2, 0]]


def test_prepare_packed_sequence_metadata_single_sequence_uses_empty_sentinel():
    cumulative, chunks = prepare_packed_sequence_metadata((128,), torch.empty(0))
    assert cumulative.shape == (0,)
    assert chunks.shape == (0, 2)


def test_prepare_packed_sequence_metadata_is_a_graph_visible_custom_op():
    class Prepare(torch.nn.Module):
        def forward(self, like):
            return prepare_packed_sequence_metadata((65, 64), like)

    # Real model anchors are differentiable hidden states; integer metadata
    # remains correctly non-differentiable without an autograd registration.
    anchor = torch.empty(0, requires_grad=True)
    torch.library.opcheck(_prepare_op, (anchor, [65, 64], 64))
    exported = torch.export.export(Prepare(), (anchor,))
    targets = {str(node.target) for node in exported.graph.nodes}
    assert "mlops.prepare_packed_sequence_metadata.default" in targets


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prepare_packed_sequence_metadata_materializes_caller_owned_cuda_inputs():
    cumulative, chunks = prepare_packed_sequence_metadata(
        (65, 64), torch.empty(0, device="cuda")
    )
    assert cumulative.is_cuda and chunks.is_cuda
    assert cumulative.cpu().tolist() == [0, 65, 129]
    assert chunks.cpu().tolist() == [[0, 0], [0, 1], [1, 0]]
