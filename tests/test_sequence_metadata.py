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


def test_a_lengths_tensor_prepares_fixed_shape_metadata():
    # Four slots, the last empty, over 128 tokens: one chunk row for each of
    # the ceil(128 / 64) + 4 chunks they could need, and the rows past the
    # real chunks belong to one more, empty sequence at the end.
    lengths = torch.tensor([73, 38, 17, 0], dtype=torch.int32)
    cumulative, chunks = prepare_packed_sequence_metadata(lengths, torch.empty(128))
    assert cumulative.dtype == chunks.dtype == torch.int64
    assert cumulative.tolist() == [0, 73, 111, 128, 128, 128]
    assert chunks.tolist() == [[0, 0], [0, 1], [1, 0], [2, 0], [4, 0], [4, 1]]


def test_a_lengths_tensor_keeps_its_shapes_across_packings():
    like = torch.empty(128)
    shapes = {
        tuple(tensor.shape)
        for packing in ([128, 0, 0, 0], [1, 1, 1, 125], [64, 64, 0, 0])
        for tensor in prepare_packed_sequence_metadata(
            torch.tensor(packing, dtype=torch.int32), like
        )
    }
    assert shapes == {(6,), (6, 2)}
