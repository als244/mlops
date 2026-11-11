"""Direct checkpoint ranges preserve bytes, bounds and borrowed-buffer lifetimes."""

import os
import time

import pytest
import torch
from safetensors.torch import save_file

from mlops.sequential_moe.precision._read_ranges import ReadRanges
from mlops.sequential_moe.precision.safetensors import SafeTensors


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.uint8]
)
def test_positional_ranges_exact_without_mapping_or_staging(tmp_path, dtype):
    value = (torch.arange(512) % 16).to(dtype).reshape(16, 32)
    save_file({"weight": value}, str(tmp_path / "model.safetensors"))
    with SafeTensors(tmp_path, read_chunk_bytes=31) as reader:
        assert reader.tensor("weight", meta=True).is_meta
        assert not reader._ranges.descriptors and reader._ranges.pool is None
        target = torch.empty(127, dtype=dtype)
        reader.copy_into("weight", 17, target)
        torch.testing.assert_close(
            target.view(torch.uint8), value.view(-1)[17:144].view(torch.uint8)
        )
        assert not reader._handles
        descriptors = list(reader._ranges.descriptors.values())
        converted = torch.empty(13, dtype=torch.float64)
        reader.copy_into("weight", 9, converted)
        torch.testing.assert_close(converted, value.view(-1)[9:22].double())
        for offset in (-1, 500):
            with pytest.raises(ValueError, match="bounds"):
                reader.copy_into("weight", offset, target)
        with pytest.raises(ValueError, match="contiguous"):
            reader.copy_into("weight", 0, target[::2])
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    with pytest.raises(RuntimeError, match="closed"):
        reader.copy_into("weight", 0, target)


def test_short_reads_are_completed_and_truncation_is_rejected(tmp_path, monkeypatch):
    expected = torch.arange(100, dtype=torch.float32)
    path = tmp_path / "model.safetensors"
    save_file({"weight": expected, "empty": expected[:0]}, str(path))
    original = os.preadv
    monkeypatch.setattr(
        os,
        "preadv",
        lambda fd, buffers, offset: original(fd, [buffers[0][:13]], offset),
    )
    with SafeTensors(tmp_path, read_chunk_bytes=37) as reader:
        reader.copy_into("empty", 0, expected[:0])
        assert not reader._ranges.descriptors
        destination = torch.empty_like(expected)
        reader.copy_into("weight", 0, destination)
        torch.testing.assert_close(destination, expected, rtol=0, atol=0)
        with path.open("r+b") as stream:
            stream.truncate(path.stat().st_size - 3)
        with pytest.raises(EOFError, match="ended"):
            reader.copy_into("weight", 0, destination)


def test_failed_read_waits_for_all_borrowed_buffer_users(tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(bytes(64))
    reader = ReadRanges(workers=2, block_bytes=16)
    completed = []

    def read(descriptor, position, destination):
        if position == 0:
            raise OSError("test read failure")
        time.sleep(0.01)
        destination[:] = bytes([1]) * destination.nbytes
        completed.append(position)

    monkeypatch.setattr(reader, "_read", read)
    destination = bytearray(64)
    try:
        with pytest.raises(OSError, match="test read failure"):
            reader.read_into(path, 0, memoryview(destination))
        assert sorted(completed) == [16, 32, 48]
        assert destination[16:] == bytes([1]) * 48
    finally:
        reader.close()
