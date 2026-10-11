"""Metadata-first access to local Hugging Face safetensors shards.

Opening a shard maps it; tensors are read only when requested. The caller must
consume each tensor before requesting the next if bounded scratch is required.
"""

import json
import math
import mmap
import struct
from pathlib import Path

import torch

DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}

SIZES = {
    name: torch.empty((), dtype=dtype).element_size() for name, dtype in DTYPES.items()
}


class SafeTensors:
    """Read shape/dtype declarations without touching checkpoint payloads."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        index = self.directory / "model.safetensors.index.json"
        self.files = (
            json.loads(index.read_text())["weight_map"] if index.exists() else None
        )
        shards = set(self.files.values()) if self.files else {"model.safetensors"}
        self.info = {}
        self._data_offsets = {}
        self._handles = {}
        self._closed = False
        for name in sorted(shards):
            path = self.path(name)
            with path.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                if length > min(path.stat().st_size - 8, 256 * 1024**2):
                    raise ValueError(f"Invalid safetensors header length: {path}")
                header = json.loads(stream.read(length))
            self._data_offsets[name] = length + 8
            for key, value in header.items():
                if key == "__metadata__":
                    continue
                if self.files is not None and self.files.get(key) != name:
                    continue
                shape = value.get("shape")
                extent = value.get("data_offsets")
                dtype = value.get("dtype")
                if dtype not in DTYPES:
                    raise ValueError(f"Unsupported safetensors dtype {dtype!r}: {key}")
                if (
                    not isinstance(shape, list)
                    or any(not isinstance(n, int) or n < 0 for n in shape)
                    or not isinstance(extent, list)
                    or len(extent) != 2
                    or not all(isinstance(n, int) for n in extent)
                    or not 0
                    <= extent[0]
                    <= extent[1]
                    <= path.stat().st_size - length - 8
                    or extent[1] - extent[0] != math.prod(shape) * SIZES[dtype]
                ):
                    raise ValueError(f"Invalid safetensors shape/extent: {key}")
                self.info[key] = value
        if self.files is None:
            self.files = dict.fromkeys(self.info, "model.safetensors")
        missing = self.files.keys() - self.info.keys()
        if missing:
            raise ValueError(
                f"Index names missing from shard headers: {sorted(missing)[:4]}"
            )

    def path(self, shard):
        relative = Path(shard)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Checkpoint shard path leaves the checkpoint directory")
        # HF snapshot directories normally contain symlinks into the shared
        # blobs cache. Validate the index's relative name, not the link target.
        return self.directory / relative

    def tensor(self, name, *, meta=False):
        if self._closed:
            raise RuntimeError("Checkpoint reader is closed")
        if name not in self.info:
            raise ValueError(f"Checkpoint is missing {name}")
        if meta:
            info = self.info[name]
            if info["dtype"] not in DTYPES:
                raise ValueError(
                    f"Unsupported safetensors dtype {info['dtype']!r}: {name}"
                )
            return torch.empty(
                info["shape"], dtype=DTYPES[info["dtype"]], device="meta"
            )
        shard = self.files[name]
        if shard not in self._handles:
            with self.path(shard).open("rb") as stream:
                # Copy-on-write protects checkpoint bytes even if a caller writes
                # a returned tensor. No private pages are needed for normal reads.
                self._handles[shard] = mmap.mmap(
                    stream.fileno(), 0, access=mmap.ACCESS_COPY
                )
        info = self.info[name]
        dtype = DTYPES[info["dtype"]]
        start, stop = info["data_offsets"]
        count = math.prod(info["shape"])
        if stop - start != count * torch.empty((), dtype=dtype).element_size():
            raise ValueError(f"Invalid safetensors byte extent: {name}")
        offset = self._data_offsets[shard] + start
        if count == 0:
            return torch.empty(info["shape"], dtype=dtype)
        return torch.frombuffer(
            self._handles[shard], dtype=dtype, count=count, offset=offset
        ).reshape(info["shape"])

    def release_pages(self, name):
        """Drop consumed file pages from this reader's RSS, retaining the mapping.

        Call after synchronous consumption. A still-live tensor remains valid:
        a later read simply faults its checkpoint pages back in. Import never
        writes source tensors, so there are no private changes to discard.
        """
        shard = self.files[name]
        mapping = self._handles.get(shard)
        if mapping is None or not hasattr(mapping, "madvise"):
            return
        start, stop = self.info[name]["data_offsets"]
        begin = (self._data_offsets[shard] + start) // mmap.PAGESIZE * mmap.PAGESIZE
        length = self._data_offsets[shard] + stop - begin
        if length:
            mapping.madvise(mmap.MADV_DONTNEED, begin, length)

    def close(self):
        # frombuffer retains its mapping owner: outstanding tensor views remain
        # valid after this reader is closed. Unreferenced mappings close here.
        self._handles.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
