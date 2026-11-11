"""Bounded positional reads into caller-owned CPU storage."""

import os
from concurrent.futures import ThreadPoolExecutor, wait


class ReadRanges:
    def __init__(self, workers=4, block_bytes=8 << 20):
        if type(workers) is not int or workers < 1:
            raise ValueError("read_workers must be a positive integer")
        if type(block_bytes) is not int or block_bytes < 1:
            raise ValueError("read_chunk_bytes must be a positive integer")
        self.workers = workers
        self.block_bytes = block_bytes
        self.descriptors = {}
        self.pool = None

    @staticmethod
    def _read(descriptor, position, destination):
        completed = 0
        while completed < destination.nbytes:
            count = os.preadv(
                descriptor, [destination[completed:]], position + completed
            )
            if not count:
                raise EOFError("Checkpoint ended before the requested tensor range")
            completed += count

    def read_into(self, path, position, destination):
        if not destination.nbytes:
            return
        if path not in self.descriptors:
            self.descriptors[path] = os.open(path, os.O_RDONLY)
        descriptor = self.descriptors[path]
        if self.workers == 1 or destination.nbytes <= self.block_bytes:
            self._read(descriptor, position, destination)
            return
        if self.pool is None:
            self.pool = ThreadPoolExecutor(
                max_workers=self.workers, thread_name_prefix="checkpoint-read"
            )
        pending = []
        try:
            for offset in range(0, destination.nbytes, self.block_bytes):
                pending.append(
                    self.pool.submit(
                        self._read,
                        descriptor,
                        position + offset,
                        destination[offset : offset + self.block_bytes],
                    )
                )
            for future in pending:
                future.result()
        finally:
            # Even on failure, no worker may retain/write a borrowed destination
            # after its callback returns to the owning storage allocator.
            wait(pending)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None
        for descriptor in self.descriptors.values():
            os.close(descriptor)
        self.descriptors.clear()
