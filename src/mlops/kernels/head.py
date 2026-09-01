"""Shared allocation policy for chunked language-model head operations."""

HEAD_CHUNK_SCRATCH_BYTES = 256 << 20


def default_head_chunk_size(vocab_size: int) -> int:
    """Return a 256-row-aligned chunk near the 256 MiB BF16 logits cap."""
    rows = HEAD_CHUNK_SCRATCH_BYTES // (2 * int(vocab_size))
    return max(512, (rows // 256) * 256)


__all__ = ["HEAD_CHUNK_SCRATCH_BYTES", "default_head_chunk_size"]
