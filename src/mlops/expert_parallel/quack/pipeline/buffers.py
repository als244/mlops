"""Caller-owned pool of identical stock MoonEP buffers for equal token chunks."""


class ChunkBufferPool:
    def __init__(self, buffers, *, num_chunks):
        if type(num_chunks) is not int or num_chunks < 1:
            raise ValueError("num_chunks must be a positive integer")
        if not buffers or not 1 <= len(buffers) <= num_chunks:
            raise ValueError("Require 1 <= buffer count <= chunk count")
        if len({id(buffer) for buffer in buffers}) != len(buffers):
            raise ValueError("Each pool slot requires a distinct MoonEP buffer")
        self.buffers = tuple(buffers)
        self.num_chunks = num_chunks
        base = buffers[0]._require_ctx()
        for buffer in buffers[1:]:
            other = buffer._require_ctx()
            for name in (
                "S",
                "H",
                "K",
                "E",
                "R",
                "B",
                "device",
                "num_sms",
                "token_padding",
            ):
                if base[name] != other[name]:
                    raise ValueError(f"Chunk buffers disagree on {name}")
            if base.get("group") is not other.get("group"):
                raise ValueError("Chunk buffers must use the same EP group")
        self._comm_stream = buffers[0]._comm_stream

    def _require_ctx(self):
        # This descriptor lets the ordinary public layer bind the whole input.
        # Actual MoonEP calls always use a real chunk buffer and its own context.
        ctx = dict(self.buffers[0]._require_ctx())
        ctx["S"] *= self.num_chunks
        return ctx

    @property
    def hidden_nvsh_buffer_view(self):
        return self.buffers[0].hidden_nvsh_buffer_view

    @property
    def destroyed(self):
        return all(buffer.destroyed for buffer in self.buffers)

    def destroy(self):
        for buffer in self.buffers:
            buffer.destroy()
