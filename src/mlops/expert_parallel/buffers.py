"""Caller-owned MoonEP buffers for the optional expert-parallel modules."""

from contextlib import contextmanager


def create_buffer(config, tokens, ep_group=None):
    from ._compat import moonep_rank1  # noqa: F401

    if tokens % getattr(config, "num_chunks", 1):
        raise ValueError("Equal-size chunks require divisible token count")
    if getattr(config, "num_chunks", 1) == 1:
        return _single_buffer(config, tokens, ep_group)
    from .quack.pipeline.buffers import ChunkBufferPool

    buffers = []
    try:
        for _ in range(getattr(config, "num_buffers", 1)):
            buffers.append(
                _single_buffer(
                    config, tokens // getattr(config, "num_chunks", 1), ep_group
                )
            )
        return ChunkBufferPool(buffers, num_chunks=getattr(config, "num_chunks", 1))
    except BaseException:
        for buffer in buffers:
            buffer.destroy()
        raise


def _single_buffer(config, tokens, ep_group=None):
    from moonep import Buffer

    return Buffer(
        S=tokens,
        H=config.feature_dim,
        K=config.top_k,
        E=config.num_experts,
        num_ep_ranks=config.ep_size,
        B=config.local_experts,
        num_sms=config.num_comm_sms,
        token_padding=config.token_padding,
        group=ep_group,
        enable_pdl=False,
        explicitly_destroy=True,
    )


@contextmanager
def buffer_context(config, tokens, ep_group=None):
    buffer = create_buffer(config, tokens, ep_group)
    try:
        yield buffer
    finally:
        buffer.destroy()
