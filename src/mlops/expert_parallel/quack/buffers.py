"""Validate the supplied MoonEP buffer's geometry and process group."""

from dataclasses import replace

import torch.distributed as dist


def bind_buffer(config, buffer, group, device):
    if buffer is None:
        raise TypeError("buffer must be an initialized caller-owned moonep.Buffer")
    ctx = buffer._require_ctx()
    expected = {
        "H": config.feature_dim,
        "K": config.top_k,
        "E": config.num_experts,
        "R": config.ep_size,
        "B": config.local_experts,
        "token_padding": config.token_padding,
        "device": device.index,
        "num_sms": config.num_comm_sms,
    }
    for key, value in expected.items():
        if int(ctx[key]) != value:
            raise ValueError(
                f"MoonEP buffer {key}={ctx[key]} does not match layer {value}"
            )
    world = dist.group.WORLD
    buffer_group = ctx.get("group")
    if buffer_group is None:
        buffer_group = world
    if group is not None and group is not buffer_group:
        raise ValueError("MoonEP buffer belongs to a different EP process group")
    if config.tokens_per_rank is not None and config.tokens_per_rank != int(ctx["S"]):
        raise ValueError("Legacy token hint disagrees with the supplied MoonEP buffer")
    # No allocation: the derived value describes the supplied resource's shape.
    return replace(config, tokens_per_rank=int(ctx["S"])), buffer_group
