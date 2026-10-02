"""Reuse communication scratch across layers without sharing model parameters."""

from weakref import WeakValueDictionary


def projection_banks(runtime, factory):
    """Borrow matching banks from the token buffer for this runtime's lifetime.

    Weak entries keep resource ownership with live layers. Closing one layer
    does not release banks still used by another; closing the last releases them.
    Forward and backward both publish their own parameters before prefetching.
    """
    config = runtime.cfg
    shapes = (
        (2 * config.expert_hidden_dim, config.feature_dim),
        (config.feature_dim, config.expert_hidden_dim),
    )
    if not config.share_expert_banks:
        return [factory(*shape) for shape in shapes]

    from .config import config_signature

    buffer = runtime.buffer
    cache = getattr(buffer, "_quack_expert_banks", None)
    if cache is None:
        cache = buffer._quack_expert_banks = WeakValueDictionary()
    signature = (
        type(runtime),
        config_signature(config),
        repr(getattr(runtime, "lora", None)),
    )
    banks = []
    for index, shape in enumerate(shapes):
        key = (*signature, index)
        bank = cache.get(key)
        if bank is None:
            bank = factory(*shape)
            cache[key] = bank
        banks.append(bank)
    return banks
