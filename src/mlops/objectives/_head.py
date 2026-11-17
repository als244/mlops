"""Bind the built-in formulas to head row inputs and runtime controls."""

from .token import clipped_policy_with_gradient, cross_entropy_with_gradient

BUILTINS = ("cross_entropy", "clipped_policy")


def validate(name, inputs, controls):
    if name == "cross_entropy":
        if len(inputs) != 2 or controls:
            raise ValueError(
                "cross_entropy requires mask and weights, without controls"
            )
    elif name == "clipped_policy":
        if (len(inputs), len(controls)) not in ((4, 1), (5, 2)):
            raise ValueError(
                "clipped_policy requires mask, weights, old logprobs, advantages and "
                "clip ratio; reference logprobs and beta must be supplied together"
            )
    else:
        raise ValueError(f"unknown built-in head objective: {name}")
    import torch

    if inputs[0].dtype != torch.bool:
        raise ValueError("head objective mask must have boolean dtype")


def evaluate(name, logprobs, inputs, controls):
    if name == "cross_entropy":
        return cross_entropy_with_gradient(logprobs, *inputs)
    if name == "clipped_policy":
        return clipped_policy_with_gradient(
            logprobs,
            *inputs[:4],
            controls[0],
            reference_logprobs=inputs[4] if len(inputs) == 5 else None,
            reference_beta=controls[1] if len(controls) == 2 else None,
        )
    raise ValueError(f"unknown built-in head objective: {name}")
