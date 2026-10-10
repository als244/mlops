"""Sigmoid router selection, with correction bias used only for selection."""

import torch


def route(
    logits,
    correction_bias,
    top_k,
    *,
    groups=1,
    selected_groups=1,
    scale=2.5,
    normalize=True,
):
    if logits.ndim != 2 or correction_bias.shape != (logits.shape[-1],):
        raise ValueError(
            "Expected logits [tokens, experts] and correction_bias [experts]"
        )
    experts = logits.shape[-1]
    if groups < 1 or experts % groups or not 1 <= selected_groups <= groups:
        raise ValueError("Invalid expert grouping")
    if not 1 <= top_k <= selected_groups * (experts // groups):
        raise ValueError("top_k exceeds the selected expert capacity")
    if logits.dtype != torch.float32:
        raise ValueError(
            "Router logits must be computed in FP32, not cast after a low-precision projection"
        )
    scores = logits.sigmoid()
    # Selection is discrete. The bias must not change the combination weights.
    choice = scores.detach() + correction_bias.detach().float()
    if groups > 1:
        width = experts // groups
        if width < 2:
            raise ValueError(
                "Grouped selection requires at least two experts per group"
            )
        group_scores = (
            choice.unflatten(-1, (groups, width)).topk(2, dim=-1).values.sum(-1)
        )
        selected = group_scores.topk(selected_groups, dim=-1, sorted=False).indices
        keep = torch.zeros_like(group_scores, dtype=torch.bool).scatter(
            1, selected, True
        )
        choice = choice.masked_fill(
            ~keep.repeat_interleave(width, dim=-1), float("-inf")
        )
    ids = choice.topk(top_k, dim=-1, sorted=False).indices
    weights = scores.gather(-1, ids)
    if normalize:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return ids, weights * scale
