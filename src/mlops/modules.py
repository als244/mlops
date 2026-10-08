"""Small model-building modules backed by MLOps operations."""

from torch import nn

from .head import head_loss


class LanguageModelHead(nn.Linear):
    """Ordinary logits through forward; bounded projection/cross entropy through loss.

    Parameters and initialization follow nn.Linear. The loss path requires
    bias=False. A module interface lets model definitions use the same calls
    when a transformation replaces this head with a low-rank version.
    """

    def __init__(self, in_features, out_features, *, device=None, dtype=None):
        super().__init__(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )

    def loss(
        self, hidden, targets, *, chunk_size=None, valid_rows=None, reduction="mean"
    ):
        return head_loss(
            hidden,
            self.weight,
            targets,
            chunk_size=chunk_size,
            valid_rows=valid_rows,
            reduction=reduction,
        )


__all__ = ["LanguageModelHead"]
