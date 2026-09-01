"""Public semantic tensor operations with ordinary PyTorch autograd."""

from . import optim as optim
from .causal_conv import causal_conv_silu
from .cross_entropy import cross_entropy
from .dsa import (
    dsa_attention,
)
from .dsa_indexer import (
    dsa_indexer_key_norm,
    dsa_selection_override,
    index_scores,
    selection_mask,
    topk_indices,
)
from .embedding import embedding
from .flash_attention import flash_attention
from .gelu import gelu
from .gated_rms_norm import gated_rms_norm
from .head import head_loss
from .l2_norm import l2_norm
from .layer_norm import layer_norm
from .linear_attention import linear_attention
from .linear_mixer import linear_mixer
from .mla import mla_attention
from .moe import moe
from .partial_rope import partial_rope
from .rms_norm import rms_norm
from .rope import rope
from .rope_tables import build_rope_tables
from .sequence_metadata import prepare_packed_sequence_metadata
from .swiglu import packed_swiglu, swiglu

__all__ = [
    "causal_conv_silu",
    "cross_entropy",
    "dsa_attention",
    "dsa_indexer_key_norm",
    "dsa_selection_override",
    "index_scores",
    "selection_mask",
    "topk_indices",
    "embedding",
    "flash_attention",
    "gelu",
    "gated_rms_norm",
    "head_loss",
    "l2_norm",
    "layer_norm",
    "linear_attention",
    "linear_mixer",
    "mla_attention",
    "moe",
    "partial_rope",
    "packed_swiglu",
    "prepare_packed_sequence_metadata",
    "rms_norm",
    "rope",
    "build_rope_tables",
    "swiglu",
]
