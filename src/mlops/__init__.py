"""Public semantic tensor operations with ordinary PyTorch autograd."""

from . import optim as optim
from .causal_conv import causal_conv_silu
from .cross_entropy import cross_entropy
from .dispatch.context import apply_environment_overrides
from .dsa import (
    dsa_attention,
    sparse_latent_attention,
)
from .dsa_indexer import (
    dsa_indexer_key_norm,
    dsa_selection_override,
    index_scores,
    pooled_topk,
    selection_mask,
    topk_indices,
)
from .embedding import embedding
from .flash_attention import flash_attention
from .gated_rms_norm import gated_rms_norm
from .gelu import gelu
from .head import head_loss
from .hyper_connection import (
    mhc_coefficients,
    mhc_combine,
    mhc_normalize_streams,
    mhc_pre,
)
from .kda import kda_decay, kimi_delta_attention
from .l2_norm import l2_norm
from .layer_norm import layer_norm
from .linear_attention import linear_attention
from .linear_mixer import linear_mixer
from .lora_head import lora_head_loss
from .mla import mla_attention, sparse_mla
from .moe import moe
from .partial_rope import partial_rope
from .rms_norm import rms_norm
from .rope import rope
from .rope_tables import build_rope_tables
from .routing import sigmoid_topk
from .sequence_metadata import prepare_packed_sequence_metadata
from .swiglu import packed_swiglu, swiglu

# One choice for a whole process tree, if the environment makes it.
_environment_choices = apply_environment_overrides()

__all__ = [
    "build_rope_tables",
    "causal_conv_silu",
    "cross_entropy",
    "dsa_attention",
    "dsa_indexer_key_norm",
    "dsa_selection_override",
    "embedding",
    "flash_attention",
    "gated_rms_norm",
    "gelu",
    "head_loss",
    "index_scores",
    "kda_decay",
    "kimi_delta_attention",
    "l2_norm",
    "layer_norm",
    "linear_attention",
    "linear_mixer",
    "lora_head_loss",
    "mhc_coefficients",
    "mhc_combine",
    "mhc_normalize_streams",
    "mhc_pre",
    "mla_attention",
    "moe",
    "packed_swiglu",
    "partial_rope",
    "pooled_topk",
    "prepare_packed_sequence_metadata",
    "rms_norm",
    "rope",
    "selection_mask",
    "sigmoid_topk",
    "sparse_latent_attention",
    "sparse_mla",
    "swiglu",
    "topk_indices",
]
