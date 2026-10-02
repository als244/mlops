"""Deterministic SM90 expert GEMM settings, selected once at layer construction.

The exported table records measured choices for D=7168, H=2048, E=64,
K=8, EP=2 and padding=128. Other shapes retain the original fixed settings.
Autotuning is an explicit alternative and does not consume this table.
"""

from dataclasses import asdict, dataclass

from quack.gemm_config import GemmConfig

FP8_SM90_CONFIG = GemmConfig(
    tile_m=128,
    tile_n=96,
    pingpong=True,
    cluster_m=2,
    is_dynamic_persistent=False,
    device_capacity=9,
)
BF16_SM90_CONFIG = GemmConfig(
    tile_m=128,
    tile_n=192,
    pingpong=True,
    cluster_m=2,
    is_dynamic_persistent=False,
    device_capacity=9,
)


def _tile(m, n, *, pingpong, cluster=(2, 1), swap_ab=False):
    return {
        "tile_m": m,
        "tile_n": n,
        "pingpong": pingpong,
        "cluster_m": cluster[0],
        "cluster_n": cluster[1],
        "swap_ab": swap_ab,
        "is_dynamic_persistent": False,
        "device_capacity": 9,
    }


# Inclusive token-count bounds for the measured model shape below.
_MEASURED = {
    "bf16": {
        2048: {
            "up": _tile(128, 256, pingpong=False, cluster=(1, 2)),
            "down": _tile(128, 256, pingpong=False, cluster=(1, 1)),
            "down_backward": _tile(128, 128, pingpong=True, cluster=(1, 2)),
            "input_gradient": _tile(128, 256, pingpong=False, cluster=(1, 2)),
            "up_weight_gradient": _tile(128, 192, pingpong=True, swap_ab=True),
            "down_weight_gradient": _tile(128, 192, pingpong=True, swap_ab=True),
        },
        4096: {
            "up": _tile(128, 256, pingpong=False),
            "down": _tile(128, 192, pingpong=True),
            "down_backward": _tile(128, 128, pingpong=True, cluster=(1, 2)),
            "input_gradient": _tile(128, 256, pingpong=False),
            "up_weight_gradient": _tile(
                192, 128, pingpong=True, cluster=(1, 2), swap_ab=True
            ),
            "down_weight_gradient": _tile(128, 192, pingpong=True, swap_ab=True),
        },
        8192: {
            "up": _tile(256, 192, pingpong=False, cluster=(1, 2)),
            "down": _tile(192, 128, pingpong=True, cluster=(1, 2)),
            "down_backward": _tile(128, 192, pingpong=True),
            "input_gradient": _tile(256, 192, pingpong=False, cluster=(1, 2)),
            "up_weight_gradient": _tile(
                192, 128, pingpong=True, cluster=(1, 2), swap_ab=True
            ),
            "down_weight_gradient": _tile(192, 128, pingpong=True, cluster=(1, 2)),
        },
        16384: {
            "up": _tile(192, 128, pingpong=True, cluster=(1, 2)),
            "down": _tile(256, 192, pingpong=False, cluster=(1, 2)),
            "down_backward": _tile(128, 192, pingpong=True),
            "input_gradient": _tile(256, 192, pingpong=False, cluster=(1, 2)),
            "up_weight_gradient": _tile(256, 192, pingpong=False, cluster=(1, 2)),
            "down_weight_gradient": _tile(
                256, 160, pingpong=False, cluster=(1, 2), swap_ab=True
            ),
        },
        32768: {
            "up": _tile(256, 160, pingpong=False, cluster=(1, 2)),
            "down": _tile(192, 128, pingpong=True, cluster=(1, 2)),
            "down_backward": _tile(192, 128, pingpong=True, cluster=(1, 2)),
            "input_gradient": _tile(256, 192, pingpong=False, cluster=(1, 2)),
            "up_weight_gradient": _tile(256, 192, pingpong=False, cluster=(1, 2)),
            "down_weight_gradient": _tile(
                256, 192, pingpong=False, cluster=(1, 2), swap_ab=True
            ),
        },
    },
    "fp8_current": {
        2048: {
            "up": _tile(128, 128, pingpong=False, cluster=(1, 2)),
            "down": _tile(128, 128, pingpong=False, cluster=(1, 2)),
            "down_backward": _tile(128, 96, pingpong=True),
            "input_gradient": _tile(128, 96, pingpong=True),
            "up_weight_gradient": _tile(128, 96, pingpong=True),
            "down_weight_gradient": _tile(128, 96, pingpong=True),
        },
        4096: {
            "up": _tile(128, 128, pingpong=False, cluster=(1, 2)),
            "down": _tile(128, 128, pingpong=False, cluster=(1, 2)),
            "down_backward": _tile(64, 128, pingpong=True),
            "input_gradient": _tile(128, 128, pingpong=False, cluster=(1, 2)),
            "up_weight_gradient": _tile(64, 128, pingpong=True),
            "down_weight_gradient": _tile(64, 128, pingpong=True, cluster=(1, 2)),
        },
        8192: {
            "up": _tile(128, 128, pingpong=False, cluster=(1, 2)),
            "down": _tile(128, 128, pingpong=False, cluster=(1, 2)),
            "down_backward": _tile(128, 96, pingpong=True),
            "input_gradient": _tile(128, 128, pingpong=False, cluster=(1, 2)),
            "up_weight_gradient": _tile(
                128, 128, pingpong=False, cluster=(1, 2), swap_ab=True
            ),
            "down_weight_gradient": _tile(64, 128, pingpong=True, swap_ab=True),
        },
        16384: {
            "up": _tile(128, 128, pingpong=False, cluster=(1, 2)),
            "down": _tile(128, 128, pingpong=False),
            "down_backward": _tile(128, 96, pingpong=True, cluster=(1, 2)),
            "input_gradient": _tile(128, 128, pingpong=False, cluster=(1, 2)),
            "up_weight_gradient": _tile(
                128, 128, pingpong=False, cluster=(1, 2), swap_ab=True
            ),
            "down_weight_gradient": _tile(128, 128, pingpong=False, swap_ab=True),
        },
        32768: {
            "up": _tile(128, 96, pingpong=True),
            "down": _tile(128, 96, pingpong=True),
            "down_backward": _tile(128, 96, pingpong=True),
            "input_gradient": _tile(128, 96, pingpong=True),
            "up_weight_gradient": _tile(128, 96, pingpong=True),
            "down_weight_gradient": _tile(128, 96, pingpong=True),
        },
    },
}


@dataclass(frozen=True)
class ExpertGemmPolicy:
    up: GemmConfig | None
    down: GemmConfig | None
    down_backward: GemmConfig | None
    input_gradient: GemmConfig | None
    up_weight_gradient: GemmConfig | None
    down_weight_gradient: GemmConfig | None
    feature_dim: int | None = None
    expert_hidden_dim: int | None = None
    token_bucket: int | None = None

    def as_dict(self):
        """Expose the actual choices for benchmark and model diagnostics."""
        return asdict(self)

    def weight_gradient(self, input_features, output_features):
        if (input_features, output_features) == (
            self.feature_dim,
            None if self.expert_hidden_dim is None else 2 * self.expert_hidden_dim,
        ):
            return self.up_weight_gradient
        return self.down_weight_gradient


def select_policy(model_config=None, *, precision, tuned=False):
    """Use only host metadata; no allocation, synchronization, or runtime search."""
    if precision not in ("bf16", "fp8_current"):
        raise ValueError(f"Unsupported GEMM precision: {precision}")
    default = (
        None
        if tuned
        else (BF16_SM90_CONFIG if precision == "bf16" else FP8_SM90_CONFIG)
    )
    names = (
        "up",
        "down",
        "down_backward",
        "input_gradient",
        "up_weight_gradient",
        "down_weight_gradient",
    )
    fields = dict.fromkeys(names, default)
    if model_config is None:
        return ExpertGemmPolicy(**fields)
    c = model_config
    dimensions = {
        "feature_dim": c.feature_dim,
        "expert_hidden_dim": c.expert_hidden_dim,
    }
    measured_shape = (
        c.feature_dim,
        c.expert_hidden_dim,
        c.num_experts,
        c.top_k,
        c.ep_size,
        c.token_padding,
    ) == (7168, 2048, 64, 8, 2, 128)
    tokens = c.tokens_per_rank
    if tuned or not measured_shape or tokens is None or tokens > 32768:
        return ExpertGemmPolicy(**fields, **dimensions)
    table = _MEASURED.get(precision, {})
    bucket = next((limit for limit in sorted(table) if tokens <= limit), None)
    if bucket is None:
        return ExpertGemmPolicy(**fields, **dimensions)
    fields.update(
        {name: GemmConfig(**config) for name, config in table[bucket].items()}
    )
    return ExpertGemmPolicy(**fields, **dimensions, token_bucket=bucket)
