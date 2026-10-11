"""Read one expert from local Hugging Face safetensors without model-wide loading.

Supported encodings are explicit. Unknown compression/packing is rejected; bytes
are never interpreted from a dtype name alone.
"""

import json
import re
import sys
from pathlib import Path

import torch

from .safetensors import SafeTensors
from .storage import DENSE, WeightSpec


class HFWeights:
    def __init__(self, directory):
        self.directory = Path(directory)
        config_path = self.directory / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        self.quant = (
            config.get("quantization_config")
            or config.get("text_config", {}).get("quantization_config")
            or {}
        )
        self._check_encoding()
        self.source = SafeTensors(directory)
        self.files = self.source.files

    def close(self):
        self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _check_encoding(self):
        q = self.quant
        method = q.get("quant_method")
        self.encoding = "dense"
        self.group = None
        if method is None:
            if (self.directory / "hf_quant_config.json").exists():
                raise ValueError(
                    "ModelOpt checkpoint encoding is not supported by this importer yet; do not relabel it as compressed-tensors"
                )
            return
        if method == "fp8":
            if q.get("fmt", "e4m3") != "e4m3" or q.get("weight_block_size") != [
                128,
                128,
            ]:
                raise ValueError(
                    "FP8 checkpoint importer requires E4M3 weight_block_size=[128,128]"
                )
            self.encoding = "fp8"
            return
        if method != "compressed-tensors":
            raise ValueError(
                f"Unsupported checkpoint quant_method={method!r}; AWQ/GPTQ/NF4/ModelOpt packing is not inferred"
            )
        fmt = q.get("format")
        if fmt == "mixed-precision":
            self.encoding = "mixed"
            return
        supported = {
            "pack-quantized": "int4",
            "mxfp4-pack-quantized": "mxfp4",
            "nvfp4-pack-quantized": "nvfp4",
        }
        if fmt not in supported:
            raise ValueError(f"Unsupported compressed-tensors format={fmt!r}")
        self.encoding = supported[fmt]
        schemes = [g.get("weights") for g in q.get("config_groups", {}).values()]
        if not schemes or any(s is None for s in schemes):
            raise ValueError("Checkpoint needs explicit weight quantization metadata")
        unique = {json.dumps(s, sort_keys=True) for s in schemes}
        if len(unique) != 1:
            raise ValueError(
                "Multiple distinct weight quantization schemes need an explicit per-projection importer"
            )
        w = schemes[0]
        expected_type = "int" if self.encoding == "int4" else "float"
        self.group = w.get("group_size")
        if (
            w.get("num_bits") != 4
            or w.get("type") != expected_type
            or not w.get("symmetric", True)
            or w.get("actorder") not in (None, False)
            or not isinstance(self.group, int)
            or self.group <= 0
        ):
            raise ValueError(
                "Supported packed checkpoints require symmetric 4-bit groups without activation ordering"
            )
        if self.encoding in ("mxfp4", "nvfp4"):
            expected = 32 if self.encoding == "mxfp4" else 16
            if self.group != expected:
                raise ValueError(f"{self.encoding} requires group_size={expected}")
        if sys.byteorder != "little" and self.encoding == "int4":
            raise ValueError(
                "INT4 int32 unpacking currently requires a little-endian host"
            )

    def _get(self, name, *, meta=False):
        return self.source.tensor(name, meta=meta).contiguous()

    def _encoding_for(self, prefix):
        if self.encoding != "mixed":
            return self.encoding, self.group
        groups = [
            group
            for group in self.quant.get("config_groups", {}).values()
            if any(
                (re.fullmatch(target[3:], prefix) is not None)
                if target.startswith("re:")
                else target == prefix
                for target in group.get("targets", [])
            )
        ]
        if len(groups) != 1:
            raise ValueError(
                f"{prefix}: expected one matching quantization recipe, got {len(groups)}"
            )
        group = groups[0]
        weights = group.get("weights", {})
        if (
            group.get("format") == "nvfp4-pack-quantized"
            and weights.get("num_bits") == 4
            and weights.get("type") == "float"
            and weights.get("group_size") == 16
            and weights.get("symmetric", True)
            and weights.get("strategy") == "tensor_group"
            and weights.get("actorder") in (None, False)
        ):
            return "nvfp4", 16
        if (
            group.get("format") == "float-quantized"
            and weights.get("num_bits") == 8
            and weights.get("type") == "float"
            and weights.get("block_structure") == [128, 128]
            and weights.get("symmetric", True)
        ):
            return "fp8", None
        raise ValueError(
            f"{prefix}: unsupported mixed checkpoint quantization recipe {group}"
        )

    def projection(self, prefix, shape, *, meta=False):
        """Return CPU packed components and metadata for one [out,in] matrix."""
        rows, cols = shape
        get = lambda name: self._get(name, meta=meta)
        encoding, group_size = self.encoding, self.group
        if prefix + ".weight" in self.files:
            data = get(prefix + ".weight")
            dense = next(
                (name for name, dtype in DENSE.items() if data.dtype == dtype), None
            )
            if dense:
                parts, spec = [data], WeightSpec(dense, rows, cols)
            elif (
                self._encoding_for(prefix)[0] if encoding == "mixed" else encoding
            ) == "fp8":
                parts = [
                    data,
                    get(
                        prefix
                        + (
                            ".weight_scale"
                            if self.encoding == "mixed"
                            else ".weight_scale_inv"
                        )
                    ),
                ]
                spec = WeightSpec("fp8", rows, cols, 128, 128)
            else:
                raise ValueError(
                    f"{prefix}: weight dtype {data.dtype} has no supported checkpoint encoding"
                )
        else:
            encoding, group_size = self._encoding_for(prefix)
            if encoding not in ("int4", "mxfp4", "nvfp4"):
                raise ValueError(
                    f"{prefix}: unsupported packed checkpoint encoding {encoding!r}"
                )
            data = get(prefix + ".weight_packed")
            if (
                prefix + ".weight_zero_point" in self.files
                or prefix + ".weight_g_idx" in self.files
            ):
                raise ValueError(
                    "Explicit zero points/reordered groups are not supported"
                )
            if encoding == "int4":
                expected = (rows, (cols + 7) // 8)
                if data.dtype != torch.int32 or tuple(data.shape) != expected:
                    raise ValueError(f"INT4 expects int32 weight_packed {expected}")
                # Eight offset-binary nibbles per int32, low nibble first.
                data = (
                    data.view(torch.uint8)
                    .reshape(rows, -1)[:, : ((cols + 1) // 2)]
                    .contiguous()
                )
            if not meta and prefix + ".weight_shape" in self.files:
                declared = tuple(get(prefix + ".weight_shape").reshape(-1).tolist())
                if declared != tuple(shape):
                    raise ValueError(
                        f"Checkpoint weight_shape={declared} does not match {shape}"
                    )
            parts = [data, get(prefix + ".weight_scale")]
            if encoding == "nvfp4":
                inverse_global = (
                    get(prefix + ".weight_global_scale").float().reshape(-1)
                )
                if (
                    inverse_global.numel() != 1
                    or (not meta and not torch.isfinite(inverse_global).all())
                    or (not meta and (inverse_global <= 0).any())
                ):
                    raise ValueError(
                        "NVFP4 weight_global_scale must be one positive finite value"
                    )
                parts.append(inverse_global.reciprocal())
            spec = WeightSpec(encoding, rows, cols, 1, group_size)
        spec.validate(parts)
        if not meta and spec.count > 1:
            scales = parts[1]
            if spec.format == "mxfp4":
                if (scales == 255).any():
                    raise ValueError("MXFP4 E8M0 scale 255 encodes NaN")
            elif not torch.isfinite(scales.float()).all() or (scales.float() < 0).any():
                raise ValueError("Weight scales must be finite and nonnegative")
        return parts, spec


def join_gate_up(gate, up):
    """Combine projections losslessly, preserving independent global scales."""
    a, sa = gate
    b, sb = up
    if sa != sb:
        raise ValueError("Gate/up need matching shape, storage, and scale layout")
    if sa.rows % sa.block_rows:
        raise ValueError("Gate/up boundary must align with weight scale blocks")
    parts = [torch.cat((x, y), 0) for x, y in zip(a[:2], b[:2], strict=True)]
    global_rows = 0
    if sa.format == "nvfp4":
        if sa.global_rows:
            raise ValueError("Nested per-projection global scales are unsupported")
        parts.append(torch.cat((a[2].reshape(-1), b[2].reshape(-1))))
        global_rows = sa.rows
    spec = WeightSpec(
        sa.format, 2 * sa.rows, sa.cols, sa.block_rows, sa.block_cols, global_rows
    )
    spec.validate(parts)
    return parts, spec


def load_hf_expert(expert, checkpoint, *, gate_prefix, up_prefix, down_prefix):
    """Load explicit tensor prefixes; no whole-model state_dict is constructed."""
    source = checkpoint if isinstance(checkpoint, HFWeights) else HFWeights(checkpoint)
    d, h = expert.config.hidden_size, expert.config.expert_size
    gate = source.projection(gate_prefix, (h, d))
    up = source.projection(up_prefix, (h, d))
    down = source.projection(down_prefix, (d, h))
    return expert.load_weights_(gate_up=join_gate_up(gate, up), down=down)
