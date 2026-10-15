"""Explicit physical weight formats, independent of matrix-multiply precision."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from . import fp8, nvfp4

DENSE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass(frozen=True)
class WeightSpec:
    format: str
    rows: int
    cols: int
    block_rows: int = 1
    block_cols: int = 1
    global_rows: int = 0

    @property
    def count(self):
        return 1 if self.format in DENSE else 3 if self.format == "nvfp4" else 2

    def encode(self):
        return f"{self.format}:{self.rows}:{self.cols}:{self.block_rows}:{self.block_cols}:{self.global_rows}"

    @classmethod
    def parse(cls, text):
        name, *values = text.split(":")
        return cls(name, *map(int, values))

    def empty(self, *, device="meta"):
        """Declare physical components without quantizing an initialization tensor."""
        packed = self.format in {"nvfp4", "mxfp4", "int4"}
        dtype = (
            torch.uint8
            if packed
            else torch.float8_e4m3fn
            if self.format == "fp8"
            else DENSE[self.format]
        )
        parts = [
            torch.empty(
                (self.rows, (self.cols + 1) // 2 if packed else self.cols),
                dtype=dtype,
                device=device,
            )
        ]
        if self.count > 1:
            dtype = (
                torch.float8_e4m3fn
                if self.format == "nvfp4"
                else torch.uint8
                if self.format == "mxfp4"
                else torch.float32
            )
            parts.append(
                torch.empty(
                    (
                        (self.rows + self.block_rows - 1) // self.block_rows,
                        (self.cols + self.block_cols - 1) // self.block_cols,
                    ),
                    dtype=dtype,
                    device=device,
                )
            )
        if self.count == 3:
            n = (
                (self.rows + self.global_rows - 1) // self.global_rows
                if self.global_rows
                else 1
            )
            parts.append(torch.empty(n, dtype=torch.float32, device=device))
        self.validate(parts)
        return parts

    def validate(self, components):
        if self.format not in {*DENSE, "fp8", "nvfp4", "mxfp4", "int4"}:
            raise ValueError(f"Unsupported weight storage {self.format!r}")
        if min(self.rows, self.cols, self.block_rows, self.block_cols) <= 0:
            raise ValueError("Weight shape and scale groups must be positive")
        if len(components) != self.count:
            raise ValueError(f"{self.format} requires {self.count} physical components")
        data = components[0]
        packed = self.format in {"nvfp4", "mxfp4", "int4"}
        shape = (self.rows, (self.cols + 1) // 2 if packed else self.cols)
        dtype = (
            torch.uint8
            if packed
            else torch.float8_e4m3fn
            if self.format == "fp8"
            else DENSE[self.format]
        )
        if tuple(data.shape) != shape or data.dtype != dtype:
            raise ValueError(
                f"{self.format}: expected data {shape} at {dtype}; got {tuple(data.shape)} at {data.dtype}"
            )
        if self.count > 1:
            scale = components[1]
            shape = (
                (self.rows + self.block_rows - 1) // self.block_rows,
                (self.cols + self.block_cols - 1) // self.block_cols,
            )
            if tuple(scale.shape) != shape:
                raise ValueError(
                    f"{self.format}: expected scale shape {shape}, got {tuple(scale.shape)}"
                )
            if self.format == "mxfp4":
                if scale.dtype != torch.uint8:
                    raise ValueError("MXFP4 scales must be E8M0 bytes")
            elif self.format == "nvfp4":
                if scale.dtype != torch.float8_e4m3fn:
                    raise ValueError("NVFP4 block scales must be E4M3")
            elif scale.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise ValueError(f"{self.format} needs floating dequantization scales")
        if self.global_rows < 0:
            raise ValueError("global_rows must be nonnegative")
        globals_count = (
            (self.rows + self.global_rows - 1) // self.global_rows
            if self.global_rows
            else 1
        )
        if self.count == 3 and (
            components[2].numel() != globals_count
            or components[2].dtype != torch.float32
        ):
            raise ValueError(
                f"NVFP4 needs {globals_count} FP32 global dequantization multipliers"
            )
        if any(not v.is_contiguous() for v in components):
            raise ValueError("Weight components must be contiguous")
        if any(v.device != data.device for v in components):
            raise ValueError("Weight components must share a device")


@triton.jit
def _decode(
    DATA,
    SCALE,
    GLOBAL,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    FORMAT: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    GR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // K, i % K
    if FORMAT == "fp8":
        value = tl.load(DATA + i, i < N, other=0.0).to(tl.float32)
    else:
        packed = tl.load(DATA + row * triton.cdiv(K, 2) + col // 2, i < N, other=0)
        code = (packed >> ((col % 2) * 4)) & 15
        if FORMAT == "int4":
            value = code.to(tl.float32) - 8.0
        else:
            mag = code & 7
            value = tl.where(
                mag <= 4,
                mag.to(tl.float32) * 0.5,
                tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0)),
            )
            value = tl.where((code & 8) != 0, -value, value)
    scale = tl.load(
        SCALE + (row // BR) * triton.cdiv(K, BC) + col // BC, i < N, other=0.0
    )
    if FORMAT == "mxfp4":
        scale = tl.exp2(scale.to(tl.float32) - 127.0)
    else:
        scale = scale.to(tl.float32)
    value *= scale
    if FORMAT == "nvfp4":
        value *= tl.load(GLOBAL + row // GR, i < N, other=0) if GR else tl.load(GLOBAL)
    tl.store(OUT + i, value, i < N)


def materialize(components, spec, dtype):
    """Decode one matrix; execution keeps this workspace within one expert.

    On CUDA a kernel decodes it, except E4M3 parts (FP8 data, NVFP4 scales)
    below compute capability 8.9, where Triton has no E4M3 type: those, like
    every other device, take the same equations in PyTorch."""
    if spec.format in DENSE:
        return components[0].to(dtype)
    data, scale, *extra = components
    if data.is_cuda and (
        torch.cuda.get_device_capability(data.device) >= (8, 9)
        or all(part.dtype != torch.float8_e4m3fn for part in components)
    ):
        out = torch.empty(spec.rows, spec.cols, device=data.device, dtype=dtype)
        _decode[(triton.cdiv(out.numel(), 512),)](
            data,
            scale,
            extra[0] if extra else scale,
            out,
            out.numel(),
            spec.cols,
            spec.format,
            spec.block_rows,
            spec.block_cols,
            spec.global_rows,
            512,
        )
        return out
    if spec.format == "fp8":
        value = data.float()
    else:
        code = torch.stack((data & 15, data >> 4), -1).reshape(spec.rows, -1)[
            :, : spec.cols
        ]
        if spec.format == "int4":
            value = code.float() - 8
        else:
            levels = torch.tensor(
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=data.device
            )
            value = levels[(code & 7).long()] * torch.where((code & 8) != 0, -1.0, 1.0)
    scales = (
        torch.exp2(scale.float() - 127) if spec.format == "mxfp4" else scale.float()
    )
    expanded = scales.repeat_interleave(spec.block_rows, 0).repeat_interleave(
        spec.block_cols, 1
    )
    value = value * expanded[: spec.rows, : spec.cols]
    if extra:
        global_scale = extra[0].reshape(-1)
        if spec.global_rows:
            global_scale = global_scale.repeat_interleave(spec.global_rows)[
                : spec.rows, None
            ]
        value = value * global_scale
    return value.to(dtype)


def pack(value, storage, *, nvfp4_block_rows=16):
    """Initialization conversion. Checkpoint import preserves existing components."""
    m, k = value.shape
    if storage in DENSE:
        return [value.to(DENSE[storage]).contiguous()], WeightSpec(storage, m, k)
    if value.is_meta:
        br, bc = (
            (128, 128)
            if storage == "fp8"
            else (nvfp4_block_rows, 16)
            if storage == "nvfp4"
            else (1, 32)
        )
        spec = WeightSpec(storage, m, k, br, bc)
        return spec.empty(device=value.device), spec
    if storage == "fp8":
        return list(fp8.pack_weight(value.contiguous())), WeightSpec(
            "fp8", m, k, 128, 128
        )
    if storage == "nvfp4":
        return list(
            nvfp4.pack_weight(value.contiguous(), block_rows=nvfp4_block_rows)
        ), WeightSpec("nvfp4", m, k, nvfp4_block_rows, 16)
    if storage not in {"mxfp4", "int4"}:
        raise ValueError(f"Unsupported storage {storage!r}")
    group = 32
    padded = torch.nn.functional.pad(value.float(), (0, (-k) % group))
    blocks = padded.reshape(m, -1, group)
    maximum = blocks.abs().amax(-1).clamp_min(1e-12)
    if storage == "int4":
        scales = maximum / 7
        code = (blocks / scales[..., None]).round().clamp(-8, 7).to(torch.int32) + 8
    else:
        exponent = torch.ceil(torch.log2(maximum / 6)).clamp(-127, 127)
        scales = (exponent + 127).to(torch.uint8)
        normalized = blocks / torch.exp2(exponent)[..., None]
        levels = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=value.device
        )
        distance = (normalized.abs()[..., None] - levels).abs()
        code = distance.argmin(-1)
        following = (code + 1).clamp_max(7)
        tied = distance.gather(-1, code[..., None]) == distance.gather(
            -1, following[..., None]
        )
        code = torch.where(tied.squeeze(-1) & (code % 2 == 1), following, code)
        code = code | ((normalized < 0).long() << 3)
    code = code.reshape(m, -1)[:, :k]
    if k % 2:
        code = torch.nn.functional.pad(code, (0, 1))
    data = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8).contiguous()
    return [data, scales.contiguous()], WeightSpec(storage, m, k, 1, group)


def split_components(weights, specs):
    cursor = 0
    result = []
    for spec in specs:
        result.append(weights[cursor : cursor + spec.count])
        cursor += spec.count
    if cursor != len(weights):
        raise ValueError("Physical weight count does not match metadata")
    return result
