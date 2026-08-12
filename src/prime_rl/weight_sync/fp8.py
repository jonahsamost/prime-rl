"""Deterministic FP8 checkpoint conversion and resident XOR state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch
from torch import Tensor

FP8_BLOCK_SIZE = 128
FP8ScaleFormat = Literal["float32", "ue8m0"]


def quantize_to_fp8_blockwise(
    weight: Tensor,
    block_size: int = FP8_BLOCK_SIZE,
) -> tuple[Tensor, Tensor]:
    """Quantize one matrix to E4M3 with a deterministic FP32 scale per block."""
    if weight.ndim != 2:
        raise ValueError(f"FP8 quantization expects a 2D tensor, got shape={tuple(weight.shape)}")
    if block_size <= 0:
        raise ValueError(f"FP8 block size must be positive, got {block_size}")

    rows, cols = weight.shape
    pad_rows = (-rows) % block_size
    pad_cols = (-cols) % block_size
    if pad_rows or pad_cols:
        padded = torch.zeros(
            rows + pad_rows,
            cols + pad_cols,
            dtype=weight.dtype,
            device=weight.device,
        )
        padded[:rows, :cols].copy_(weight)
    else:
        padded = weight.contiguous()

    padded_rows, padded_cols = padded.shape
    blocks = padded.view(
        padded_rows // block_size,
        block_size,
        padded_cols // block_size,
        block_size,
    ).permute(0, 2, 1, 3)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scales = (blocks.float().abs().amax(dim=(2, 3)) / fp8_max).clamp(min=1e-12)
    blocks_fp8 = (blocks.float() / scales[:, :, None, None]).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    quantized = blocks_fp8.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)[:rows, :cols].contiguous()
    return quantized, scales.float().contiguous()


def quantize_to_vllm_kernel_format(
    weight: Tensor,
    block_size: int = FP8_BLOCK_SIZE,
    *,
    scale_format: FP8ScaleFormat = "float32",
) -> tuple[Tensor, Tensor]:
    """Quantize a matrix into an explicitly selected vLLM kernel representation."""
    quantized, scales = quantize_to_fp8_blockwise(weight, block_size)
    if scale_format == "float32":
        return quantized, scales
    if scale_format != "ue8m0":
        raise ValueError(f"unsupported FP8 scale format {scale_format!r}")
    if not weight.is_cuda:
        raise ValueError("vLLM UE8M0 FP8 post-processing requires a CUDA tensor")

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )

    return deepgemm_post_process_fp8_weight_block(
        wq=quantized,
        ws=scales,
        quant_block_shape=(block_size, block_size),
        use_e8m0=True,
    )


def xor_tensor_bytes(left: Tensor, right: Tensor) -> Tensor:
    """Return the bytewise XOR of two identically represented tensors."""
    if left.dtype != right.dtype or left.shape != right.shape:
        raise ValueError(
            "FP8 resident XOR operands must have identical shapes and dtypes, "
            f"got {tuple(left.shape)}/{left.dtype} and {tuple(right.shape)}/{right.dtype}"
        )
    if left.device != right.device:
        raise ValueError(f"FP8 kernel XOR operands are on different devices: {left.device} and {right.device}")
    if not left.is_contiguous() or not right.is_contiguous():
        raise ValueError("FP8 resident XOR operands must be contiguous")
    delta = torch.empty_like(right)
    delta.view(torch.uint8).copy_(left.view(torch.uint8)).bitwise_xor_(right.view(torch.uint8))
    return delta


@dataclass(frozen=True)
class FP8ResidentSnapshot:
    """One version of TP-local tensors consumed directly by inference kernels."""

    step: int
    tensors: dict[str, Tensor]


class FP8ResidentDeltaState:
    """Track the exact resident FP8 representation used as the next XOR base."""

    def __init__(self) -> None:
        self._snapshot: FP8ResidentSnapshot | None = None

    @property
    def snapshot(self) -> FP8ResidentSnapshot | None:
        return self._snapshot

    def initialize(self, step: int, tensors: Mapping[str, Tensor]) -> FP8ResidentSnapshot:
        snapshot = FP8ResidentSnapshot(step=step, tensors=_owned_resident_tensors(tensors))
        self._snapshot = snapshot
        return snapshot

    def advance(
        self,
        *,
        base_step: int,
        step: int,
        tensors: Mapping[str, Tensor],
    ) -> tuple[FP8ResidentSnapshot, dict[str, Tensor]]:
        previous = self._snapshot
        if previous is None:
            raise RuntimeError("FP8 resident delta state has no full-snapshot base")
        if previous.step != base_step or step != base_step + 1:
            raise ValueError(
                f"FP8 resident updates must be consecutive from the resident base, "
                f"got resident={previous.step}, requested={base_step}->{step}"
            )
        current_tensors = _owned_resident_tensors(tensors)
        if current_tensors.keys() != previous.tensors.keys():
            missing = sorted(previous.tensors.keys() - current_tensors.keys())
            added = sorted(current_tensors.keys() - previous.tensors.keys())
            raise ValueError(f"FP8 resident tensor set changed between updates: missing={missing}, added={added}")
        deltas = {name: xor_tensor_bytes(previous.tensors[name], current) for name, current in current_tensors.items()}
        snapshot = FP8ResidentSnapshot(step=step, tensors=current_tensors)
        self._snapshot = snapshot
        return snapshot, deltas


def _owned_resident_tensors(tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    if not tensors:
        raise ValueError("an FP8 resident snapshot must contain at least one tensor")
    owned: dict[str, Tensor] = {}
    for name, tensor in tensors.items():
        if not name:
            raise ValueError("FP8 resident tensor names must be non-empty")
        if not tensor.is_floating_point() and tensor.dtype not in (torch.uint8, torch.int32):
            raise TypeError(f"FP8 resident tensor {name!r} has unsupported dtype {tensor.dtype}")
        if tensor.numel() == 0:
            raise ValueError(f"FP8 resident tensor {name!r} is empty")
        owned[name] = tensor.detach().contiguous()
    return owned


__all__ = [
    "FP8_BLOCK_SIZE",
    "FP8ResidentDeltaState",
    "FP8ResidentSnapshot",
    "FP8ScaleFormat",
    "quantize_to_fp8_blockwise",
    "quantize_to_vllm_kernel_format",
    "xor_tensor_bytes",
]
