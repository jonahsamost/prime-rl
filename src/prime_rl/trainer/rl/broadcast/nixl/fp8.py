"""Trainer-side production of checkpoint and resident FP8 tensors."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import prod
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.weight_sync.fp8 import FP8ResidentDeltaState, FP8ResidentSnapshot, FP8ScaleFormat
from prime_rl.weight_sync.fp8_resident import parse_resident_tensor_name, resident_tensor_name
from prime_rl.weight_sync.grouping import LAYER_RE, weight_transfer_group_name
from prime_rl.weight_sync.xor_delta import DeltaEncoder, DeltaUpdate, NvcompLZ4Codec


@dataclass(frozen=True)
class FP8TransferTensors:
    checkpoint: dict[str, Tensor]
    resident: dict[str, Tensor]


class FP8ResidentProducer:
    """Build checkpoint tensors and retain distributed TP-local resident tensors."""

    def __init__(
        self,
        *,
        device: torch.device,
        scale_format: FP8ScaleFormat,
        bucket_bytes: int,
        pipeline_depth: int,
        rank: int,
        world_size: int,
        inference_tp_size: int,
        retain_resident: bool,
    ) -> None:
        if bucket_bytes <= 0:
            raise ValueError(f"FP8 delta bucket size must be positive, got {bucket_bytes}")
        self.device = device
        self.scale_format = scale_format
        self.bucket_bytes = bucket_bytes
        self.pipeline_depth = pipeline_depth
        self.rank = rank
        self.world_size = world_size
        self.inference_tp_size = inference_tp_size
        self.retain_resident = retain_resident
        self.state = FP8ResidentDeltaState()
        self.codec: NvcompLZ4Codec | None = None

    def initialize_tensors(self, tensors: dict[str, Tensor], *, step: int) -> FP8ResidentSnapshot:
        return self.state.initialize(step, tensors)

    def advance_tensors(
        self,
        tensors: dict[str, Tensor],
        *,
        base_step: int,
        step: int,
    ) -> tuple[FP8ResidentSnapshot, DeltaUpdate]:
        snapshot, deltas = self.state.advance(
            base_step=base_step,
            step=step,
            tensors=tensors,
        )
        return snapshot, self.encode_deltas(deltas, base_step=base_step, step=step)

    def encode_deltas(self, deltas: dict[str, Tensor], *, base_step: int, step: int) -> DeltaUpdate:
        if self.codec is None:
            self.codec = NvcompLZ4Codec(self.device)
        encoder = DeltaEncoder(
            base_step=base_step,
            step=step,
            codec=self.codec,
            pipeline_depth=self.pipeline_depth,
        )
        for values in _delta_buckets(deltas, self.bucket_bytes):
            encoder.append_batch(values)
        return encoder.finish()

    @torch.no_grad()
    def build_transfer_tensors(self, model: nn.Module, *, include_checkpoint: bool) -> FP8TransferTensors:
        if not isinstance(model, PreTrainedModelPrimeRL):
            raise TypeError("FP8-kernel XOR requires a PrimeRL custom model implementation")

        grouped: dict[int | None, dict[str, Tensor]] = defaultdict(dict)
        for name, value in model.state_dict().items():
            if not value.is_floating_point():
                continue
            match = LAYER_RE.search(name)
            grouped[int(match.group(1)) if match is not None else None][name] = value

        checkpoint_tensors: dict[str, Tensor] = {}
        resident_tensors: dict[str, Tensor] = {}
        non_layer = _resolve_group(grouped.pop(None, {}))
        if non_layer and self.rank == 0:
            checkpoint_non_layer = model.convert_layer_to_hf(non_layer, -1)
            if include_checkpoint:
                checkpoint_tensors.update(checkpoint_non_layer)
            if self.retain_resident:
                resident_tensors.update(_build_non_layer_resident(checkpoint_non_layer, self.inference_tp_size))
        del non_layer
        for layer_index in sorted(cast(dict[int, dict[str, Tensor]], grouped)):
            resolved = _resolve_group(grouped[layer_index])
            if layer_index % self.world_size != self.rank:
                continue
            checkpoint_layer = (
                model.convert_layer_to_vllm_kernel(
                    resolved.copy(),
                    layer_index,
                    quantize_fp8=True,
                    fp8_scale_format=self.scale_format,
                )
                if include_checkpoint
                else {}
            )
            resident_layer = (
                model.convert_layer_to_vllm_resident(
                    resolved,
                    layer_index,
                    inference_tp_size=self.inference_tp_size,
                    fp8_scale_format=self.scale_format,
                )
                if self.retain_resident
                else {}
            )
            overlap = checkpoint_tensors.keys() & checkpoint_layer.keys()
            if overlap:
                raise ValueError(f"duplicate FP8 checkpoint tensor names: {sorted(overlap)}")
            checkpoint_tensors.update(checkpoint_layer)
            overlap = resident_tensors.keys() & resident_layer.keys()
            if overlap:
                raise ValueError(f"duplicate FP8 resident tensor names: {sorted(overlap)}")
            resident_tensors.update(resident_layer)
        return FP8TransferTensors(checkpoint=checkpoint_tensors, resident=resident_tensors)


def _full_tensor(value: Tensor) -> Tensor:
    if isinstance(value, DTensor):
        return value.full_tensor().detach().contiguous()
    return value.detach().contiguous()


def _resolve_group(values: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: _full_tensor(value) for name, value in values.items()}


def _delta_buckets(deltas: dict[str, Tensor], bucket_bytes: int) -> list[list[tuple[str, Tensor]]]:
    buckets: list[list[tuple[str, Tensor]]] = []
    current: list[tuple[str, Tensor]] = []
    current_dtype: torch.dtype | None = None
    current_group: str | None = None
    current_rank: int | None = None
    current_bytes = 0
    for name, delta in deltas.items():
        tensor_bytes = prod(delta.shape) * delta.element_size()
        group = weight_transfer_group_name(name)
        inference_rank, _ = parse_resident_tensor_name(name)
        if current and (
            delta.dtype != current_dtype
            or group != current_group
            or inference_rank != current_rank
            or current_bytes + tensor_bytes > bucket_bytes
        ):
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append((name, delta))
        current_dtype = delta.dtype
        current_group = group
        current_rank = inference_rank
        current_bytes += tensor_bytes
    if current:
        buckets.append(current)
    return buckets


def _build_non_layer_resident(tensors: dict[str, Tensor], tp_size: int) -> dict[str, Tensor]:
    resident: dict[str, Tensor] = {}
    for name, value in tensors.items():
        shard_vocab = value.ndim == 2 and (name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"))
        if shard_vocab:
            if value.shape[0] % tp_size:
                raise ValueError(
                    f"resident FP8 transfer requires {name!r} rows ({value.shape[0]}) to be divisible by TP={tp_size}"
                )
            rows = value.shape[0] // tp_size
            for rank in range(tp_size):
                resident[resident_tensor_name(rank, name)] = value.narrow(0, rank * rows, rows).contiguous()
        else:
            for rank in range(tp_size):
                resident[resident_tensor_name(rank, name)] = value.detach().contiguous()
    return resident


__all__ = ["FP8ResidentProducer", "FP8TransferTensors"]
