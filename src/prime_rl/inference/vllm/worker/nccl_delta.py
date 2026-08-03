"""BF16 XOR delta receive, reconstruction, and application over NCCL."""

from __future__ import annotations

import pickle
import re
from collections.abc import Callable

import torch
from torch.nn import Module
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.qwen3_bf16_delta import apply_qwen3_bf16_source_deltas_
from prime_rl.weight_sync.bf16_delta import (
    BF16DeltaUpdate,
    CompressedDeltaFrame,
    DeltaTensorMetadata,
    NvcompLZ4Codec,
    ShardedBF16DeltaUpdate,
    WeightUpdateHeader,
    decode_delta_tensors,
    packed_delta_nbytes,
    reconstruct_delta_tensors,
    validate_sharded_delta_update,
)

ReceiveTensor = Callable[[torch.Tensor, PyNcclCommunicator], None]
ReceiveBytes = Callable[[PyNcclCommunicator], bytes]

logger = init_logger("vllm.inference.vllm.worker_nccl_delta")
_QWEN_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")


class NCCLDeltaHandler:
    """Own the nvCOMP decoder and the complete delta receive/apply path."""

    def __init__(self, device: torch.device | str | int) -> None:
        self.device = torch.device(device)
        self.codec = NvcompLZ4Codec(self.device)

    def receive_and_apply(
        self,
        model: Module,
        communicator: PyNcclCommunicator,
        header: WeightUpdateHeader,
        *,
        receive_tensor: ReceiveTensor,
        receive_bytes: ReceiveBytes,
    ) -> None:
        update = receive_compressed_delta(
            communicator,
            base_step=header.base_step,
            step=header.step,
            receive_tensor=receive_tensor,
            receive_bytes=receive_bytes,
        )
        apply_compressed_delta(model, update, codec=self.codec)


def receive_compressed_delta(
    communicator: PyNcclCommunicator,
    *,
    base_step: int,
    step: int,
    receive_tensor: ReceiveTensor,
    receive_bytes: ReceiveBytes,
) -> ShardedBF16DeltaUpdate:
    """Receive and validate every trainer rank's compressed delta payload."""
    shard_metadata = pickle.loads(receive_bytes(communicator))
    if not isinstance(shard_metadata, tuple) or not shard_metadata:
        raise RuntimeError("invalid distributed BF16 delta metadata")
    shards: list[BF16DeltaUpdate] = []
    for rank, item in enumerate(shard_metadata):
        if not isinstance(item, tuple) or len(item) != 3:
            raise RuntimeError(f"invalid BF16 delta metadata for trainer shard {rank}")
        tensors, frames, compressed_nbytes = item
        if not isinstance(tensors, tuple) or not all(isinstance(tensor, DeltaTensorMetadata) for tensor in tensors):
            raise RuntimeError(f"invalid BF16 delta tensor metadata for trainer shard {rank}")
        if not isinstance(frames, tuple) or not all(isinstance(frame, CompressedDeltaFrame) for frame in frames):
            raise RuntimeError(f"invalid BF16 delta frame metadata for trainer shard {rank}")
        if compressed_nbytes != packed_delta_nbytes(frames):
            raise RuntimeError(
                f"trainer shard {rank} metadata describes {packed_delta_nbytes(frames)} compressed bytes; "
                f"sender announced {compressed_nbytes}"
            )
        payload = torch.empty(compressed_nbytes, dtype=torch.uint8, device=communicator.device)
        receive_tensor(payload, communicator)
        shards.append(
            BF16DeltaUpdate(
                base_step=base_step,
                step=step,
                tensors=tensors,
                frames=frames,
                payload=payload,
            )
        )
    update = ShardedBF16DeltaUpdate(base_step=base_step, step=step, shards=tuple(shards))
    try:
        validate_sharded_delta_update(update)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    return update


def apply_compressed_delta(
    model: Module,
    update: ShardedBF16DeltaUpdate,
    *,
    codec: NvcompLZ4Codec,
) -> None:
    """Decode FSDP shards and apply one reconstructed Qwen source layer at a time."""
    logger.info(
        "Received nvCOMP LZ4 BF16 XOR update: %d tensors in %d frames from %d trainer shards, "
        "%.2f MiB compressed from %.2f MiB (%.1fx)",
        update.tensor_count,
        update.frame_count,
        len(update.shards),
        update.compressed_nbytes / (1024 * 1024),
        update.uncompressed_nbytes / (1024 * 1024),
        update.uncompressed_nbytes / update.compressed_nbytes,
    )
    frame_payloads = [list(shard.frame_payloads()) for shard in update.shards]
    for frame_index in range(update.frame_count):
        decoded_shards: list[list[tuple[str, torch.Tensor]]] = []
        metadata_shards: list[tuple[DeltaTensorMetadata, ...]] = []
        for shard_index, shard in enumerate(update.shards):
            frame = shard.frames[frame_index]
            decoded_shards.append(
                decode_delta_tensors(
                    codec,
                    shard.tensors,
                    [frame],
                    [frame_payloads[shard_index][frame_index]],
                )
            )
            metadata_shards.append(
                shard.tensors[frame.first_tensor_index : frame.first_tensor_index + frame.tensor_count]
            )

        reference_metadata = metadata_shards[0]
        group_start = 0
        while group_start < len(reference_metadata):
            layer_index = _qwen_layer_index(reference_metadata[group_start].name)
            group_end = group_start + 1
            while (
                group_end < len(reference_metadata)
                and _qwen_layer_index(reference_metadata[group_end].name) == layer_index
            ):
                group_end += 1
            decoded_group = reconstruct_delta_tensors(
                [values[group_start:group_end] for values in decoded_shards],
                [metadata[group_start:group_end] for metadata in metadata_shards],
            )
            apply_qwen3_bf16_source_deltas_(
                model,
                dict(decoded_group),
                layer_index=layer_index,
            )
            del decoded_group
            group_start = group_end
        del decoded_shards, metadata_shards


def _qwen_layer_index(name: str) -> int:
    match = _QWEN_LAYER_PATTERN.match(name)
    return int(match.group(1)) if match else -1


__all__ = ["NCCLDeltaHandler", "apply_compressed_delta", "receive_compressed_delta"]
