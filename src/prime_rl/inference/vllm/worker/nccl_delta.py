"""XOR delta receive, reconstruction, and application over NCCL."""

from __future__ import annotations

import pickle
from collections.abc import Callable

import torch
from torch.nn import Module
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.dense_xor import apply_dense_source_deltas_, source_layer_module_path
from prime_rl.weight_sync.xor_delta import (
    CompressedDeltaFrame,
    DeltaTensorMetadata,
    DeltaUpdate,
    NvcompLZ4Codec,
    ShardedDeltaUpdate,
    WeightUpdateHeader,
    decode_delta_tensors,
    packed_delta_nbytes,
    reconstruct_delta_tensors,
    validate_sharded_delta_update,
)

ReceiveTensor = Callable[[torch.Tensor, PyNcclCommunicator], None]
ReceiveBytes = Callable[[PyNcclCommunicator], bytes]

logger = init_logger("vllm.inference.vllm.worker_nccl_delta")


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
        model_dtype: torch.dtype,
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
        apply_compressed_delta(model, update, model_dtype=model_dtype, codec=self.codec)


def receive_compressed_delta(
    communicator: PyNcclCommunicator,
    *,
    base_step: int,
    step: int,
    receive_tensor: ReceiveTensor,
    receive_bytes: ReceiveBytes,
) -> ShardedDeltaUpdate:
    """Receive and validate every trainer rank's compressed delta payload."""
    shard_metadata = pickle.loads(receive_bytes(communicator))
    if not isinstance(shard_metadata, tuple) or not shard_metadata:
        raise RuntimeError("invalid distributed XOR delta metadata")
    shards: list[DeltaUpdate] = []
    for rank, item in enumerate(shard_metadata):
        if not isinstance(item, tuple) or len(item) != 3:
            raise RuntimeError(f"invalid XOR delta metadata for trainer shard {rank}")
        tensors, frames, compressed_nbytes = item
        if not isinstance(tensors, tuple) or not all(isinstance(tensor, DeltaTensorMetadata) for tensor in tensors):
            raise RuntimeError(f"invalid XOR delta tensor metadata for trainer shard {rank}")
        if not isinstance(frames, tuple) or not all(isinstance(frame, CompressedDeltaFrame) for frame in frames):
            raise RuntimeError(f"invalid XOR delta frame metadata for trainer shard {rank}")
        if compressed_nbytes != packed_delta_nbytes(frames):
            raise RuntimeError(
                f"trainer shard {rank} metadata describes {packed_delta_nbytes(frames)} compressed bytes; "
                f"sender announced {compressed_nbytes}"
            )
        payload = torch.empty(compressed_nbytes, dtype=torch.uint8, device=communicator.device)
        receive_tensor(payload, communicator)
        shards.append(
            DeltaUpdate(
                base_step=base_step,
                step=step,
                tensors=tensors,
                frames=frames,
                payload=payload,
            )
        )
    update = ShardedDeltaUpdate(base_step=base_step, step=step, shards=tuple(shards))
    try:
        validate_sharded_delta_update(update)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    return update


def apply_compressed_delta(
    model: Module,
    update: ShardedDeltaUpdate,
    *,
    model_dtype: torch.dtype,
    codec: NvcompLZ4Codec,
) -> None:
    """Decode FSDP shards and apply one reconstructed source layer at a time."""
    logger.info(
        "Received nvCOMP LZ4 XOR update: %d tensors in %d frames from %d trainer shards, "
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
            layer_path = source_layer_module_path(reference_metadata[group_start].name)
            group_end = group_start + 1
            while (
                group_end < len(reference_metadata)
                and source_layer_module_path(reference_metadata[group_end].name) == layer_path
            ):
                group_end += 1
            decoded_group = reconstruct_delta_tensors(
                [values[group_start:group_end] for values in decoded_shards],
                [metadata[group_start:group_end] for metadata in metadata_shards],
            )
            apply_dense_source_deltas_(model, dict(decoded_group), model_dtype=model_dtype)
            del decoded_group
            group_start = group_end
        del decoded_shards, metadata_shards


__all__ = ["NCCLDeltaHandler", "apply_compressed_delta", "receive_compressed_delta"]
