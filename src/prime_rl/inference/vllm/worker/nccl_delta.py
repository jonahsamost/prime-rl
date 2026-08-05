"""XOR delta receive, reconstruction, and application over NCCL."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn import Module
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.dense_xor import DenseDeltaRouter, source_layer_module_path
from prime_rl.weight_sync.xor_delta import (
    CompressedDeltaFrame,
    DeltaTensorMetadata,
    DeltaUpdate,
    NvcompLZ4Codec,
    ShardedDeltaUpdate,
    WeightUpdateHeader,
    packed_delta_nbytes,
    reconstruct_delta_tensors,
    unpack_delta_frame,
    validate_sharded_delta_update,
)

ReceiveTensor = Callable[[torch.Tensor, PyNcclCommunicator], None]
ReceiveBytes = Callable[[PyNcclCommunicator], bytes]

logger = init_logger("vllm.inference.vllm.worker_nccl_delta")


@dataclass(frozen=True)
class _SourceGroup:
    start: int
    end: int
    layer_path: str | None


class NCCLDeltaHandler:
    """Own the nvCOMP decoder and the complete delta receive/apply path."""

    def __init__(self, device: torch.device | str | int) -> None:
        self.device = torch.device(device)
        self.codec = NvcompLZ4Codec(self.device)
        self.router: DenseDeltaRouter | None = None
        self.group_cache: dict[tuple[str, ...], tuple[_SourceGroup, ...]] = {}

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
        if self.router is None:
            self.router = DenseDeltaRouter(model, model_dtype)
        elif self.router.model is not model or self.router.model_dtype != model_dtype:
            raise RuntimeError("NCCL XOR delta handler cannot switch resident models")
        apply_compressed_delta(
            model,
            update,
            model_dtype=model_dtype,
            codec=self.codec,
            router=self.router,
            group_cache=self.group_cache,
        )


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
    router: DenseDeltaRouter | None = None,
    group_cache: dict[tuple[str, ...], tuple[_SourceGroup, ...]] | None = None,
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
    if router is None:
        router = DenseDeltaRouter(model, model_dtype)
    elif router.model is not model or router.model_dtype != model_dtype:
        raise ValueError("XOR delta router does not match the resident model")
    if group_cache is None:
        group_cache = {}
    frame_payloads = [list(shard.frame_payloads()) for shard in update.shards]
    for frame_index in range(update.frame_count):
        frames = [shard.frames[frame_index] for shard in update.shards]
        decoded_frames = codec.decode(
            [payloads[frame_index] for payloads in frame_payloads],
            [frame.uncompressed_nbytes for frame in frames],
        )
        decoded_shards: list[list[tuple[str, torch.Tensor]]] = []
        metadata_shards: list[tuple[DeltaTensorMetadata, ...]] = []
        for shard, frame, decoded in zip(update.shards, frames, decoded_frames, strict=True):
            decoded_shards.append(unpack_delta_frame(decoded, shard.tensors, frame))
            metadata_shards.append(
                shard.tensors[frame.first_tensor_index : frame.first_tensor_index + frame.tensor_count]
            )

        reference_metadata = metadata_shards[0]
        source_names = tuple(item.name for item in reference_metadata)
        groups = group_cache.get(source_names)
        if groups is None:
            groups = _source_groups(reference_metadata)
            group_cache[source_names] = groups
        for group in groups:
            decoded_group = reconstruct_delta_tensors(
                [values[group.start : group.end] for values in decoded_shards],
                [metadata[group.start : group.end] for metadata in metadata_shards],
            )
            router.apply(dict(decoded_group), layer_path=group.layer_path)
            del decoded_group
        del decoded_shards, metadata_shards


def _source_groups(metadata: tuple[DeltaTensorMetadata, ...]) -> tuple[_SourceGroup, ...]:
    groups: list[_SourceGroup] = []
    start = 0
    while start < len(metadata):
        layer_path = source_layer_module_path(metadata[start].name)
        end = start + 1
        while end < len(metadata) and source_layer_module_path(metadata[end].name) == layer_path:
            end += 1
        groups.append(_SourceGroup(start, end, layer_path))
        start = end
    return tuple(groups)


__all__ = ["NCCLDeltaHandler", "apply_compressed_delta", "receive_compressed_delta"]
