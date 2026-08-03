"""BF16 XOR delta protocol helpers for the trainer NCCL broadcaster."""

from __future__ import annotations

import pickle
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch import Tensor
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from prime_rl.weight_sync.bf16_delta import (
    BF16DeltaUpdate,
    CompressedDeltaFrame,
    DeltaTensorMetadata,
    ShardedBF16DeltaUpdate,
    packed_delta_nbytes,
    validate_sharded_delta_update,
)

BroadcastTensor = Callable[[Tensor, PyNcclCommunicator], None]
BroadcastBytes = Callable[[bytes, PyNcclCommunicator], None]


def broadcast_compressed_delta(
    update: ShardedBF16DeltaUpdate,
    communicator: PyNcclCommunicator,
    *,
    broadcast_tensor: BroadcastTensor,
    broadcast_bytes: BroadcastBytes,
) -> None:
    """Broadcast distributed delta metadata followed by each rank's CUDA payload."""
    metadata = pickle.dumps(tuple((shard.tensors, shard.frames, shard.compressed_nbytes) for shard in update.shards))
    broadcast_bytes(metadata, communicator)
    for rank, shard in enumerate(update.shards):
        if shard.payload.device != communicator.device:
            raise ValueError(
                f"BF16 delta payload for trainer rank {rank} is on {shard.payload.device}; "
                f"NCCL communicator uses {communicator.device}"
            )
        broadcast_tensor(shard.payload, communicator)


def _serialize_local_delta_metadata(update: BF16DeltaUpdate) -> bytes:
    return pickle.dumps((update.base_step, update.step, update.tensors, update.frames))


def _deserialize_local_delta_metadata(payload: Tensor, compressed_payload: Tensor) -> BF16DeltaUpdate:
    decoded = pickle.loads(payload.cpu().numpy().tobytes())
    if not isinstance(decoded, tuple) or len(decoded) != 4:
        raise ValueError("invalid trainer BF16 delta metadata envelope")
    base_step, step, tensors, frames = decoded
    if not isinstance(base_step, int) or not isinstance(step, int):
        raise ValueError("trainer BF16 delta metadata has invalid policy versions")
    if not isinstance(tensors, tuple) or not all(isinstance(item, DeltaTensorMetadata) for item in tensors):
        raise ValueError("trainer BF16 delta metadata has invalid tensor manifest")
    if not isinstance(frames, tuple) or not all(isinstance(item, CompressedDeltaFrame) for item in frames):
        raise ValueError("trainer BF16 delta metadata has invalid frame manifest")
    if compressed_payload.numel() != packed_delta_nbytes(frames):
        raise ValueError(
            f"trainer BF16 delta payload has {compressed_payload.numel()} bytes; "
            f"metadata requires {packed_delta_nbytes(frames)}"
        )
    return BF16DeltaUpdate(
        base_step=base_step,
        step=step,
        tensors=tensors,
        frames=frames,
        payload=compressed_payload,
    )


def _gather_variable_cuda_tensor_to_rank_zero(local_value: Tensor, sizes: list[int]) -> Tensor | None:
    """Collect exact-sized CUDA byte segments on rank 0 through one NCCL collective."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if len(sizes) != world_size or local_value.numel() != sizes[rank]:
        raise ValueError(
            f"invalid variable gather sizes {sizes} for rank {rank} tensor with {local_value.numel()} elements"
        )
    input_splits = [0] * world_size
    input_splits[0] = local_value.numel()
    output_splits = sizes if rank == 0 else [0] * world_size
    output = torch.empty(sum(output_splits), dtype=local_value.dtype, device=local_value.device)
    dist.all_to_all_single(
        output,
        local_value,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
    )
    return output if rank == 0 else None


def _split_gathered_tensor(value: Tensor, sizes: list[int]) -> list[Tensor]:
    pieces: list[Tensor] = []
    offset = 0
    for size in sizes:
        pieces.append(value.narrow(0, offset, size))
        offset += size
    if offset != value.numel():
        raise ValueError(f"variable gather describes {offset} elements; received {value.numel()}")
    return pieces


def gather_compressed_delta_updates(
    local_update: BF16DeltaUpdate | None,
) -> tuple[ShardedBF16DeltaUpdate | None, bool]:
    """Gather rank-local compressed FSDP deltas onto trainer rank 0."""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if world_size == 1:
        if local_update is None:
            return None, False
        sharded = ShardedBF16DeltaUpdate(
            base_step=local_update.base_step,
            step=local_update.step,
            shards=(local_update,),
        )
        validate_sharded_delta_update(sharded)
        return sharded, True

    device = (
        local_update.payload.device if local_update is not None else torch.device("cuda", torch.cuda.current_device())
    )
    available = torch.tensor([local_update is not None], dtype=torch.uint8, device=device)
    dist.all_reduce(available, op=dist.ReduceOp.MIN)
    if not bool(available.item()):
        return None, False
    assert local_update is not None

    metadata = torch.frombuffer(bytearray(_serialize_local_delta_metadata(local_update)), dtype=torch.uint8).to(device)
    local_sizes = torch.tensor(
        [metadata.numel(), local_update.payload.numel()],
        dtype=torch.long,
        device=device,
    )
    gathered_sizes = [torch.empty_like(local_sizes) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_sizes)
    sizes = [tuple(int(value) for value in item.tolist()) for item in gathered_sizes]

    metadata_sizes = [metadata_size for metadata_size, _payload_size in sizes]
    payload_sizes = [payload_size for _metadata_size, payload_size in sizes]
    gathered_metadata = _gather_variable_cuda_tensor_to_rank_zero(metadata, metadata_sizes)
    gathered_payload = _gather_variable_cuda_tensor_to_rank_zero(local_update.payload, payload_sizes)
    metadata_by_rank: list[Tensor] | None = None
    payload_by_rank: list[Tensor] | None = None
    if rank == 0:
        assert gathered_metadata is not None and gathered_payload is not None
        metadata_by_rank = _split_gathered_tensor(gathered_metadata, metadata_sizes)
        # Variable rank segments are not guaranteed to begin on nvCOMP's
        # required alignment. Clone each compressed segment into its own CUDA
        # allocation without ever reconstructing an uncompressed delta here.
        payload_by_rank = [piece.clone() for piece in _split_gathered_tensor(gathered_payload, payload_sizes)]
    dist.barrier()

    if rank != 0:
        return None, True
    assert metadata_by_rank is not None and payload_by_rank is not None
    shards = [
        _deserialize_local_delta_metadata(metadata_by_rank[index], payload_by_rank[index])
        for index in range(world_size)
    ]
    sharded = ShardedBF16DeltaUpdate(
        base_step=local_update.base_step,
        step=local_update.step,
        shards=tuple(shards),
    )
    validate_sharded_delta_update(sharded)
    return sharded, True


__all__ = ["broadcast_compressed_delta", "gather_compressed_delta_updates"]
