from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from prime_rl.trainer.rl.broadcast.nccl_delta import gather_compressed_delta_updates
from prime_rl.weight_sync.xor_delta import (
    NVCOMP_FRAME_ALIGNMENT,
    DeltaEncoder,
    NvcompLZ4Codec,
    reconstruct_delta_tensors,
    unpack_delta_frame,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def cleanup_zombies():
    """Override the global autouse fixture: this module runs inside the torchrun process it would kill."""
    return None


def test_four_rank_compressed_delta_gather_is_byte_exact():
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("run with torchrun --nproc-per-node=4")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)

    local = torch.full((2, 8), rank + 1, dtype=torch.int16, device=device).view(torch.bfloat16)
    codec = NvcompLZ4Codec(device)
    encoder = DeltaEncoder(base_step=3, step=4, codec=codec)
    encoder.append_sharded_bucket(
        [("weight", local)],
        local.reshape(-1),
        global_shapes=[(8, 8)],
        shard_dim=0,
        shard_index=rank,
        shard_count=4,
    )
    local_update = encoder.finish()
    gathered, ready = gather_compressed_delta_updates(local_update)

    assert ready
    if rank == 0:
        assert gathered is not None
        assert all(shard.payload.data_ptr() % NVCOMP_FRAME_ALIGNMENT == 0 for shard in gathered.shards)
        assert len({shard.payload.untyped_storage().data_ptr() for shard in gathered.shards}) == 1
        frames = [shard.frames[0] for shard in gathered.shards]
        codec = NvcompLZ4Codec(device)
        decoded_frames = codec.decode(
            [next(shard.frame_payloads()) for shard in gathered.shards],
            [frame.uncompressed_nbytes for frame in frames],
        )
        codec.synchronize()
        decoded_shards = [
            unpack_delta_frame(decoded, shard.tensors, frame)
            for decoded, shard, frame in zip(decoded_frames, gathered.shards, frames, strict=True)
        ]
        metadata_shards = [shard.tensors for shard in gathered.shards]
        reconstructed = dict(reconstruct_delta_tensors(decoded_shards, metadata_shards))["weight"]
        expected = torch.cat([torch.full((2, 8), index + 1, dtype=torch.int16, device=device) for index in range(4)])
        assert torch.equal(reconstructed.view(torch.int16), expected)
    else:
        assert gathered is None

    dist.barrier()


def test_one_missing_rank_forces_collective_full_weight_fallback():
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("run with torchrun --nproc-per-node=4")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)

    update = None
    if rank != 3:
        local = torch.zeros((2, 8), dtype=torch.bfloat16, device=device)
        encoder = DeltaEncoder(base_step=4, step=5, codec=NvcompLZ4Codec(device))
        encoder.append_sharded_bucket(
            [("weight", local)],
            local.reshape(-1),
            global_shapes=[(8, 8)],
            shard_dim=0,
            shard_index=rank,
            shard_count=4,
        )
        update = encoder.finish()

    gathered, ready = gather_compressed_delta_updates(update)
    assert not ready
    assert gathered is None
