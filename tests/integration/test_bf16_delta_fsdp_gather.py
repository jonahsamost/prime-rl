from __future__ import annotations

import os
import time

import pytest
import torch
import torch.distributed as dist

from prime_rl.trainer.rl.broadcast.nccl import gather_compressed_delta_updates
from prime_rl.weight_sync.bf16_delta import (
    BF16DeltaEncoder,
    NVCOMP_FRAME_ALIGNMENT,
    NvcompLZ4Codec,
    decode_delta_tensors,
    reconstruct_delta_tensors,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def cleanup_zombies():
    """Override the global autouse fixture: this module runs inside the torchrun process it would kill."""
    return None


def _raw_progress(message: str) -> None:
    rank = os.environ.get("RANK", "unknown")
    local_rank = os.environ.get("LOCAL_RANK", "unknown")
    line = (
        f"[{time.time():.6f}] [bf16-delta-gather pid={os.getpid()} rank={rank} "
        f"local_rank={local_rank}] {message}\n"
    )
    os.write(2, line.encode())


def _progress(
    rank: int | str,
    message: str,
    *,
    local_rank: int | str = "unknown",
    device: str | None = None,
) -> None:
    if device is None:
        device = f"cuda:{local_rank}" if local_rank != "unknown" else "cuda:unknown"
    line = (
        f"[{time.time():.6f}] [bf16-delta-gather pid={os.getpid()} rank={rank} "
        f"local_rank={local_rank} device={device}] {message}\n"
    )
    os.write(2, line.encode())


def test_four_rank_compressed_delta_gather_is_byte_exact(monkeypatch: pytest.MonkeyPatch):
    _raw_progress("first test-body statement")
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("run with torchrun --nproc-per-node=4")
    local_rank = int(os.environ["LOCAL_RANK"])
    env_rank = os.environ.get("RANK", "unknown")
    _progress(env_rank, "test entry", local_rank=local_rank, device="unset")
    _progress(env_rank, "before torch.cuda.set_device", local_rank=local_rank, device="unset")
    torch.cuda.set_device(local_rank)
    cuda_device = f"cuda:{local_rank}"
    _progress(env_rank, "after torch.cuda.set_device", local_rank=local_rank, device=cuda_device)
    if not dist.is_initialized():
        _progress(
            env_rank,
            "before dist.init_process_group(nccl)",
            local_rank=local_rank,
            device=cuda_device,
        )
        dist.init_process_group("nccl")
        _progress(
            env_rank,
            "after dist.init_process_group(nccl)",
            local_rank=local_rank,
            device=cuda_device,
        )
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    monkeypatch.setenv("PRIME_RL_DELTA_GATHER_DEBUG", "1")

    local = torch.full((2, 8), rank + 1, dtype=torch.int16, device=device).view(torch.bfloat16)
    _progress(rank, "before NvcompLZ4Codec construction", local_rank=local_rank)
    codec = NvcompLZ4Codec(device)
    _progress(rank, "after NvcompLZ4Codec construction", local_rank=local_rank)
    _progress(rank, "before BF16DeltaEncoder construction", local_rank=local_rank)
    encoder = BF16DeltaEncoder(base_step=3, step=4, codec=codec)
    _progress(rank, "after BF16DeltaEncoder construction", local_rank=local_rank)
    _progress(rank, "before append_sharded_bucket", local_rank=local_rank)
    encoder.append_sharded_bucket(
        [("weight", local)],
        local.reshape(-1),
        global_shapes=[(8, 8)],
        shard_dim=0,
        shard_index=rank,
        shard_count=4,
    )
    _progress(rank, "after append_sharded_bucket", local_rank=local_rank)
    _progress(rank, "before encoder.finish", local_rank=local_rank)
    local_update = encoder.finish()
    _progress(
        rank,
        f"after encoder.finish compressed_bytes={local_update.compressed_nbytes}",
        local_rank=local_rank,
    )
    _progress(rank, "before gather_compressed_delta_updates", local_rank=local_rank)
    gathered, ready = gather_compressed_delta_updates(local_update)
    _progress(rank, "after gather_compressed_delta_updates", local_rank=local_rank)

    assert ready
    if rank == 0:
        assert gathered is not None
        assert all(shard.payload.data_ptr() % NVCOMP_FRAME_ALIGNMENT == 0 for shard in gathered.shards)
        _progress(rank, "before rank-zero decode", local_rank=local_rank)
        decoded_shards = []
        metadata_shards = []
        for shard_index, shard in enumerate(gathered.shards):
            _progress(rank, f"before decoder codec construction shard={shard_index}", local_rank=local_rank)
            codec = NvcompLZ4Codec(device)
            _progress(rank, f"after decoder codec construction shard={shard_index}", local_rank=local_rank)
            _progress(rank, f"before decode shard={shard_index}", local_rank=local_rank)
            decoded, _events = decode_delta_tensors(
                codec,
                shard.tensors,
                shard.frames,
                list(shard.frame_payloads()),
            )
            codec.synchronize()
            _progress(rank, f"after decode shard={shard_index}", local_rank=local_rank)
            decoded_shards.append(decoded)
            metadata_shards.append(shard.tensors)
        _progress(rank, "before rank-zero reconstruction", local_rank=local_rank)
        reconstructed = dict(reconstruct_delta_tensors(decoded_shards, metadata_shards))["weight"]
        _progress(rank, "after rank-zero reconstruction", local_rank=local_rank)
        expected = torch.cat(
            [torch.full((2, 8), index + 1, dtype=torch.int16, device=device) for index in range(4)]
        )
        assert torch.equal(reconstructed.view(torch.int16), expected)
        _progress(rank, "rank-zero reconstruction is byte exact", local_rank=local_rank)
    else:
        assert gathered is None

    _progress(rank, "before test-final barrier", local_rank=local_rank)
    dist.barrier()
    _progress(rank, "after test-final barrier; first test complete", local_rank=local_rank)


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
        encoder = BF16DeltaEncoder(base_step=4, step=5, codec=NvcompLZ4Codec(device))
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
    _progress(rank, "collective fallback decision completed")

    assert not ready
    assert gathered is None
