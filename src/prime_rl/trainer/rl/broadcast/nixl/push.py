"""Trainer-initiated, groupwise NIXL weight broadcast."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from modelexpress import p2p_pb2

from prime_rl.configs.trainer import NIXLWeightBroadcastConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.nixl.nixl import NIXLWeightBroadcast
from prime_rl.trainer.rl.broadcast.nixl.receiver_table import ReceiverTable


@dataclass(frozen=True)
class PreparedWrite:
    receiver_rank: int
    local: Any
    remote: Any
    indices: list[int]


class NIXLPushWeightBroadcast(NIXLWeightBroadcast):
    """Push canonical FSDP shards into inference-owned staging buffers."""

    def __init__(self, output_dir: Path, config: NIXLWeightBroadcastConfig, parallel_dims: ParallelDims) -> None:
        super().__init__(output_dir, config, parallel_dims)
        self.push_initialized = False
        self.prepared_writes: list[list[PreparedWrite]] = []

    def choose_staging_buffer_count(self, largest_group_bytes: int) -> int:
        del largest_group_bytes
        count = torch.tensor(1, dtype=torch.int64, device=torch.device("cuda", torch.cuda.current_device()))
        dist.all_reduce(count, op=dist.ReduceOp.MIN)
        return int(count.item())

    def initialize_transfer(self, model: nn.Module) -> None:
        super().initialize_transfer(model)
        if self.push_initialized:
            return

        encoded_tables: list[bytes] | None = None
        if self.world.is_master:
            refs = self.model_express.wait_for(
                "inference",
                count=self.config.inference_world_size,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.config.timeout,
            )
            encoded_tables = [self.model_express.fetch(ref).nixl_metadata for ref in refs]

        objects = [encoded_tables]
        dist.broadcast_object_list(objects, src=0)
        encoded_tables = objects[0]
        if encoded_tables is None:
            raise RuntimeError("trainer rank 0 did not publish NIXL receiver tables")
        receiver_tables = [ReceiverTable.decode(encoded) for encoded in encoded_tables]
        self.prepared_writes = self.prepare_writes(receiver_tables) if self.is_serving_rank else []
        self.push_initialized = True

    def prepare_writes(self, receiver_tables: list[ReceiverTable]) -> list[list[PreparedWrite]]:
        groups: list[list[PreparedWrite]] = [[] for _ in self.transfer_group_names]
        for receiver in receiver_tables:
            if [group.name for group in receiver.groups] != self.transfer_group_names:
                raise RuntimeError(f"inference rank {receiver.agent.rank} NIXL groups do not match trainer groups")
            routes_by_group = [
                [route for route in group.routes if route.trainer_agent_name == self.nixl_agent.name]
                for group in receiver.groups
            ]
            if not any(routes_by_group):
                continue

            peer_name = self.nixl_agent.add_remote_agent(receiver.agent.metadata)
            self.nixl_agent.make_connection(peer_name)
            for group_index, routes in enumerate(routes_by_group):
                if not routes:
                    continue
                local = self.nixl_agent.prepare_xfer_dlist([route.source for route in routes])
                remote = self.nixl_agent.prepare_xfer_dlist(
                    [route.destination for route in routes],
                    agent_name=peer_name,
                )
                groups[group_index].append(
                    PreparedWrite(
                        receiver_rank=receiver.agent.rank,
                        local=local,
                        remote=remote,
                        indices=list(range(len(routes))),
                    )
                )
        return groups

    def write_group(self, group_index: int) -> None:
        pending: list[tuple[PreparedWrite, Any]] = []
        if self.is_serving_rank:
            for write in self.prepared_writes[group_index]:
                pending.append((write, self.nixl_agent.post_write(write.local, write.indices, write.remote)))
            for write, handle in pending:
                self.nixl_agent.wait(
                    handle,
                    context=(
                        f"weight write for {self.transfer_group_names[group_index]} "
                        f"to inference rank {write.receiver_rank}"
                    ),
                    timeout=self.config.timeout,
                )

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        self.initialize_transfer(model)
        start = time.perf_counter()

        if self.world.is_master:
            self.model_express.set_status(p2p_pb2.SOURCE_STATUS_READY)
            self.model_express.wait_for(
                "orchestrator",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_READY,
                timeout=self.config.timeout,
            )
            self.model_express.wait_for(
                "inference",
                count=self.config.inference_world_size,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.config.timeout,
            )

        for group_index, group_name in enumerate(self.transfer_group_names):
            group_start = time.perf_counter()
            if group_index:
                self.finish_staging_buffer_transfer(0)

            if self.is_serving_rank:
                for shard in self.staged_shards_by_group.get(group_index, ()):
                    shard.copy_to_staging()
                torch.cuda.synchronize()
            dist.barrier()
            self.write_group(group_index)
            dist.barrier()

            if self.world.is_master:
                self.buffer_sessions[0].set_status(p2p_pb2.SOURCE_STATUS_READY)
                self.logger.debug(
                    f"NIXL push policy v{step} group {group_name} transferred in "
                    f"{time.perf_counter() - group_start:.2f}s"
                )

        self.finish_staging_buffer_transfer(0)
        if self.world.is_master:
            self.model_express.wait_for(
                "inference",
                count=self.config.inference_world_size,
                status=p2p_pb2.SOURCE_STATUS_READY,
                timeout=self.config.timeout,
            )
            self.model_express.wait_for(
                "orchestrator",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.config.timeout,
            )
            self.model_express.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
        dist.barrier()
        self.logger.info(f"NIXL push policy v{step} synchronized in {time.perf_counter() - start:.2f}s")
