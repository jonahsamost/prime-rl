"""Trainer-initiated, groupwise NIXL weight broadcast."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from modelexpress import p2p_pb2

from prime_rl.trainer.rl.broadcast.nixl.cuda_malloc_memory import size_cuda_buffers
from prime_rl.trainer.rl.broadcast.nixl.nixl import NIXLWeightBroadcast
from prime_rl.trainer.rl.broadcast.nixl.receiver_table import ReceiverTable


@dataclass(frozen=True)
class PreparedWrite:
    receiver_rank: int
    local: Any
    remote: Any
    indices: list[int]


@dataclass(frozen=True)
class PendingWrite:
    receiver_rank: int
    handle: Any


class NIXLPushWeightBroadcast(NIXLWeightBroadcast):
    """Push canonical FSDP shards into inference-owned staging buffers."""

    def choose_staging_buffer_count(self, largest_group_bytes: int) -> int:
        requested = self.config.push_buffer_count
        target_count = min(
            2 if requested == "auto" else requested,
            len(self.transfer_group_names),
        )
        local_count = target_count
        if self.is_serving_rank and largest_group_bytes and target_count:
            device = self.staged_shards[0].source_tensor.device
            allocated_bytes = torch.cuda.memory_allocated(device)
            peak_growth_bytes = max(0, torch.cuda.max_memory_allocated(device) - allocated_bytes)
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if peak_growth_bytes or free_bytes < target_count * largest_group_bytes:
                torch.cuda.empty_cache()
            local_count = size_cuda_buffers(
                largest_group_bytes,
                target_count,
                device,
                extra_headroom_bytes=peak_growth_bytes,
            )

        # Every serving rank must use the same ring size because the published
        # source addresses are indexed by group modulo this count.
        count = torch.tensor(
            local_count,
            dtype=torch.int64,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        dist.all_reduce(count, op=dist.ReduceOp.MIN)
        available_count = int(count.item())
        if requested != "auto" and available_count < target_count:
            raise RuntimeError(
                f"NIXL push requested {target_count} buffers, but the trainer has memory for "
                f"only {available_count}"
            )
        return available_count

    def _initialize_protocol(self) -> None:
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
        self.prepared_writes = (
            self.prepare_writes(receiver_tables)
            if self.is_serving_rank
            else [[] for _ in self.transfer_group_names]
        )
        if self.is_serving_rank:
            self.staging_stream = torch.cuda.Stream(device=torch.cuda.current_device())

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

    def stage_group(self, group_index: int) -> None:
        if not self.is_serving_rank:
            return
        # Preserve the optimizer/default-stream happens-before relationship
        # when moving staging copies onto their own stream.
        self.staging_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.staging_stream):
            for shard in self.staged_shards_by_group.get(group_index, ()):
                shard.copy_to_staging()
            staged = torch.cuda.Event()
            staged.record(self.staging_stream)
        # NIXL is not ordered against PyTorch streams. Only wait for this
        # group's copies; a device-wide synchronize would also stall unrelated
        # trainer CUDA work.
        staged.synchronize()

    def post_group_writes(self, group_index: int) -> list[PendingWrite]:
        return [
            PendingWrite(
                receiver_rank=write.receiver_rank,
                handle=self.nixl_agent.post_write(write.local, write.indices, write.remote),
            )
            for write in self.prepared_writes[group_index]
        ]

    def finish_group_writes(self, group_index: int, pending: list[PendingWrite]) -> None:
        for write in pending:
            self.nixl_agent.wait(
                write.handle,
                context=(
                    f"weight write for {self.transfer_group_names[group_index]} "
                    f"to inference rank {write.receiver_rank}"
                ),
                timeout=self.config.timeout,
            )

    def publish_group(self, group_index: int) -> None:
        if self.world.is_master:
            self.buffer_sessions[group_index % self.staging_buffer_count].set_status(
                p2p_pb2.SOURCE_STATUS_READY
            )

    def _transfer_group(self, group_index: int, step: int, reset_buffer: int | None = None) -> None:
        started = time.perf_counter()
        self.stage_group(group_index)
        pending = self.post_group_writes(group_index)
        self.finish_group_writes(group_index, pending)
        if reset_buffer is not None:
            self.reset_staging_buffer(reset_buffer)

        # The group may be published only after every contributing trainer
        # rank has completed its writes and any next-use slot is reusable.
        dist.barrier()
        self.publish_group(group_index)
        if self.world.is_master:
            self.logger.debug(
                f"NIXL push policy v{step} group {self.transfer_group_names[group_index]} transferred in "
                f"{time.perf_counter() - started:.2f}s"
            )

    def _broadcast_single_buffer(self, step: int) -> None:
        for group_index in range(len(self.transfer_group_names)):
            if group_index:
                self.finish_staging_buffer_transfer(0)
            self._transfer_group(group_index, step)
        if self.transfer_group_names:
            self.finish_staging_buffer_transfer(0)

    def _broadcast_buffer_ring(self, step: int) -> None:
        for group_index in range(len(self.transfer_group_names)):
            next_group = group_index + 1
            reset_buffer = (
                next_group % self.staging_buffer_count
                if next_group >= self.staging_buffer_count
                else None
            )
            self._transfer_group(group_index, step, reset_buffer)

        first_pending_group = max(
            0,
            len(self.transfer_group_names) - self.staging_buffer_count + 1,
        )
        for group_index in range(first_pending_group, len(self.transfer_group_names)):
            self.reset_staging_buffer(group_index % self.staging_buffer_count)

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        self.initialize_transfer(model)
        start = time.perf_counter()
        self._begin_update()

        if self.staging_buffer_count == 1:
            self._broadcast_single_buffer(step)
        else:
            self._broadcast_buffer_ring(step)

        self._finish_update()
        self.logger.info(f"NIXL push policy v{step} synchronized in {time.perf_counter() - start:.2f}s")
