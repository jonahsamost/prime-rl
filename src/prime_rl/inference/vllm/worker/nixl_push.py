"""vLLM worker for trainer-initiated writes into bounded reload buffers."""

from __future__ import annotations

from threading import Event

import torch

from prime_rl.inference.vllm.worker.nixl import NIXLWeightUpdateWorker, WeightTransferPlan
from prime_rl.trainer.rl.broadcast.nixl.agent import MemDesc, group_notification
from prime_rl.trainer.rl.broadcast.nixl.cuda_malloc_memory import size_cuda_buffers
from prime_rl.trainer.rl.broadcast.nixl.receiver_table import (
    ReceiverAgent,
    ReceiverGroup,
    ReceiverRoute,
    ReceiverTable,
)
from prime_rl.trainer.rl.broadcast.nixl.trainer_tensor_table import TrainerTensorTable


class NIXLPushWeightUpdateWorker(NIXLWeightUpdateWorker):
    """Receive canonical tensors by NIXL WRITE and replay vLLM's load graph."""

    ack_before_replay = False

    def init_broadcaster(
        self,
        host: str,
        port: int,
        rank_offset: int,
        inference_world_size: int,
        timeout: int,
        quantize_in_weight_transfer: bool = False,
        session_id: str = "default",
    ) -> None:
        super().init_broadcaster(
            host,
            port,
            rank_offset,
            inference_world_size,
            timeout,
            quantize_in_weight_transfer,
            session_id,
        )
        self._receiver_routes: list[list[ReceiverRoute]] = []
        self._trainer_peers_by_group: list[list[str]] = []
        self._group_generations: list[int] = []

    def choose_receive_buffer_count(
        self,
        receive_buffer_elements: dict[torch.dtype, int],
        staging_buffer_count: int,
    ) -> int:
        buffer_bytes = sum(
            elements * dtype.itemsize for dtype, elements in receive_buffer_elements.items()
        )
        if buffer_bytes == 0:
            return staging_buffer_count

        available = size_cuda_buffers(
            buffer_bytes,
            staging_buffer_count,
            self.device,
            extra_headroom_bytes=buffer_bytes,
        )
        if available < staging_buffer_count:
            raise RuntimeError(
                "NIXL push requires matching trainer and inference staging counts, but inference has "
                f"memory for {available} of {staging_buffer_count} requested buffers"
            )
        return staging_buffer_count

    def prepare_group_pulls(
        self,
        table: TrainerTensorTable,
        local_descs: dict[int, list[MemDesc]],
        remote_descs: dict[int, list[MemDesc]],
        peer_names: dict[int, str],
    ) -> list[tuple[object, object, list[int]]]:
        """Record trainer-to-inference routes for subsequent NIXL writes."""
        routes: list[ReceiverRoute] = []
        trainer_peers: list[str] = []
        for agent_index, sources in sorted(remote_descs.items()):
            destinations = local_descs[agent_index]
            if len(sources) != len(destinations):
                raise RuntimeError("NIXL push source and destination route counts differ")
            peer_name = peer_names.get(agent_index)
            if peer_name is None:
                peer_name = self.nixl_agent.add_remote_agent(table.agents[agent_index].metadata)
                self.nixl_agent.make_connection(peer_name)
                peer_names[agent_index] = peer_name
            trainer_peers.append(peer_name)
            routes.extend(
                ReceiverRoute(
                    trainer_agent_name=table.agents[agent_index].name,
                    source=source,
                    destination=destination,
                )
                for source, destination in zip(sources, destinations)
            )
        self._receiver_routes.append(routes)
        self._trainer_peers_by_group.append(trainer_peers)
        self._group_generations.append(0)
        return []

    def _wait_for_group_ready(self, group_index: int, cancelled: Event) -> None:
        notification = group_notification(group_index, self._group_generations[group_index])
        expected = {
            peer_name: notification
            for peer_name in self._trainer_peers_by_group[group_index]
        }
        if expected:
            self.nixl_agent.wait_for_notifications(
                expected,
                timeout=self.weight_transfer_timeout,
                cancelled=cancelled.is_set,
            )

    def _acknowledge_group(self, group_index: int, cancelled: Event) -> None:
        del cancelled
        notification = group_notification(group_index, self._group_generations[group_index])
        for peer_name in self._trainer_peers_by_group[group_index]:
            self.nixl_agent.send_notification(peer_name, notification)
        self._group_generations[group_index] += 1

    def transfer_metadata(
        self,
        table: TrainerTensorTable,
        plan: WeightTransferPlan,
    ) -> bytes:
        if len(self._receiver_routes) != len(plan.groups):
            raise RuntimeError("NIXL push route groups do not match the transfer plan")
        receiver_table = ReceiverTable(
            agent=ReceiverAgent(
                name=self.nixl_agent.name,
                metadata=self.nixl_agent.get_metadata(),
                device_id=self.device.index,
                rank=self.model_express.rank,
            ),
            groups=[
                ReceiverGroup(name=group.name, routes=routes)
                for group, routes in zip(plan.groups, self._receiver_routes)
            ],
        )
        return receiver_table.encode()
