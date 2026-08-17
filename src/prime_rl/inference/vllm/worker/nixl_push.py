"""vLLM worker for trainer-initiated writes into bounded reload buffers."""

from __future__ import annotations

import torch

from prime_rl.inference.vllm.worker.nixl import NIXLWeightUpdateWorker, WeightTransferPlan
from prime_rl.trainer.rl.broadcast.nixl.agent import MemDesc
from prime_rl.trainer.rl.broadcast.nixl.receiver_table import (
    ReceiverAgent,
    ReceiverGroup,
    ReceiverRoute,
    ReceiverTable,
)
from prime_rl.trainer.rl.broadcast.nixl.trainer_tensor_table import TrainerTensorTable


class NIXLPushWeightUpdateWorker(NIXLWeightUpdateWorker):
    """Receive canonical tensors by NIXL WRITE and replay vLLM's load graph."""

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

    def choose_receive_buffer_count(
        self,
        receive_buffer_elements: dict[torch.dtype, int],
        staging_buffer_count: int,
    ) -> int:
        del receive_buffer_elements, staging_buffer_count
        return 1

    def prepare_group_pulls(
        self,
        table: TrainerTensorTable,
        local_descs: dict[int, list[MemDesc]],
        remote_descs: dict[int, list[MemDesc]],
        peer_names: dict[int, str],
    ) -> list[tuple[object, object, list[int]]]:
        del peer_names
        routes: list[ReceiverRoute] = []
        for agent_index, sources in sorted(remote_descs.items()):
            destinations = local_descs[agent_index]
            if len(sources) != len(destinations):
                raise RuntimeError("NIXL push source and destination route counts differ")
            routes.extend(
                ReceiverRoute(
                    trainer_agent_name=table.agents[agent_index].name,
                    source=source,
                    destination=destination,
                )
                for source, destination in zip(sources, destinations)
            )
        self._receiver_routes.append(routes)
        return []

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
