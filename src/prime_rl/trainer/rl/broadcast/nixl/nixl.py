"""Serve FSDP weight shards through reusable typed NIXL arenas."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from prime_rl.configs.trainer import NIXLWeightBroadcastConfig
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.broadcast.nixl.agent import NixlAgent, make_agent_name, set_ucx_env_defaults
from prime_rl.trainer.rl.broadcast.nixl.cuda_malloc_memory import (
    size_cuda_buffers,
    use_cuda_malloc_pool,
)
from prime_rl.trainer.rl.broadcast.nixl.delta_manifest import NIXLPolicyMetadata
from prime_rl.trainer.rl.broadcast.nixl.fp8 import FP8ResidentProducer
from prime_rl.trainer.rl.broadcast.nixl.model_express import ModelExpressSession
from prime_rl.trainer.rl.broadcast.nixl.notifications import (
    NIXLNotificationInbox,
    group_notification,
    wait_for_notifications,
)
from prime_rl.trainer.rl.broadcast.nixl.trainer_tensor_table import (
    TrainerAgent,
    TrainerGroup,
    TrainerShard,
    TrainerTensor,
    TrainerTensorTable,
)
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.weight_sync.fp8 import FP8ScaleFormat
from prime_rl.weight_sync.grouping import LAYER_RE

MAX_STAGING_BUFFER_COUNT = 8


@dataclass
class StagedTensorShard:
    name: str
    global_shape: tuple[int, ...]
    group_index: int
    tensor_offset: int
    source_tensor: torch.Tensor
    wire_dtype: torch.dtype
    staging_tensor: torch.Tensor | None = None

    def assign_staging_tensor(self, arena: torch.Tensor, arena_offset: int) -> None:
        self.staging_tensor = arena.narrow(0, arena_offset, self.source_tensor.numel()).view(self.source_tensor.shape)

    def copy_to_staging(self) -> None:
        assert self.staging_tensor is not None
        self.staging_tensor.copy_(self.source_tensor)


@dataclass(frozen=True)
class TransferGroupIndex:
    group_names: list[str]
    layer_to_group: dict[int, int]


class NIXLWeightBroadcast(WeightBroadcast):
    def __init__(
        self,
        output_dir: Path,
        config: NIXLWeightBroadcastConfig,
        parallel_dims: ParallelDims,
        *,
        retain_fp8_resident: bool = False,
    ) -> None:
        super().__init__(output_dir)
        self.config = config
        self.parallel_dims = parallel_dims
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        if self.is_serving_rank:
            set_ucx_env_defaults()
            self.nixl_agent = NixlAgent(make_agent_name("trainer", self.world.rank))
        self.initialized = False
        self.transfer_group_names: list[str] = []
        self.staged_shards: list[StagedTensorShard] = []
        self.staged_shards_by_group: dict[int, list[StagedTensorShard]] = {}
        self.staging_arenas: dict[torch.dtype, torch.Tensor] = {}
        self.staging_registrations: list[object] = []
        self.full_transfer_resources_active = False
        self.staging_buffer_count: int
        self.trainer_table: TrainerTensorTable | None = None
        self.last_broadcast_step: int | None = None
        self.inference_notification_peers: dict[int, str] = {}
        self.notification_inbox = NIXLNotificationInbox()
        self.fp8_producer = (
            FP8ResidentProducer(
                device=torch.device("cuda", torch.cuda.current_device()),
                scale_format=cast(FP8ScaleFormat, self.config.delta_fp8_scale_format),
                bucket_bytes=self.config.delta_adam_bucket_mb * 1024 * 1024,
                pipeline_depth=self.config.delta_pipeline_depth,
                rank=self.world.rank,
                world_size=self.world.world_size,
                inference_tp_size=self.config.inference_world_size,
                retain_resident=retain_fp8_resident,
            )
            if self.config.delta_representation == "fp8_kernel"
            else None
        )

    @property
    def is_serving_rank(self) -> bool:
        if self.parallel_dims.dp_replicate_enabled:
            return self.parallel_dims.get_mesh("dp_replicate").get_local_rank() == 0
        return True

    @staticmethod
    def build_transfer_group_index(state_dict: dict[str, torch.Tensor]) -> TransferGroupIndex:
        layer_numbers = sorted(
            {
                int(match.group(1))
                for name, value in state_dict.items()
                if value.is_floating_point() and (match := LAYER_RE.search(name)) is not None
            }
        )
        return TransferGroupIndex(
            group_names=["non_layer", *(f"layer.{layer}" for layer in layer_numbers)],
            layer_to_group={layer: group for group, layer in enumerate(layer_numbers, start=1)},
        )

    @staticmethod
    def find_transfer_group_index(tensor_name: str, transfer_groups: TransferGroupIndex) -> int:
        match = LAYER_RE.search(tensor_name)
        return 0 if match is None else transfer_groups.layer_to_group[int(match.group(1))]

    def resolve_wire_dtype(
        self,
        tensor_name: str,
        value: torch.Tensor,
        keep_in_fp32: Callable[[str], bool],
    ) -> torch.dtype:
        if self.config.delta_representation == "fp8_kernel":
            return value.dtype
        return torch.float32 if keep_in_fp32(tensor_name) else torch.bfloat16

    def collect_local_tensor_shards(
        self,
        state_dict: dict[str, torch.Tensor],
        transfer_groups: TransferGroupIndex,
        keep_in_fp32: Callable[[str], bool],
    ) -> list[StagedTensorShard]:
        local_shards: list[StagedTensorShard] = []
        for name, value in state_dict.items():
            # Non-floating state is not part of model weight transfer.
            if not value.is_floating_point() and not (
                self.config.delta_representation == "fp8_kernel" and value.dtype == torch.uint8
            ):
                continue
            full_shape = tuple(value.shape)
            group_index = self.find_transfer_group_index(name, transfer_groups)
            wire_dtype = self.resolve_wire_dtype(name, value, keep_in_fp32)

            # Unsharded tensors are identical on every rank, so rank 0 serves the only copy.
            if not isinstance(value, DTensor):
                if self.config.delta_representation == "fp8_kernel" or self.world.is_master:
                    local_shards.append(
                        StagedTensorShard(
                            name=name,
                            global_shape=full_shape,
                            group_index=group_index,
                            tensor_offset=0,
                            source_tensor=value.detach(),
                            wire_dtype=wire_dtype,
                        )
                    )
                continue

            placements = value.placements
            local_shape, global_offset = compute_local_shape_and_global_offset(
                value.shape, value.device_mesh, placements
            )
            local = value.to_local().detach()
            if tuple(local.shape) != tuple(local_shape):
                local = local[tuple(slice(size) for size in local_shape)]

            # Replicated DTensors are identical on every rank, so rank 0 serves the only copy.
            if all(placement.is_replicate() for placement in placements):
                if self.world.is_master:
                    local_shards.append(
                        StagedTensorShard(
                            name=name,
                            global_shape=full_shape,
                            group_index=group_index,
                            tensor_offset=0,
                            source_tensor=local,
                            wire_dtype=wire_dtype,
                        )
                    )
                continue

            # FSDP DTensors contribute this rank's contiguous shard along tensor dimension 0.
            if local.numel():
                row_numel = prod(full_shape[1:]) if full_shape else 1
                offset = global_offset[0] * row_numel if full_shape else 0
                local_shards.append(
                    StagedTensorShard(
                        name=name,
                        global_shape=full_shape,
                        group_index=group_index,
                        tensor_offset=offset,
                        source_tensor=local,
                        wire_dtype=wire_dtype,
                    )
                )
        return local_shards

    def choose_staging_buffer_count(self, largest_group_bytes: int) -> int:
        local_buffer_count = min(len(self.transfer_group_names), MAX_STAGING_BUFFER_COUNT)
        if self.fp8_producer is not None:
            local_buffer_count = 1
        if self.is_serving_rank and largest_group_bytes:
            device = self.staged_shards[0].source_tensor.device
            allocated_bytes = torch.cuda.memory_allocated()
            peak_growth_bytes = max(0, torch.cuda.max_memory_allocated() - allocated_bytes)
            free_bytes, _ = torch.cuda.mem_get_info(device)
            max_buffers = local_buffer_count if peak_growth_bytes else 1
            if peak_growth_bytes or free_bytes < largest_group_bytes:
                torch.cuda.empty_cache()
            local_buffer_count = size_cuda_buffers(
                largest_group_bytes,
                max_buffers,
                device,
                extra_headroom_bytes=peak_growth_bytes,
            )

        staging_buffer_count = torch.tensor(
            local_buffer_count,
            dtype=torch.int64,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        dist.all_reduce(staging_buffer_count, op=dist.ReduceOp.MIN)
        return int(staging_buffer_count.item())

    def allocate_staging_arenas(self, largest_group_elements: dict[torch.dtype, int]) -> None:
        if not self.is_serving_rank or not any(largest_group_elements.values()):
            return

        device = self.staged_shards[0].source_tensor.device
        with use_cuda_malloc_pool():
            self.staging_arenas = {
                dtype: torch.empty(
                    self.staging_buffer_count * elements,
                    dtype=dtype,
                    device=device,
                )
                for dtype, elements in largest_group_elements.items()
                if elements
            }

        offsets = {
            dtype: [
                (group % self.staging_buffer_count) * largest_group_elements[dtype]
                for group in range(len(self.transfer_group_names))
            ]
            for dtype in self.staging_arenas
        }
        for shard in self.staged_shards:
            group_offsets = offsets[shard.wire_dtype]
            shard.assign_staging_tensor(
                self.staging_arenas[shard.wire_dtype],
                group_offsets[shard.group_index],
            )
            group_offsets[shard.group_index] += shard.source_tensor.numel()

        for arena in self.staging_arenas.values():
            self.staging_registrations.append(self.nixl_agent.register_tensor(arena))

    def prepare_staging_buffers(self) -> None:
        wire_dtypes = {shard.wire_dtype for shard in self.staged_shards}
        group_elements = {dtype: [0] * len(self.transfer_group_names) for dtype in wire_dtypes}
        for shard in self.staged_shards:
            group_elements[shard.wire_dtype][shard.group_index] += shard.source_tensor.numel()
        largest_group_elements = {dtype: max(elements, default=0) for dtype, elements in group_elements.items()}
        largest_group_bytes = sum(elements * dtype.itemsize for dtype, elements in largest_group_elements.items())
        self.staging_buffer_count = self.choose_staging_buffer_count(largest_group_bytes)
        self.allocate_staging_arenas(largest_group_elements)

        grouped: dict[int, list[StagedTensorShard]] = defaultdict(list)
        for shard in self.staged_shards:
            grouped[shard.group_index].append(shard)
        self.staged_shards_by_group = dict(grouped)

    def build_local_trainer_table_fragment(self) -> TrainerTensorTable:
        tensors_by_group: list[dict[str, TrainerTensor]] = [{} for _ in self.transfer_group_names]
        for shard in self.staged_shards:
            tensors = tensors_by_group[shard.group_index]
            tensor = tensors.setdefault(
                shard.name,
                TrainerTensor(
                    name=shard.name,
                    wire_dtype=str(shard.wire_dtype).removeprefix("torch."),
                    shape=shard.global_shape,
                    shards=[],
                ),
            )
            tensor.shards.append(
                TrainerShard(
                    agent=0,
                    offset=shard.tensor_offset,
                    numel=shard.source_tensor.numel(),
                    addr=cast(torch.Tensor, shard.staging_tensor).data_ptr(),
                )
            )

        return TrainerTensorTable(
            agents=[
                TrainerAgent(
                    name=self.nixl_agent.name,
                    metadata=self.nixl_agent.get_metadata(),
                    device_id=torch.cuda.current_device(),
                )
            ],
            staging_buffer_count=self.staging_buffer_count,
            groups=[
                TrainerGroup(name=group_name, tensors=list(tensors.values()))
                for group_name, tensors in zip(self.transfer_group_names, tensors_by_group)
            ],
            representation=self.config.delta_representation,
            fp8_scale_format=(
                self.config.delta_fp8_scale_format if self.config.delta_representation == "fp8_kernel" else ""
            ),
        )

    def gather_trainer_table_fragments(self) -> list[bytes] | None:
        table_fragment = self.build_local_trainer_table_fragment().encode() if self.is_serving_rank else None
        gathered: list[bytes | None] | None = [None] * self.world.world_size if self.world.is_master else None
        dist.gather_object(table_fragment, gathered, dst=0)
        if gathered is None:
            return None
        return [fragment for fragment in gathered if fragment is not None]

    def merge_trainer_table_fragments(self, table_fragments: list[bytes]) -> TrainerTensorTable:
        agents: list[TrainerAgent] = []
        tensors_by_group: list[dict[str, TrainerTensor]] = [{} for _ in self.transfer_group_names]
        for agent_index, encoded_fragment in enumerate(table_fragments):
            fragment = TrainerTensorTable.decode(encoded_fragment)
            if fragment.representation != self.config.delta_representation or fragment.fp8_scale_format != (
                self.config.delta_fp8_scale_format if self.config.delta_representation == "fp8_kernel" else ""
            ):
                raise RuntimeError("trainer ranks produced incompatible NIXL tensor representations")
            agents.append(fragment.agents[0])
            for group_index, group in enumerate(fragment.groups):
                tensors = tensors_by_group[group_index]
                for fragment_tensor in group.tensors:
                    tensor = tensors.setdefault(
                        fragment_tensor.name,
                        TrainerTensor(
                            name=fragment_tensor.name,
                            wire_dtype=fragment_tensor.wire_dtype,
                            shape=fragment_tensor.shape,
                            shards=[],
                        ),
                    )
                    tensor.shards.extend(
                        TrainerShard(
                            agent=agent_index,
                            offset=shard.offset,
                            numel=shard.numel,
                            addr=shard.addr,
                        )
                        for shard in fragment_tensor.shards
                    )

        for tensors in tensors_by_group:
            for tensor in tensors.values():
                tensor.shards.sort(key=lambda shard: shard.offset)

        return TrainerTensorTable(
            agents=agents,
            staging_buffer_count=self.staging_buffer_count,
            groups=[
                TrainerGroup(name=group_name, tensors=list(tensors.values()))
                for group_name, tensors in zip(self.transfer_group_names, tensors_by_group)
            ],
            representation=self.config.delta_representation,
            fp8_scale_format=(
                self.config.delta_fp8_scale_format if self.config.delta_representation == "fp8_kernel" else ""
            ),
        )

    def initialize_transfer(
        self,
        model: nn.Module,
        state_dict: dict[str, torch.Tensor] | None = None,
        transfer_groups: TransferGroupIndex | None = None,
    ) -> None:
        if self.initialized:
            return
        model = cast(PreTrainedModelPrimeRL, model)
        state_dict = model.state_dict() if state_dict is None else state_dict
        transfer_groups = transfer_groups or self.build_transfer_group_index(state_dict)
        self.transfer_group_names = transfer_groups.group_names
        if self.is_serving_rank:
            self.staged_shards = self.collect_local_tensor_shards(
                state_dict,
                transfer_groups,
                model.keep_in_fp32_for_weight_transfer,
            )
        self.prepare_staging_buffers()
        self.full_transfer_resources_active = True
        table_fragments = self.gather_trainer_table_fragments()

        if table_fragments is not None:
            table = self.merge_trainer_table_fragments(table_fragments)
            self.trainer_table = table
            server_url = f"{self.config.host}:{self.config.port}"
            client = MxClient(server_url=server_url)
            self.model_express = ModelExpressSession(
                client=client,
                role="trainer",
                rank=0,
                session_id=self.config.session_id,
                worker_id="trainer-table",
            )
            self.model_express.publish(nixl_metadata=table.encode())
            self.model_express.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
            self.update_session = ModelExpressSession(
                client=client,
                role="trainer",
                rank=0,
                session_id=f"{self.config.session_id}:updates",
                worker_id="trainer-update",
            )
            self.update_session.publish(nixl_metadata=NIXLPolicyMetadata(kind="full", payload=table.encode()).encode())
            self.update_session.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
            tensor_count = sum(len(group.tensors) for group in table.groups)
            self.logger.info(
                f"Published {tensor_count} trainer tensors in {len(table.groups)} groups "
                f"from {len(table.agents)} agents with {self.staging_buffer_count} staging buffers"
            )
        self.initialized = True

    def update_staged_sources(self, state_dict: dict[str, torch.Tensor]) -> None:
        if not self.is_serving_rank:
            return
        for shard in self.staged_shards:
            try:
                source = state_dict[shard.name]
            except KeyError as error:
                raise RuntimeError(f"FP8 kernel snapshot is missing staged tensor {shard.name!r}") from error
            if tuple(source.shape) != shard.global_shape or source.dtype != shard.wire_dtype:
                raise RuntimeError(
                    f"FP8 kernel tensor {shard.name!r} changed representation: "
                    f"{shard.global_shape}/{shard.wire_dtype} -> {tuple(source.shape)}/{source.dtype}"
                )
            shard.source_tensor = source

    def stage_full_group(self, group_index: int) -> None:
        if not self.is_serving_rank:
            return
        for shard in self.staged_shards_by_group.get(group_index, ()):
            shard.copy_to_staging()
        torch.cuda.synchronize()

    def deregister_full_staging_arenas(self) -> None:
        if not self.is_serving_rank:
            return
        torch.cuda.synchronize()
        for registration in self.staging_registrations:
            self.nixl_agent.deregister_tensor(registration)

    def prepare_inference_notification_peers(self) -> None:
        if not self.world.is_master or self.inference_notification_peers:
            return
        refs = self.model_express.wait_for(
            "inference",
            count=self.config.inference_world_size,
            status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
            timeout=self.config.timeout,
        )
        for ref in refs:
            metadata = self.model_express.fetch(ref).nixl_metadata
            if not metadata:
                raise RuntimeError(f"inference rank {ref.worker_rank} published no NIXL agent metadata")
            peer_name = self.nixl_agent.add_remote_agent(metadata)
            self.nixl_agent.make_connection(peer_name)
            self.inference_notification_peers[ref.worker_rank] = peer_name

    def publish_group_ready(self, step: int, group_index: int) -> None:
        if not self.world.is_master:
            return
        buffer_index = group_index % self.staging_buffer_count
        for inference_rank, peer_name in self.inference_notification_peers.items():
            self.nixl_agent.send_notification(
                peer_name,
                group_notification(
                    session_id=self.config.session_id,
                    kind="ready",
                    step=step,
                    group_index=group_index,
                    buffer_index=buffer_index,
                    inference_rank=inference_rank,
                ).encode(),
            )

    def finish_staging_buffer_transfer(self, step: int, group_index: int) -> None:
        if self.world.is_master:
            buffer_index = group_index % self.staging_buffer_count
            expected = {
                peer_name: group_notification(
                    session_id=self.config.session_id,
                    kind="ack",
                    step=step,
                    group_index=group_index,
                    buffer_index=buffer_index,
                    inference_rank=inference_rank,
                )
                for inference_rank, peer_name in self.inference_notification_peers.items()
            }
            wait_for_notifications(
                self.nixl_agent,
                self.notification_inbox,
                expected,
                timeout=self.config.timeout,
                context=f"policy v{step} group {group_index} acknowledgements",
            )
        dist.barrier()

    def finish_policy_transfer(self) -> None:
        if self.world.is_master:
            self.model_express.wait_for(
                "inference",
                count=self.config.inference_world_size,
                status=p2p_pb2.SOURCE_STATUS_READY,
                timeout=self.config.timeout,
            )
            # Keep INITIALIZING visible until the orchestrator completes this cycle.
            # Otherwise a fast next step can publish READY before its polling watcher
            # observes the reset, leaving both sides waiting on different versions.
            self.model_express.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
            self.update_session.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
            self.model_express.wait_for(
                "orchestrator",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.config.timeout,
            )
        dist.barrier()

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        ready_runs = list(self.multi_run_manager.ready_to_update_idxs)
        if self.fp8_producer is not None:
            transfer_groups = self.build_transfer_group_index(model.state_dict())
            transfer_tensors = self.fp8_producer.build_transfer_tensors(model, include_checkpoint=True)
            if not self.initialized:
                self.initialize_transfer(model, transfer_tensors.checkpoint, transfer_groups)
            else:
                self.update_staged_sources(transfer_tensors.checkpoint)
            del transfer_tensors
        else:
            self.initialize_transfer(model)
        self.broadcast_full(step, ready_runs)

    def broadcast_full(self, step: int, ready_runs: list[int], *, reason: str | None = None) -> None:
        if not self.full_transfer_resources_active:
            raise RuntimeError(
                "NIXL full-transfer resources are inactive; full recovery must rebuild "
                "the checkpoint snapshot and transfer plan"
            )
        if self.world.is_master:
            assert self.trainer_table is not None
            self.update_session.publish(
                nixl_metadata=NIXLPolicyMetadata(
                    kind="full",
                    payload=self.trainer_table.encode(),
                    step=step,
                ).encode()
            )
            self.update_session.set_status(p2p_pb2.SOURCE_STATUS_READY)
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
            self.prepare_inference_notification_peers()

        for group, group_name in enumerate(self.transfer_group_names):
            group_start = time.perf_counter()
            buffer_index = group % self.staging_buffer_count
            if group >= self.staging_buffer_count:
                self.finish_staging_buffer_transfer(step, group - self.staging_buffer_count)

            self.stage_full_group(group)
            dist.barrier()
            if self.world.is_master:
                self.publish_group_ready(step, group)
                self.logger.debug(
                    f"NIXL+ModelExpress policy v{step} group {group_name} staged in buffer {buffer_index} in "
                    f"{time.perf_counter() - group_start:.2f}s"
                )

        first_pending_group = max(0, len(self.transfer_group_names) - self.staging_buffer_count)
        for group in range(first_pending_group, len(self.transfer_group_names)):
            self.finish_staging_buffer_transfer(step, group)

        self.finish_policy_transfer()
        self.last_broadcast_step = step
        for run_index in ready_runs:
            self.multi_run_manager.ready_to_update[run_index] = False
        reason_suffix = f" ({reason})" if reason is not None else ""
        self.logger.info(
            f"NIXL+ModelExpress full policy v{step} synchronized in {time.perf_counter() - start:.2f}s{reason_suffix}"
        )
        self.after_full_transfer()

    def after_full_transfer(self) -> None:
        return None
