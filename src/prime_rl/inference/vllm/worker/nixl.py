"""vLLM worker extension for composed, sharded NIXL weight pulls."""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
from math import prod
from threading import Event
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import update_mla_absorbed_weights
from prime_rl.trainer.rl.broadcast.nixl.agent import MemDesc, NixlAgent, make_agent_name, set_ucx_env_defaults
from prime_rl.trainer.rl.broadcast.nixl.cuda_malloc_memory import (
    size_cuda_buffers,
    use_cuda_malloc_pool,
)
from prime_rl.trainer.rl.broadcast.nixl.delta_manifest import (
    NIXLDeltaFrame,
    NIXLDeltaManifest,
    NIXLPolicyMetadata,
)
from prime_rl.trainer.rl.broadcast.nixl.graph import (
    Destination,
    OperationChain,
    RecordedCopy,
    TensorReplayPlan,
    WeightLoadRecorder,
    apply_chain,
    chain_preserves_dtype,
    make_hf_lazy_weights,
    plan_tensor_replay,
)
from prime_rl.trainer.rl.broadcast.nixl.model_express import ModelExpressSession
from prime_rl.trainer.rl.broadcast.nixl.tensor_routing import TensorRoute, route_sharded_tensor
from prime_rl.trainer.rl.broadcast.nixl.trainer_tensor_table import TrainerTensor, TrainerTensorTable
from prime_rl.weight_sync.xor_delta import (
    NVCOMP_FRAME_ALIGNMENT,
    CompressedDeltaFrame,
    DeltaTensorMetadata,
    NvcompLZ4Codec,
    align_nvcomp_nbytes,
    integer_view,
    unpack_delta_frame,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nixl")
_BUFFER_POLL_INTERVAL = 0.01


@dataclass
class TensorCopyPlan:
    recorded_copy: RecordedCopy
    staging_tensor: torch.Tensor
    source_tensor: TrainerTensor
    source_plan: TensorReplayPlan
    replay_ops: OperationChain
    routes: list[TensorRoute]


@dataclass
class LayerWeightTransferPlan:
    reload_layer: nn.Module | None
    copies: list[TensorCopyPlan]
    persistent_copies: list[TensorCopyPlan]

    @property
    def destination_names(self) -> set[str]:
        return {plan.recorded_copy.destination_name for plan in self.copies}


@dataclass
class WeightTransferGroup:
    name: str
    layers: list[LayerWeightTransferPlan]
    pulls: list[tuple[Any, Any, list[int]]]
    pull_nbytes: int
    required_delta_sources: frozenset[tuple[int, str]]
    required_delta_nbytes: int


@dataclass
class WeightTransferPlan:
    table: TrainerTensorTable
    receive_arenas: dict[torch.dtype, torch.Tensor]
    receive_buffer_count: int
    groups: list[WeightTransferGroup]


@dataclass
class DeltaGroupMetrics:
    published_frames: int = 0
    pulled_frames: int = 0
    published_bytes: int = 0
    pulled_bytes: int = 0
    published_uncompressed_bytes: int = 0
    pulled_uncompressed_bytes: int = 0
    routed_bytes: int = 0
    wait_seconds: float = 0.0
    pull_seconds: float = 0.0
    acknowledge_seconds: float = 0.0


@dataclass
class PreparedDeltaGroup:
    transfer_group: WeightTransferGroup
    decoded: dict[tuple[int, str], tuple[torch.Tensor, DeltaTensorMetadata]]
    decode_started: torch.cuda.Event
    decode_finished: torch.cuda.Event
    metrics: DeltaGroupMetrics


@dataclass
class FullGroupMetrics:
    wait_seconds: float = 0.0
    pull_seconds: float = 0.0
    acknowledge_seconds: float = 0.0


def _covered_nbytes(ranges: list[tuple[int, int]]) -> int:
    if not ranges:
        return 0
    ordered = sorted(ranges)
    covered = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            covered += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return covered + end - start


class NIXLWeightUpdateWorker(Worker):
    @property
    def raw_model(self) -> nn.Module:
        return cast(nn.Module, self.model_runner.get_model())

    def liveness_probe(self) -> None:
        return None

    def init_broadcaster(
        self,
        host: str,
        port: int,
        rank_offset: int,
        inference_world_size: int,
        timeout: int,
        quantize_in_weight_transfer: bool = False,
        session_id: str = "default",
        delta_mode: str = "none",
    ) -> None:
        del inference_world_size, quantize_in_weight_transfer
        global_rank = rank_offset + self.device.index
        server_url = f"{host}:{port}"
        set_ucx_env_defaults()
        self.nixl_agent = NixlAgent(make_agent_name("inference", global_rank))
        self.model_express = ModelExpressSession(
            client=MxClient(server_url=server_url),
            role="inference",
            rank=global_rank,
            session_id=session_id,
            worker_id=f"inference-{global_rank}",
        )
        self.weight_transfer_timeout = timeout
        self.delta_mode = delta_mode
        self.current_step: int | None = None
        self.full_sync_required = True
        self.delta_codec = NvcompLZ4Codec(self.device) if delta_mode == "xor" else None
        self.delta_peer_metadata: dict[str, bytes] = {}
        self.delta_peer_names: dict[str, str] = {}
        self.delta_receive_arenas: list[torch.Tensor] = []
        self.delta_receive_registrations: list[Any] = []
        self.delta_receive_slot_bytes = 0
        self.delta_prefetch_stream = torch.cuda.Stream(device=self.device)
        self.receive_registrations: list[Any] = []
        self.weight_transfer_plan: WeightTransferPlan | None = None
        self.update_session = ModelExpressSession(
            client=self.model_express.client,
            role="inference",
            rank=global_rank,
            session_id=f"{session_id}:updates",
            worker_id=f"inference-update-{global_rank}",
        )
        logger.info(
            "NIXL worker configured: global_rank=%d, ModelExpress=%s, session=%s",
            global_rank,
            server_url,
            session_id,
        )

    @torch.no_grad()
    def initialize_transfer(self) -> WeightTransferPlan:
        if self.weight_transfer_plan is not None:
            return self.weight_transfer_plan

        trainer_ref = self.model_express.wait_for(
            "trainer",
            count=1,
            status=None,
            timeout=self.weight_transfer_timeout,
        )[0]
        table = TrainerTensorTable.decode(self.model_express.fetch(trainer_ref).nixl_metadata)
        copies = self.trace_weight_loads(table)
        plan = self.build_transfer_plan(table, copies)
        if self.delta_mode == "xor":
            self.validate_delta_transfer_plan(plan)
        self.buffer_sessions = []
        for buffer_index in range(table.staging_buffer_count):
            session = ModelExpressSession(
                client=self.model_express.client,
                role="inference",
                rank=self.model_express.rank,
                session_id=f"{self.model_express.session_id}:layers:{buffer_index}",
                worker_id=f"inference-buffer-{self.model_express.rank}-{buffer_index}",
            )
            session.publish()
            session.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
            self.buffer_sessions.append(session)
        # Join the current generation directly. Publishing a transient READY
        # before the first pull would let the trainer mistake initialization
        # for a completed acknowledgement.
        self.model_express.publish(nixl_metadata=self.nixl_agent.get_metadata())
        self.model_express.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
        self.weight_transfer_plan = plan
        logger.info(
            "Initialized NIXL transfer plan on rank %d with %d groups",
            self.model_express.rank,
            len(plan.groups),
        )
        return plan

    @staticmethod
    def validate_delta_transfer_plan(plan: WeightTransferPlan) -> None:
        for group in plan.groups:
            for layer in group.layers:
                for copy_plan in (*layer.copies, *layer.persistent_copies):
                    copy = copy_plan.recorded_copy
                    source = copy_plan.source_tensor
                    source_dtype = getattr(torch, source.wire_dtype)
                    if not chain_preserves_dtype(tuple(source.shape), source_dtype, copy.ops):
                        raise RuntimeError(
                            f"NIXL XOR route for {copy.source_name!r} changes dtype during weight loading"
                        )
                    parameter = getattr(copy.destination_module, copy.destination_name)
                    destination = parameter.as_strided(
                        copy.destination_shape,
                        copy.destination_stride,
                        copy.destination_offset,
                    )
                    value = apply_chain(copy_plan.staging_tensor, copy_plan.replay_ops)
                    if value.dtype != destination.dtype or tuple(value.shape) != tuple(destination.shape):
                        raise RuntimeError(
                            f"NIXL XOR route for {copy.source_name!r} changes representation: "
                            f"{value.dtype}{tuple(value.shape)} -> "
                            f"{destination.dtype}{tuple(destination.shape)}"
                        )

    def trace_weight_loads(
        self,
        table: TrainerTensorTable,
    ) -> list[RecordedCopy]:
        """Trace vLLM weight loading into source-to-destination copies."""
        from vllm.model_executor.model_loader.reload.layerwise import (
            _get_original_loader,
            initialize_layerwise_reload,
        )
        from vllm.model_executor.model_loader.reload.meta import SKIP_TENSORS
        from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

        model = self.raw_model
        recorder = WeightLoadRecorder()
        regular_by_layer: dict[int, list[RecordedCopy]] = defaultdict(list)
        persistent: list[RecordedCopy] = []
        original_loaders: list[tuple[torch.Tensor, Any]] = []
        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(model)
            try:
                for module in model.modules():
                    for name, tensor in get_layer_tensors(module).items():
                        destination = Destination(module, name, tensor)
                        if not tensor.is_meta:
                            recorder.register_destination_storage(destination)
                        loader = _get_original_loader(tensor)
                        original_loaders.append((tensor, loader))
                        tensor.weight_loader = self.wrap_weight_loader_for_recording(
                            recorder,
                            destination,
                            loader,
                        )

                model.load_weights(
                    make_hf_lazy_weights(
                        table,
                        device=self.device,
                        recorder=recorder,
                        hf_config=self.model_runner.model_config.hf_text_config,
                    )
                )

                for copy in recorder.copies:
                    if copy.is_persistent or copy.destination_name in SKIP_TENSORS:
                        copy.is_persistent = True
                        persistent.append(copy)
                    else:
                        regular_by_layer[id(copy.destination_module)].append(copy)
            finally:
                try:
                    for tensor, loader in reversed(original_loaders):
                        tensor.weight_loader = loader
                finally:
                    self._restore_layerwise_state(model)

        regular = [copy for copies in regular_by_layer.values() for copy in copies]
        return regular + persistent

    @staticmethod
    def wrap_weight_loader_for_recording(
        recorder: WeightLoadRecorder,
        destination: Destination,
        loader: Any,
    ):
        @wraps(loader)
        def recording_loader(*args, **kwargs):
            recorder.active_destination = destination
            try:
                return loader(*args, **kwargs)
            finally:
                recorder.active_destination = None

        return recording_loader

    @staticmethod
    def _restore_layerwise_state(model: nn.Module) -> None:
        from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO, _place_kernel_tensors

        for layer in model.modules():
            info = LAYERWISE_INFO.get(layer)
            if info is not None and info.can_load():
                if info.kernel_tensors is not None:
                    _place_kernel_tensors(layer, info)
                info.reset()
        if hasattr(model, "_original_do_torchao_reload"):
            model._do_torchao_reload = model._original_do_torchao_reload

    def build_transfer_plan(
        self,
        table: TrainerTensorTable,
        copies: list[RecordedCopy],
    ) -> WeightTransferPlan:
        replay_plans = self.plan_tensor_replays(table, copies)
        receive_buffer_elements = self.calculate_receive_buffer_elements(
            table,
            copies,
            replay_plans,
        )
        receive_buffer_count = self.choose_receive_buffer_count(
            receive_buffer_elements,
            table.staging_buffer_count,
        )
        receive_arenas = self.allocate_receive_arenas(
            receive_buffer_elements,
            receive_buffer_count,
        )
        groups = self.build_transfer_groups(
            table,
            copies,
            replay_plans,
            receive_buffer_elements,
            receive_arenas,
            receive_buffer_count,
        )
        return WeightTransferPlan(
            table=table,
            receive_arenas=receive_arenas,
            receive_buffer_count=receive_buffer_count,
            groups=groups,
        )

    def plan_tensor_replays(
        self,
        table: TrainerTensorTable,
        copies: list[RecordedCopy],
    ) -> dict[int, TensorReplayPlan]:
        tensors = {tensor.name: tensor for group in table.groups for tensor in group.tensors}
        replay_plans: dict[int, TensorReplayPlan] = {}
        for copy in copies:
            source = tensors[copy.source_name]
            replay_plans[id(copy)] = plan_tensor_replay(
                tuple(source.shape),
                getattr(torch, source.wire_dtype),
                copy.ops,
            )
        return replay_plans

    def calculate_receive_buffer_elements(
        self,
        table: TrainerTensorTable,
        copies: list[RecordedCopy],
        replay_plans: dict[int, TensorReplayPlan],
    ) -> dict[torch.dtype, int]:
        tensors = {tensor.name: tensor for group in table.groups for tensor in group.tensors}
        tensor_groups = {
            tensor.name: group_index for group_index, group in enumerate(table.groups) for tensor in group.tensors
        }
        group_elements: dict[torch.dtype, list[int]] = defaultdict(lambda: [0] * len(table.groups))
        for copy in copies:
            source = tensors[copy.source_name]
            source_dtype = getattr(torch, source.wire_dtype)
            group_elements[source_dtype][tensor_groups[source.name]] += prod(replay_plans[id(copy)].source_shape)
        return {dtype: max(elements, default=0) for dtype, elements in group_elements.items()}

    def choose_receive_buffer_count(
        self,
        receive_buffer_elements: dict[torch.dtype, int],
        staging_buffer_count: int,
    ) -> int:
        receive_buffer_bytes = max(
            1,
            sum(elements * dtype.itemsize for dtype, elements in receive_buffer_elements.items()),
        )
        allocated_bytes = torch.cuda.memory_allocated(self.device)
        peak_growth_bytes = max(
            0,
            torch.cuda.max_memory_allocated(self.device) - allocated_bytes,
        )
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        max_receive_buffers = min(2, staging_buffer_count) if peak_growth_bytes else 1
        if peak_growth_bytes or free_bytes < receive_buffer_bytes:
            torch.cuda.empty_cache()
        return size_cuda_buffers(
            receive_buffer_bytes,
            max_receive_buffers,
            self.device,
            extra_headroom_bytes=receive_buffer_bytes + peak_growth_bytes,
        )

    def allocate_receive_arenas(
        self,
        receive_buffer_elements: dict[torch.dtype, int],
        receive_buffer_count: int,
    ) -> dict[torch.dtype, torch.Tensor]:
        with use_cuda_malloc_pool():
            receive_arenas = {
                dtype: torch.empty(
                    receive_buffer_count * elements,
                    dtype=dtype,
                    device=self.device,
                )
                for dtype, elements in receive_buffer_elements.items()
                if elements
            }
        for arena in receive_arenas.values():
            self.receive_registrations.append(self.nixl_agent.register_tensor(arena))
        return receive_arenas

    def build_transfer_groups(
        self,
        table: TrainerTensorTable,
        copies: list[RecordedCopy],
        replay_plans: dict[int, TensorReplayPlan],
        receive_buffer_elements: dict[torch.dtype, int],
        receive_arenas: dict[torch.dtype, torch.Tensor],
        receive_buffer_count: int,
    ) -> list[WeightTransferGroup]:
        tensors = {tensor.name: tensor for group in table.groups for tensor in group.tensors}
        tensor_groups = {
            tensor.name: group_index for group_index, group in enumerate(table.groups) for tensor in group.tensors
        }
        copies_by_group: dict[int, list[RecordedCopy]] = defaultdict(list)
        for copy in copies:
            copies_by_group[tensor_groups[copy.source_name]].append(copy)

        reload_layers: dict[int, nn.Module] = {}
        for copy in copies:
            if not copy.is_persistent:
                reload_layers.setdefault(id(copy.destination_module), copy.destination_module)

        reload_layer_groups: dict[int, int] = {}
        for copy in copies:
            layer_id = id(copy.destination_module)
            if layer_id not in reload_layers:
                continue
            source_group = tensor_groups[copy.source_name]
            previous_group = reload_layer_groups.setdefault(layer_id, source_group)
            if previous_group != source_group:
                raise RuntimeError(
                    f"vLLM reload layer {type(copy.destination_module).__name__} reads trainer groups "
                    f"{table.groups[previous_group].name!r} and {table.groups[source_group].name!r}"
                )

        agent_devices = {agent_index: agent.device_id for agent_index, agent in enumerate(table.agents)}
        peer_names: dict[int, str] = {}
        transfer_groups: list[WeightTransferGroup] = []

        for group_index, group in enumerate(table.groups):
            copy_plans_by_layer: dict[int, list[TensorCopyPlan]] = defaultdict(list)
            persistent_plans_by_layer: dict[int, list[TensorCopyPlan]] = defaultdict(list)
            local_descs: dict[int, list[MemDesc]] = defaultdict(list)
            remote_descs: dict[int, list[MemDesc]] = defaultdict(list)
            cursors = {
                dtype: (group_index % receive_buffer_count) * elements
                for dtype, elements in receive_buffer_elements.items()
            }
            delta_source_ranges: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)

            for copy in copies_by_group[group_index]:
                replay_plan = replay_plans[id(copy)]
                source = tensors[copy.source_name]
                source_dtype = getattr(torch, source.wire_dtype)
                numel = prod(replay_plan.source_shape)
                cursor = cursors[source_dtype]
                staging_tensor = receive_arenas[source_dtype].narrow(0, cursor, numel).view(replay_plan.source_shape)
                cursors[source_dtype] += numel
                copy_plan = TensorCopyPlan(
                    recorded_copy=copy,
                    staging_tensor=staging_tensor,
                    source_tensor=source,
                    source_plan=replay_plan,
                    replay_ops=replay_plan.replay_ops,
                    routes=[],
                )
                copy_plan.routes = route_sharded_tensor(replay_plan, source, staging_tensor)
                plans = persistent_plans_by_layer if copy.is_persistent else copy_plans_by_layer
                plans[id(copy.destination_module)].append(copy_plan)

                for route in copy_plan.routes:
                    local_descs[route.agent].append((route.destination_addr, route.nbytes, self.device.index))
                    remote_descs[route.agent].append((route.source_addr, route.nbytes, agent_devices[route.agent]))
                    delta_source_ranges[(route.agent, copy.source_name)].append(
                        (route.source_addr, route.source_addr + route.nbytes)
                    )

            transfer_groups.append(
                WeightTransferGroup(
                    name=group.name,
                    layers=self.build_layer_transfer_plans(
                        reload_layers,
                        copy_plans_by_layer,
                        persistent_plans_by_layer,
                    ),
                    pulls=self.prepare_group_pulls(
                        table,
                        local_descs,
                        remote_descs,
                        peer_names,
                    ),
                    pull_nbytes=sum(size for descs in local_descs.values() for _addr, size, _device in descs),
                    required_delta_sources=frozenset(
                        (route.agent, copy_plan.recorded_copy.source_name)
                        for plans in (*copy_plans_by_layer.values(), *persistent_plans_by_layer.values())
                        for copy_plan in plans
                        for route in copy_plan.routes
                    ),
                    required_delta_nbytes=sum(_covered_nbytes(ranges) for ranges in delta_source_ranges.values()),
                )
            )
        return transfer_groups

    def build_layer_transfer_plans(
        self,
        reload_layers: dict[int, nn.Module],
        copy_plans_by_layer: dict[int, list[TensorCopyPlan]],
        persistent_plans_by_layer: dict[int, list[TensorCopyPlan]],
    ) -> list[LayerWeightTransferPlan]:
        layer_plans: list[LayerWeightTransferPlan] = []
        for layer_id, layer in reload_layers.items():
            copies = copy_plans_by_layer.get(layer_id, [])
            persistent_copies = persistent_plans_by_layer.get(layer_id, [])
            if copies:
                layer_plans.append(
                    LayerWeightTransferPlan(
                        reload_layer=layer,
                        copies=copies,
                        persistent_copies=persistent_copies,
                    )
                )
            elif persistent_copies:
                layer_plans.append(
                    LayerWeightTransferPlan(
                        reload_layer=None,
                        copies=[],
                        persistent_copies=persistent_copies,
                    )
                )

        remaining_persistent = [
            plan
            for layer_id, plans in persistent_plans_by_layer.items()
            if layer_id not in reload_layers
            for plan in plans
        ]
        if remaining_persistent:
            layer_plans.append(
                LayerWeightTransferPlan(
                    reload_layer=None,
                    copies=[],
                    persistent_copies=remaining_persistent,
                )
            )
        return layer_plans

    def prepare_group_pulls(
        self,
        table: TrainerTensorTable,
        local_descs: dict[int, list[MemDesc]],
        remote_descs: dict[int, list[MemDesc]],
        peer_names: dict[int, str],
    ) -> list[tuple[Any, Any, list[int]]]:
        pulls: list[tuple[Any, Any, list[int]]] = []
        for agent_index, remote in sorted(remote_descs.items()):
            peer_name = peer_names.get(agent_index)
            if peer_name is None:
                peer_name = self.nixl_agent.add_remote_agent(table.agents[agent_index].metadata)
                self.nixl_agent.make_connection(peer_name)
                peer_names[agent_index] = peer_name
            local_prepared = self.nixl_agent.prepare_xfer_dlist(local_descs[agent_index])
            remote_prepared = self.nixl_agent.prepare_xfer_dlist(remote, agent_name=peer_name)
            pulls.append((local_prepared, remote_prepared, list(range(len(remote)))))
        return pulls

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None) -> None:
        del weight_dir
        plan = self.initialize_transfer()
        self.model_express.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
        self.model_express.wait_for(
            "trainer",
            count=1,
            status=p2p_pb2.SOURCE_STATUS_READY,
            timeout=self.weight_transfer_timeout,
        )

        started = time.perf_counter()
        update_ref = self.update_session.wait_for(
            "trainer",
            count=1,
            status=p2p_pb2.SOURCE_STATUS_READY,
            timeout=self.weight_transfer_timeout,
        )[0]
        policy = NIXLPolicyMetadata.decode(self.update_session.fetch(update_ref).nixl_metadata)
        try:
            if policy.kind == "xor":
                if self.delta_mode != "xor":
                    raise RuntimeError("trainer published a NIXL XOR update while delta mode is disabled")
                if self.full_sync_required or self.current_step != policy.base_step:
                    raise RuntimeError(
                        f"cannot apply NIXL XOR delta {policy.base_step}->{policy.step}: "
                        f"resident policy is {self.current_step}, full_sync_required={self.full_sync_required}"
                    )
                manifest = NIXLDeltaManifest.decode(policy.payload)
                if (manifest.base_step, manifest.step) != (policy.base_step, policy.step):
                    raise RuntimeError(
                        "NIXL XOR manifest transition does not match its policy metadata: "
                        f"manifest={manifest.base_step}->{manifest.step}, "
                        f"policy={policy.base_step}->{policy.step}"
                    )
                self.apply_delta_manifest(plan, manifest)
            else:
                self.apply_transfer_plan(plan)
            update_mla_absorbed_weights(self.raw_model)
            torch.cuda.synchronize(self.device)
        except BaseException:
            self.current_step = None
            self.full_sync_required = True
            self.model_express.set_status(p2p_pb2.SOURCE_STATUS_STALE)
            raise
        self.current_step = policy.step
        self.full_sync_required = False
        self.model_express.set_status(p2p_pb2.SOURCE_STATUS_READY)
        logger.info(
            "Applied NIXL policy update on rank %d in %.2fs",
            self.model_express.rank,
            time.perf_counter() - started,
        )

    def apply_delta_manifest(self, plan: WeightTransferPlan, manifest: NIXLDeltaManifest) -> None:
        if manifest.base_step != self.current_step:
            raise RuntimeError(
                f"NIXL delta manifest base {manifest.base_step} does not match resident policy {self.current_step}"
            )
        if tuple(group.name for group in manifest.groups) != tuple(group.name for group in plan.groups):
            raise RuntimeError("NIXL delta manifest groups do not match the full-transfer plan")

        selected_by_group = [
            self.select_delta_frames(transfer_group, delta_group.frames)
            for transfer_group, delta_group in zip(plan.groups, manifest.groups, strict=True)
        ]
        receive_buffer_count = min(2, len(plan.groups))
        slot_bytes = max((self.packed_delta_frame_bytes(frames) for frames in selected_by_group), default=0)
        self.ensure_delta_receive_arenas(slot_bytes, receive_buffer_count)
        slot_events: list[torch.cuda.Event | None] = [None] * receive_buffer_count
        decode_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        apply_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        group_metrics: list[DeltaGroupMetrics] = []
        current_stream = torch.cuda.current_stream(self.device)
        pipeline_started = time.perf_counter()
        cancelled = Event()

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nixl-delta-prefetch")
        try:
            future = executor.submit(
                self.prepare_delta_group,
                0,
                plan.groups[0],
                manifest.groups[0].frames,
                selected_by_group[0],
                manifest,
                slot_events,
                cancelled,
            )
            for group_index in range(len(plan.groups)):
                prepared = future.result()
                if group_index + 1 < len(plan.groups):
                    future = executor.submit(
                        self.prepare_delta_group,
                        group_index + 1,
                        plan.groups[group_index + 1],
                        manifest.groups[group_index + 1].frames,
                        selected_by_group[group_index + 1],
                        manifest,
                        slot_events,
                        cancelled,
                    )

                current_stream.wait_event(prepared.decode_finished)
                # Decode allocates on a side stream; keep its storage alive through asynchronous apply.
                for value, _metadata in prepared.decoded.values():
                    value.record_stream(current_stream)
                apply_started = torch.cuda.Event(enable_timing=True)
                apply_finished = torch.cuda.Event(enable_timing=True)
                apply_started.record(current_stream)
                self.apply_decoded_delta_group(prepared.transfer_group, prepared.decoded)
                apply_finished.record(current_stream)
                decode_events.append((prepared.decode_started, prepared.decode_finished))
                apply_events.append((apply_started, apply_finished))
                group_metrics.append(prepared.metrics)
        finally:
            cancelled.set()
            executor.shutdown(wait=True, cancel_futures=True)

        synchronization_started = time.perf_counter()
        torch.cuda.synchronize(self.device)
        synchronization_seconds = time.perf_counter() - synchronization_started
        pipeline_seconds = time.perf_counter() - pipeline_started
        decode_seconds = sum(
            prepared_start.elapsed_time(prepared_end) / 1000
            for prepared_start, prepared_end in decode_events
        )
        apply_seconds = sum(start.elapsed_time(end) / 1000 for start, end in apply_events)
        self.log_delta_metrics(
            manifest,
            group_metrics,
            decode_seconds=decode_seconds,
            apply_seconds=apply_seconds,
            synchronization_seconds=synchronization_seconds,
            pipeline_seconds=pipeline_seconds,
        )

    def prepare_delta_group(
        self,
        group_index: int,
        transfer_group: WeightTransferGroup,
        published_frames: tuple[NIXLDeltaFrame, ...],
        selected_frames: list[NIXLDeltaFrame],
        manifest: NIXLDeltaManifest,
        slot_events: list[torch.cuda.Event | None],
        cancelled: Event,
    ) -> PreparedDeltaGroup:
        torch.cuda.set_device(self.device)
        slot = group_index % len(self.delta_receive_arenas)
        previous_event = slot_events[slot]
        if previous_event is not None:
            previous_event.synchronize()

        session = self.buffer_sessions[group_index % len(self.buffer_sessions)]
        wait_started = time.perf_counter()
        session.wait_for(
            "trainer",
            count=1,
            status=p2p_pb2.SOURCE_STATUS_READY,
            timeout=self.weight_transfer_timeout,
            poll_interval=_BUFFER_POLL_INTERVAL,
            cancelled=cancelled.is_set,
        )
        ready_wait_seconds = time.perf_counter() - wait_started

        pull_started = time.perf_counter()
        pulled = self.pull_delta_group(
            transfer_group,
            selected_frames,
            manifest,
            receive_slot=slot,
            cancelled=cancelled,
        )
        pull_seconds = time.perf_counter() - pull_started

        decode_started = torch.cuda.Event(enable_timing=True)
        decode_finished = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.delta_prefetch_stream):
            decode_started.record(self.delta_prefetch_stream)
            decoded = self.decode_delta_frames(pulled)
            decode_finished.record(self.delta_prefetch_stream)
        slot_events[slot] = decode_finished

        acknowledgement_started = time.perf_counter()
        session.set_status(p2p_pb2.SOURCE_STATUS_READY)
        session.wait_for(
            "trainer",
            count=1,
            status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
            timeout=self.weight_transfer_timeout,
            poll_interval=_BUFFER_POLL_INTERVAL,
            cancelled=cancelled.is_set,
        )
        session.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
        acknowledgement_seconds = time.perf_counter() - acknowledgement_started

        return PreparedDeltaGroup(
            transfer_group=transfer_group,
            decoded=decoded,
            decode_started=decode_started,
            decode_finished=decode_finished,
            metrics=DeltaGroupMetrics(
                published_frames=len(published_frames),
                pulled_frames=len(selected_frames),
                published_bytes=sum(frame.compressed_nbytes for frame in published_frames),
                pulled_bytes=sum(frame.compressed_nbytes for frame in selected_frames),
                published_uncompressed_bytes=sum(frame.uncompressed_nbytes for frame in published_frames),
                pulled_uncompressed_bytes=sum(frame.uncompressed_nbytes for frame in selected_frames),
                routed_bytes=transfer_group.required_delta_nbytes,
                wait_seconds=ready_wait_seconds,
                pull_seconds=pull_seconds,
                acknowledge_seconds=acknowledgement_seconds,
            ),
        )

    def pull_delta_group(
        self,
        transfer_group: WeightTransferGroup,
        selected: list[NIXLDeltaFrame],
        manifest: NIXLDeltaManifest,
        *,
        receive_slot: int,
        cancelled: Event,
    ) -> list[tuple[NIXLDeltaFrame, torch.Tensor]]:
        if not selected:
            return []
        logger.debug(
            "NIXL XOR rank %d pulling %d frames for %s (%.2f MiB compressed)",
            self.model_express.rank,
            len(selected),
            transfer_group.name,
            sum(frame.compressed_nbytes for frame in selected) / 2**20,
        )

        offsets: list[int] = []
        used = 0
        for frame in selected:
            used = align_nvcomp_nbytes(used)
            offsets.append(used)
            used += frame.compressed_nbytes
        arena = self.delta_receive_arenas[receive_slot]

        local_descs: dict[int, list[MemDesc]] = defaultdict(list)
        remote_descs: dict[int, list[MemDesc]] = defaultdict(list)
        payloads: list[torch.Tensor] = []
        for frame, offset in zip(selected, offsets, strict=True):
            payload = arena.narrow(0, offset, frame.compressed_nbytes)
            payloads.append(payload)
            local_descs[frame.agent].append((payload.data_ptr(), payload.numel(), self.device.index))
            agent = manifest.agents[frame.agent]
            remote_descs[frame.agent].append((frame.addr, frame.compressed_nbytes, agent.device_id))

        for agent_index, remote in sorted(remote_descs.items()):
            agent = manifest.agents[agent_index]
            peer_name = self.prepare_delta_peer(agent.name, agent.metadata)
            local = self.nixl_agent.prepare_xfer_dlist(local_descs[agent_index])
            remote_prepared = self.nixl_agent.prepare_xfer_dlist(remote, agent_name=peer_name)
            indices = list(range(len(remote)))
            handle = self.nixl_agent.post_read(local, indices, remote_prepared)
            self.nixl_agent.wait(
                handle,
                context=f"NIXL XOR pull for {transfer_group.name} from {agent.name}",
                timeout=self.weight_transfer_timeout,
                cancelled=cancelled.is_set,
            )
        return list(zip(selected, payloads, strict=True))

    def ensure_delta_receive_arenas(self, required_bytes: int, count: int) -> None:
        required_bytes = max(NVCOMP_FRAME_ALIGNMENT, align_nvcomp_nbytes(required_bytes))
        if len(self.delta_receive_arenas) == count and self.delta_receive_slot_bytes >= required_bytes:
            return
        for registration in self.delta_receive_registrations:
            self.nixl_agent.deregister_tensor(registration)
        self.delta_receive_slot_bytes = required_bytes
        with use_cuda_malloc_pool():
            self.delta_receive_arenas = [
                torch.empty(required_bytes, dtype=torch.uint8, device=self.device) for _ in range(count)
            ]
        self.delta_receive_registrations = [
            self.nixl_agent.register_tensor(arena) for arena in self.delta_receive_arenas
        ]

    @staticmethod
    def select_delta_frames(
        transfer_group: WeightTransferGroup,
        frames: tuple[NIXLDeltaFrame, ...],
    ) -> list[NIXLDeltaFrame]:
        required = transfer_group.required_delta_sources
        return [frame for frame in frames if any((frame.agent, tensor.name) in required for tensor in frame.tensors)]

    @staticmethod
    def packed_delta_frame_bytes(frames: list[NIXLDeltaFrame]) -> int:
        used = 0
        for frame in frames:
            used = align_nvcomp_nbytes(used)
            used += frame.compressed_nbytes
        return align_nvcomp_nbytes(used)

    def log_delta_metrics(
        self,
        manifest: NIXLDeltaManifest,
        groups: list[DeltaGroupMetrics],
        *,
        decode_seconds: float,
        apply_seconds: float,
        synchronization_seconds: float,
        pipeline_seconds: float,
    ) -> None:
        published_frames = sum(group.published_frames for group in groups)
        pulled_frames = sum(group.pulled_frames for group in groups)
        published_bytes = sum(group.published_bytes for group in groups)
        pulled_bytes = sum(group.pulled_bytes for group in groups)
        published_uncompressed_bytes = sum(group.published_uncompressed_bytes for group in groups)
        pulled_uncompressed_bytes = sum(group.pulled_uncompressed_bytes for group in groups)
        routed_bytes = sum(group.routed_bytes for group in groups)
        wait_seconds = sum(group.wait_seconds for group in groups)
        pull_seconds = sum(group.pull_seconds for group in groups)
        acknowledge_seconds = sum(group.acknowledge_seconds for group in groups)
        ownership_filter = published_bytes / pulled_bytes if pulled_bytes else float("inf")
        frame_inflation = pulled_uncompressed_bytes / routed_bytes if routed_bytes else float("inf")
        route_filter = published_uncompressed_bytes / routed_bytes if routed_bytes else float("inf")
        logger.info(
            "NIXL XOR rank %d policy v%d metrics: groups=%d, frames=%d/%d, "
            "compressed=%.2f/%.2f MiB, ownership_filter=%.2fx, wait=%.3fs, pull=%.3fs, "
            "source=%.2f/%.2f/%.2f MiB, frame_inflation=%.2fx, route_filter=%.2fx, "
            "acknowledge=%.3fs, decode=%.3fs, apply=%.3fs, final_sync=%.3fs, pipeline=%.3fs, "
            "receive_arenas=%.2f MiB, peak_allocated=%.2f GiB",
            self.model_express.rank,
            manifest.step,
            len(groups),
            pulled_frames,
            published_frames,
            pulled_bytes / 2**20,
            published_bytes / 2**20,
            ownership_filter,
            wait_seconds,
            pull_seconds,
            routed_bytes / 2**20,
            pulled_uncompressed_bytes / 2**20,
            published_uncompressed_bytes / 2**20,
            frame_inflation,
            route_filter,
            acknowledge_seconds,
            decode_seconds,
            apply_seconds,
            synchronization_seconds,
            pipeline_seconds,
            sum(arena.numel() for arena in self.delta_receive_arenas) / 2**20,
            torch.cuda.max_memory_allocated(self.device) / 2**30,
        )
        for group_index, group in enumerate(groups):
            logger.debug(
                "NIXL XOR rank %d policy v%d group %d metrics: frames=%d/%d, "
                "compressed=%.2f/%.2f MiB, wait=%.3fs, pull=%.3fs, acknowledge=%.3fs",
                self.model_express.rank,
                manifest.step,
                group_index,
                group.pulled_frames,
                group.published_frames,
                group.pulled_bytes / 2**20,
                group.published_bytes / 2**20,
                group.wait_seconds,
                group.pull_seconds,
                group.acknowledge_seconds,
            )

    def prepare_delta_peer(self, agent_name: str, metadata: bytes) -> str:
        if self.delta_peer_metadata.get(agent_name) == metadata:
            return self.delta_peer_names[agent_name]
        peer_name = self.nixl_agent.add_remote_agent(metadata)
        self.nixl_agent.make_connection(peer_name)
        self.delta_peer_metadata[agent_name] = metadata
        self.delta_peer_names[agent_name] = peer_name
        return peer_name

    def decode_delta_frames(
        self,
        pulled: list[tuple[NIXLDeltaFrame, torch.Tensor]],
    ) -> dict[tuple[int, str], tuple[torch.Tensor, DeltaTensorMetadata]]:
        if not pulled:
            return {}
        assert self.delta_codec is not None
        decoded_frames = self.delta_codec.decode(
            [payload for _frame, payload in pulled],
            [frame.uncompressed_nbytes for frame, _payload in pulled],
        )
        decoded: dict[tuple[int, str], tuple[torch.Tensor, DeltaTensorMetadata]] = {}
        for (frame, _payload), raw in zip(pulled, decoded_frames, strict=True):
            metadata = tuple(tensor.to_metadata() for tensor in frame.tensors)
            values = unpack_delta_frame(
                raw,
                metadata,
                CompressedDeltaFrame(
                    first_tensor_index=0,
                    tensor_count=len(metadata),
                    uncompressed_nbytes=frame.uncompressed_nbytes,
                    compressed_nbytes=frame.compressed_nbytes,
                ),
            )
            for tensor_metadata, (name, value) in zip(metadata, values, strict=True):
                key = (frame.agent, name)
                if key in decoded:
                    raise RuntimeError(f"NIXL XOR group contains duplicate source shard {key}")
                decoded[key] = (value, tensor_metadata)
        return decoded

    @staticmethod
    def apply_decoded_delta_group(
        transfer_group: WeightTransferGroup,
        decoded: dict[tuple[int, str], tuple[torch.Tensor, DeltaTensorMetadata]],
    ) -> None:
        for layer in transfer_group.layers:
            for copy_plan in (*layer.copies, *layer.persistent_copies):
                routes = copy_plan.routes
                present = [(route.agent, copy_plan.recorded_copy.source_name) in decoded for route in routes]
                if not any(present):
                    continue
                if not all(present):
                    raise RuntimeError(
                        f"NIXL XOR update has incomplete shards for {copy_plan.recorded_copy.source_name!r}"
                    )
                NIXLWeightUpdateWorker.populate_delta_staging(copy_plan, decoded)
                copy = copy_plan.recorded_copy
                parameter = getattr(copy.destination_module, copy.destination_name)
                destination = parameter.as_strided(
                    copy.destination_shape,
                    copy.destination_stride,
                    copy.destination_offset,
                )
                value = apply_chain(copy_plan.staging_tensor, copy_plan.replay_ops)
                if value.dtype != destination.dtype or tuple(value.shape) != tuple(destination.shape):
                    raise RuntimeError(
                        f"NIXL XOR route for {copy.source_name!r} changes representation: "
                        f"{value.dtype}{tuple(value.shape)} -> {destination.dtype}{tuple(destination.shape)}"
                    )
                if destination.is_contiguous() and value.is_contiguous():
                    integer_view(destination).bitwise_xor_(integer_view(value))
                else:
                    updated = destination.contiguous()
                    integer_view(updated).bitwise_xor_(integer_view(value.contiguous()))
                    destination.copy_(updated)

    @staticmethod
    def populate_delta_staging(
        copy_plan: TensorCopyPlan,
        decoded: dict[tuple[int, str], tuple[torch.Tensor, DeltaTensorMetadata]],
    ) -> None:
        source = copy_plan.source_tensor
        destination_bytes = copy_plan.staging_tensor.view(torch.uint8).reshape(-1)
        routes = copy_plan.routes
        for route in routes:
            shard = next(
                shard
                for shard in source.shards
                if shard.agent == route.agent
                and shard.addr <= route.source_addr < shard.addr + shard.numel * copy_plan.staging_tensor.element_size()
            )
            try:
                value, metadata = decoded[(route.agent, copy_plan.recorded_copy.source_name)]
            except KeyError as error:
                raise RuntimeError(
                    f"missing decoded NIXL XOR shard for agent {route.agent}, "
                    f"tensor {copy_plan.recorded_copy.source_name!r}"
                ) from error
            if metadata.dtype != str(copy_plan.staging_tensor.dtype).removeprefix("torch."):
                raise RuntimeError(
                    f"NIXL XOR source dtype {metadata.dtype} does not match wire dtype "
                    f"{copy_plan.staging_tensor.dtype} for {metadata.name!r}"
                )
            if metadata.resolved_global_shape != tuple(source.shape):
                raise RuntimeError(
                    f"NIXL XOR source shape {metadata.resolved_global_shape} does not match "
                    f"the transfer table shape {tuple(source.shape)} for {metadata.name!r}"
                )
            expected_shard_bytes = shard.numel * copy_plan.staging_tensor.element_size()
            if metadata.nbytes != expected_shard_bytes:
                raise RuntimeError(
                    f"NIXL XOR shard for {metadata.name!r} has {metadata.nbytes} bytes; "
                    f"the transfer table requires {expected_shard_bytes}"
                )
            source_offset = route.source_addr - shard.addr
            destination_offset = route.destination_addr - copy_plan.staging_tensor.data_ptr()
            source_bytes = value.view(torch.uint8).reshape(-1)
            destination_bytes.narrow(0, destination_offset, route.nbytes).copy_(
                source_bytes.narrow(0, source_offset, route.nbytes)
            )

    def apply_transfer_plan(self, plan: WeightTransferPlan) -> None:
        from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _copy_and_restore_kernel_tensors,
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )
        from vllm.model_executor.model_loader.reload.meta import materialize_layer
        from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

        model = self.raw_model
        cancelled = Event()
        group_metrics: list[FullGroupMetrics] = []
        replay_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        pipeline_started = time.perf_counter()

        def pull_group(group_index: int) -> tuple[WeightTransferGroup, FullGroupMetrics]:
            transfer_group = plan.groups[group_index]
            session = self.buffer_sessions[group_index % len(self.buffer_sessions)]
            wait_started = time.perf_counter()
            session.wait_for(
                "trainer",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_READY,
                timeout=self.weight_transfer_timeout,
                poll_interval=_BUFFER_POLL_INTERVAL,
                cancelled=cancelled.is_set,
            )
            wait_seconds = time.perf_counter() - wait_started

            pull_started = time.perf_counter()
            for local, remote, indices in transfer_group.pulls:
                handle = self.nixl_agent.post_read(local, indices, remote)
                self.nixl_agent.wait(
                    handle,
                    context=f"weight pull for {transfer_group.name}",
                    timeout=self.weight_transfer_timeout,
                    cancelled=cancelled.is_set,
                )
            return transfer_group, FullGroupMetrics(
                wait_seconds=wait_seconds,
                pull_seconds=time.perf_counter() - pull_started,
            )

        def acknowledge_group(group_index: int) -> float:
            started = time.perf_counter()
            session = self.buffer_sessions[group_index % len(self.buffer_sessions)]
            session.set_status(p2p_pb2.SOURCE_STATUS_READY)
            session.wait_for(
                "trainer",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.weight_transfer_timeout,
                poll_interval=_BUFFER_POLL_INTERVAL,
                cancelled=cancelled.is_set,
            )
            session.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
            return time.perf_counter() - started

        def prefetch_group(group_index: int) -> tuple[WeightTransferGroup, FullGroupMetrics]:
            torch.cuda.set_device(self.device)
            transfer_group, metrics = pull_group(group_index)
            metrics.acknowledge_seconds = acknowledge_group(group_index)
            return transfer_group, metrics

        def replay_group(transfer_group: WeightTransferGroup) -> None:
            for layer_plan in transfer_group.layers:
                layer = layer_plan.reload_layer
                if layer is None:
                    for copy_plan in layer_plan.persistent_copies:
                        self.replay_tensor_copy(copy_plan)
                    continue

                info = LAYERWISE_INFO[layer]
                materialize_layer(layer, info)
                # Match loader semantics for destinations with unwritten padding.
                destination_names = layer_plan.destination_names
                for name, tensor in get_layer_tensors(layer).items():
                    if name in destination_names and not tensor.is_meta:
                        tensor.zero_()
                for copy_plan in layer_plan.copies:
                    self.replay_tensor_copy(copy_plan)
                for copy_plan in layer_plan.persistent_copies:
                    self.replay_tensor_copy(copy_plan)

                if hasattr(layer, "_already_called_process_weights_after_loading"):
                    delattr(layer, "_already_called_process_weights_after_loading")
                quant_method = getattr(layer, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    quant_method.process_weights_after_loading(layer)
                if info.kernel_tensors is not None:
                    _copy_and_restore_kernel_tensors(layer, info)
                info.reset()

        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(model)
            pipelined = plan.receive_buffer_count > 1
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nixl-prefetch") if pipelined else None
            try:
                pull = executor.submit(prefetch_group, 0) if executor is not None else None
                for group_index in range(len(plan.groups)):
                    transfer_group, metrics = pull.result() if pull is not None else pull_group(group_index)

                    torch.cuda.synchronize(self.device)
                    if executor is not None and group_index + 1 < len(plan.groups):
                        pull = executor.submit(prefetch_group, group_index + 1)

                    replay_started = torch.cuda.Event(enable_timing=True)
                    replay_finished = torch.cuda.Event(enable_timing=True)
                    replay_started.record(torch.cuda.current_stream(self.device))
                    replay_group(transfer_group)
                    replay_finished.record(torch.cuda.current_stream(self.device))
                    torch.cuda.synchronize(self.device)
                    replay_events.append((replay_started, replay_finished))

                    if not pipelined:
                        metrics.acknowledge_seconds = acknowledge_group(group_index)
                    group_metrics.append(metrics)
            finally:
                cancelled.set()
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

            finalize_layerwise_reload(model, self.model_runner.model_config)
        self.log_full_metrics(
            plan,
            group_metrics,
            replay_seconds=sum(start.elapsed_time(end) / 1000 for start, end in replay_events),
            pipeline_seconds=time.perf_counter() - pipeline_started,
        )

    def log_full_metrics(
        self,
        plan: WeightTransferPlan,
        groups: list[FullGroupMetrics],
        *,
        replay_seconds: float,
        pipeline_seconds: float,
    ) -> None:
        published_bytes = sum(
            prod(tensor.shape) * getattr(torch, tensor.wire_dtype).itemsize
            for group in plan.table.groups
            for tensor in group.tensors
        )
        pulled_bytes = sum(group.pull_nbytes for group in plan.groups)
        ownership_filter = published_bytes / pulled_bytes if pulled_bytes else float("inf")
        logger.info(
            "NIXL full rank %d metrics: groups=%d, bytes=%.2f/%.2f MiB, ownership_filter=%.2fx, "
            "wait=%.3fs, pull=%.3fs, acknowledge=%.3fs, replay=%.3fs, pipeline=%.3fs, "
            "receive_arenas=%.2f MiB, peak_allocated=%.2f GiB",
            self.model_express.rank,
            len(groups),
            pulled_bytes / 2**20,
            published_bytes / 2**20,
            ownership_filter,
            sum(group.wait_seconds for group in groups),
            sum(group.pull_seconds for group in groups),
            sum(group.acknowledge_seconds for group in groups),
            replay_seconds,
            pipeline_seconds,
            sum(arena.numel() * arena.element_size() for arena in plan.receive_arenas.values()) / 2**20,
            torch.cuda.max_memory_allocated(self.device) / 2**30,
        )

    @staticmethod
    def replay_tensor_copy(plan: TensorCopyPlan) -> None:
        copy = plan.recorded_copy
        parameter = getattr(copy.destination_module, copy.destination_name)
        destination = parameter.as_strided(
            copy.destination_shape,
            copy.destination_stride,
            copy.destination_offset,
        )
        value = apply_chain(plan.staging_tensor, plan.replay_ops)
        destination.copy_(value)
