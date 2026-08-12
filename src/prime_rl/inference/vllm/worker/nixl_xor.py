"""XOR-compressed NIXL weight-update worker for vLLM."""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event
from typing import Any

import torch
from modelexpress import p2p_pb2
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.nixl import (
    NIXLWeightUpdateWorker,
    TensorCopyPlan,
    WeightTransferGroup,
    WeightTransferPlan,
)
from prime_rl.inference.vllm.worker.weight_transfer import update_mla_absorbed_weights
from prime_rl.trainer.rl.broadcast.nixl.agent import MemDesc
from prime_rl.trainer.rl.broadcast.nixl.cuda_malloc_memory import use_cuda_malloc_pool
from prime_rl.trainer.rl.broadcast.nixl.delta_manifest import (
    NIXLDeltaFrame,
    NIXLDeltaManifest,
    NIXLPolicyMetadata,
)
from prime_rl.trainer.rl.broadcast.nixl.graph import apply_chain, chain_preserves_dtype
from prime_rl.weight_sync.fp8_resident import parse_resident_tensor_name
from prime_rl.weight_sync.xor_delta import (
    NVCOMP_FRAME_ALIGNMENT,
    CompressedDeltaFrame,
    DeltaTensorMetadata,
    NvcompLZ4Codec,
    align_nvcomp_nbytes,
    integer_view,
    unpack_delta_frame,
)

logger = init_logger("vllm.inference.vllm.worker_nixl_xor")


@dataclass(frozen=True)
class DeltaSourceSpec:
    key: tuple[int, str]
    dtype: str
    global_shape: tuple[int, ...]
    nbytes: int


@dataclass(frozen=True)
class DeltaRouteBinding:
    source: DeltaSourceSpec
    source_offset: int
    staging_offset: int
    nbytes: int


@dataclass
class DeltaCopyPlan:
    routes: tuple[DeltaRouteBinding, ...]
    required_sources: frozenset[tuple[int, str]]
    direct_xor: bool = False


@dataclass(frozen=True)
class DeltaTransferGroupPlan:
    required_sources: frozenset[tuple[int, str]]
    required_nbytes: int
    source_specs: dict[tuple[int, str], DeltaSourceSpec]


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
    graph_captures: int = 0
    graph_replays: int = 0
    graphed_routes: int = 0
    eager_routes: int = 0


@dataclass(frozen=True)
class DeltaXorGraph:
    graph: torch.cuda.CUDAGraph
    signature: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class DeltaXorGraphMetrics:
    captures: int = 0
    replays: int = 0
    graphed_routes: int = 0
    eager_routes: int = 0

    def __add__(self, other: DeltaXorGraphMetrics) -> DeltaXorGraphMetrics:
        return DeltaXorGraphMetrics(
            captures=self.captures + other.captures,
            replays=self.replays + other.replays,
            graphed_routes=self.graphed_routes + other.graphed_routes,
            eager_routes=self.eager_routes + other.eager_routes,
        )


@dataclass
class PreparedDeltaGroup:
    group_index: int
    decode_slot: int
    transfer_group: WeightTransferGroup
    decoded: dict[tuple[int, str], tuple[torch.Tensor, DeltaTensorMetadata]]
    decode_started: torch.cuda.Event
    decode_finished: torch.cuda.Event
    metrics: DeltaGroupMetrics


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


class NIXLXorWeightUpdateWorker(NIXLWeightUpdateWorker):
    def init_broadcaster(
        self,
        host: str,
        port: int,
        rank_offset: int,
        inference_world_size: int,
        timeout: int,
        quantize_in_weight_transfer: bool = False,
        session_id: str = "default",
        delta_mode: str = "xor",
        delta_representation: str = "source",
        delta_fp8_scale_format: str = "float32",
        delta_cuda_graphs: bool = True,
    ) -> None:
        if delta_mode != "xor":
            raise ValueError(f"NIXLXorWeightUpdateWorker requires delta_mode='xor', got {delta_mode!r}")
        super().init_broadcaster(
            host,
            port,
            rank_offset,
            inference_world_size,
            timeout,
            quantize_in_weight_transfer,
            session_id,
            delta_mode,
            delta_representation,
            delta_fp8_scale_format,
            delta_cuda_graphs,
        )
        self.delta_mode = delta_mode
        self.delta_cuda_graphs = delta_cuda_graphs
        self.current_step: int | None = None
        self.full_sync_required = True
        self.delta_codec = NvcompLZ4Codec(self.device)
        self.delta_peer_metadata: dict[str, bytes] = {}
        self.delta_peer_names: dict[str, str] = {}
        self.delta_receive_arenas: list[torch.Tensor] = []
        self.delta_receive_registrations: list[Any] = []
        self.delta_receive_slot_bytes = 0
        self.delta_decode_arenas: list[torch.Tensor] = []
        self.delta_decode_slot_bytes = 0
        self.delta_prefetch_stream = torch.cuda.Stream(device=self.device)
        self.delta_apply_stream = torch.cuda.Stream(device=self.device)
        self.delta_xor_graphs: dict[tuple[int, int, int], DeltaXorGraph] = {}
        self.delta_xor_graph_warmup: torch.Tensor | None = None
        self.delta_xor_graphs_invalidated = False
        self.fp8_resident_tensors: dict[str, torch.Tensor] | None = None
        self.delta_copy_plans: dict[int, DeltaCopyPlan] = {}
        self.delta_group_plans: dict[int, DeltaTransferGroupPlan] = {}

    def validate_transfer_plan(self, plan: WeightTransferPlan) -> None:
        self.build_delta_transfer_plan(plan)
        self.validate_delta_transfer_plan(plan)

    def build_delta_transfer_plan(self, plan: WeightTransferPlan) -> None:
        for group in plan.groups:
            source_ranges: dict[tuple[int, str], list[tuple[int, int]]] = {}
            source_specs: dict[tuple[int, str], DeltaSourceSpec] = {}
            required_sources: set[tuple[int, str]] = set()
            for layer in group.layers:
                for copy_plan in (*layer.copies, *layer.persistent_copies):
                    bindings = self.bind_delta_routes(copy_plan)
                    delta_copy = DeltaCopyPlan(
                        routes=bindings,
                        required_sources=frozenset(binding.source.key for binding in bindings),
                    )
                    self.delta_copy_plans[id(copy_plan)] = delta_copy
                    required_sources.update(delta_copy.required_sources)
                    for binding in bindings:
                        previous = source_specs.setdefault(binding.source.key, binding.source)
                        if previous != binding.source:
                            raise RuntimeError(
                                f"inconsistent NIXL XOR source metadata for {binding.source.key}"
                            )
                    for route in copy_plan.routes:
                        key = (route.agent, copy_plan.recorded_copy.source_name)
                        source_ranges.setdefault(key, []).append(
                            (route.source_addr, route.source_addr + route.nbytes)
                        )
            self.delta_group_plans[id(group)] = DeltaTransferGroupPlan(
                required_sources=frozenset(required_sources),
                required_nbytes=sum(_covered_nbytes(ranges) for ranges in source_ranges.values()),
                source_specs=source_specs,
            )

    @staticmethod
    def bind_delta_routes(copy_plan: TensorCopyPlan) -> tuple[DeltaRouteBinding, ...]:
        copy = copy_plan.recorded_copy
        source = copy_plan.source_tensor
        staging_tensor = copy_plan.staging_tensor
        itemsize = staging_tensor.element_size()
        bindings: list[DeltaRouteBinding] = []
        for route in copy_plan.routes:
            shard = next(
                (
                    candidate
                    for candidate in source.shards
                    if candidate.agent == route.agent
                    and candidate.addr <= route.source_addr < candidate.addr + candidate.numel * itemsize
                ),
                None,
            )
            if shard is None:
                raise RuntimeError(
                    f"NIXL XOR route for {copy.source_name!r} does not resolve to trainer agent {route.agent}"
                )
            source_offset = route.source_addr - shard.addr
            staging_offset = route.destination_addr - staging_tensor.data_ptr()
            source_nbytes = shard.numel * itemsize
            staging_nbytes = staging_tensor.numel() * itemsize
            if source_offset < 0 or source_offset + route.nbytes > source_nbytes:
                raise RuntimeError(f"NIXL XOR route exceeds source shard bounds for {copy.source_name!r}")
            if staging_offset < 0 or staging_offset + route.nbytes > staging_nbytes:
                raise RuntimeError(f"NIXL XOR route exceeds staging bounds for {copy.source_name!r}")
            bindings.append(
                DeltaRouteBinding(
                    source=DeltaSourceSpec(
                        key=(route.agent, copy.source_name),
                        dtype=source.wire_dtype,
                        global_shape=tuple(source.shape),
                        nbytes=source_nbytes,
                    ),
                    source_offset=source_offset,
                    staging_offset=staging_offset,
                    nbytes=route.nbytes,
                )
            )
        return tuple(bindings)

    def validate_delta_transfer_plan(self, plan: WeightTransferPlan) -> None:
        if plan.table.representation == "fp8_kernel":
            logger.info("NIXL FP8 XOR will target TP-local resident tensors directly")
            return
        direct_bytes = 0
        routed_bytes = 0
        direct_copies = 0
        copy_count = 0
        for group in plan.groups:
            for layer in group.layers:
                for copy_plan in (*layer.copies, *layer.persistent_copies):
                    delta_copy = self.delta_copy_plans[id(copy_plan)]
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
                    delta_copy.direct_xor = not copy_plan.replay_ops and destination.is_contiguous()
                    nbytes = destination.numel() * destination.element_size()
                    routed_bytes += nbytes
                    copy_count += 1
                    if delta_copy.direct_xor:
                        direct_bytes += nbytes
                        direct_copies += 1
        logger.info(
            "NIXL XOR direct route coverage: %.2f/%.2f MiB (%.1f%%), copies=%d/%d",
            direct_bytes / 2**20,
            routed_bytes / 2**20,
            100 * direct_bytes / routed_bytes if routed_bytes else 100.0,
            direct_copies,
            copy_count,
        )

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
                self.apply_transfer_plan(plan, step=policy.step)
            update_mla_absorbed_weights(self.raw_model)
            torch.cuda.synchronize(self.device)
        except BaseException:
            self.delta_xor_graphs.clear()
            self.delta_xor_graphs_invalidated = False
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

    def apply_transfer_plan(self, plan: WeightTransferPlan, *, step: int) -> None:
        if plan.table.representation == "fp8_kernel":
            self.fp8_resident_tensors = None
        super().apply_transfer_plan(plan, step=step)

    def apply_delta_manifest(self, plan: WeightTransferPlan, manifest: NIXLDeltaManifest) -> None:
        if manifest.base_step != self.current_step:
            raise RuntimeError(
                f"NIXL delta manifest base {manifest.base_step} does not match resident policy {self.current_step}"
            )
        if tuple(group.name for group in manifest.groups) != tuple(group.name for group in plan.groups):
            raise RuntimeError("NIXL delta manifest groups do not match the full-transfer plan")
        if (
            manifest.representation != plan.table.representation
            or manifest.fp8_scale_format != plan.table.fp8_scale_format
        ):
            raise RuntimeError("NIXL delta representation does not match the full-transfer base")

        selected_by_group = [
            self.select_resident_delta_frames(delta_group.frames)
            if plan.table.representation == "fp8_kernel"
            else self.select_delta_frames(transfer_group, delta_group.frames)
            for transfer_group, delta_group in zip(plan.groups, manifest.groups, strict=True)
        ]
        receive_buffer_count = min(2, len(plan.groups))
        slot_bytes = max((self.packed_delta_frame_bytes(frames) for frames in selected_by_group), default=0)
        decode_slot_bytes = max(
            (self.packed_delta_decoded_frame_bytes(frames) for frames in selected_by_group),
            default=0,
        )
        self.ensure_delta_receive_arenas(slot_bytes, receive_buffer_count)
        self.ensure_delta_decode_arenas(decode_slot_bytes, receive_buffer_count)
        capture_graphs = self.delta_cuda_graphs and not self.delta_xor_graphs
        if capture_graphs:
            self.warm_delta_xor_graphs()
        receive_slot_events: list[torch.cuda.Event | None] = [None] * receive_buffer_count
        decode_slot_events: list[torch.cuda.Event | None] = [None] * receive_buffer_count
        decode_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        apply_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        group_metrics: list[DeltaGroupMetrics] = []
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
                receive_slot_events,
                decode_slot_events,
                cancelled,
            )
            for group_index in range(len(plan.groups)):
                prepared = future.result()
                if not capture_graphs and group_index + 1 < len(plan.groups):
                    future = executor.submit(
                        self.prepare_delta_group,
                        group_index + 1,
                        plan.groups[group_index + 1],
                        manifest.groups[group_index + 1].frames,
                        selected_by_group[group_index + 1],
                        manifest,
                        receive_slot_events,
                        decode_slot_events,
                        cancelled,
                    )

                apply_started = torch.cuda.Event(enable_timing=True)
                apply_finished = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(self.delta_apply_stream):
                    self.delta_apply_stream.wait_event(prepared.decode_finished)
                    apply_started.record(self.delta_apply_stream)
                    graph_metrics = self.apply_decoded_delta_group(
                        prepared.transfer_group,
                        prepared.decoded,
                        group_index=prepared.group_index,
                        decode_slot=prepared.decode_slot,
                        allow_capture=capture_graphs,
                    )
                    apply_finished.record(self.delta_apply_stream)
                prepared.metrics.graph_captures = graph_metrics.captures
                prepared.metrics.graph_replays = graph_metrics.replays
                prepared.metrics.graphed_routes = graph_metrics.graphed_routes
                prepared.metrics.eager_routes = graph_metrics.eager_routes
                decode_slot_events[prepared.decode_slot] = apply_finished
                decode_events.append((prepared.decode_started, prepared.decode_finished))
                apply_events.append((apply_started, apply_finished))
                group_metrics.append(prepared.metrics)
                if capture_graphs and group_index + 1 < len(plan.groups):
                    future = executor.submit(
                        self.prepare_delta_group,
                        group_index + 1,
                        plan.groups[group_index + 1],
                        manifest.groups[group_index + 1].frames,
                        selected_by_group[group_index + 1],
                        manifest,
                        receive_slot_events,
                        decode_slot_events,
                        cancelled,
                    )
        finally:
            cancelled.set()
            executor.shutdown(wait=True, cancel_futures=True)

        synchronization_started = time.perf_counter()
        torch.cuda.synchronize(self.device)
        synchronization_seconds = time.perf_counter() - synchronization_started
        pipeline_seconds = time.perf_counter() - pipeline_started
        decode_seconds = sum(
            prepared_start.elapsed_time(prepared_end) / 1000 for prepared_start, prepared_end in decode_events
        )
        apply_seconds = sum(start.elapsed_time(end) / 1000 for start, end in apply_events)
        if self.delta_xor_graphs_invalidated:
            logger.warning("NIXL XOR CUDA graph route addresses changed; recapturing on the next delta update")
            self.delta_xor_graphs.clear()
            self.delta_xor_graphs_invalidated = False
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
        receive_slot_events: list[torch.cuda.Event | None],
        decode_slot_events: list[torch.cuda.Event | None],
        cancelled: Event,
    ) -> PreparedDeltaGroup:
        torch.cuda.set_device(self.device)
        slot = group_index % len(self.delta_receive_arenas)
        previous_receive_event = receive_slot_events[slot]
        if previous_receive_event is not None:
            previous_receive_event.synchronize()

        wait_started = time.perf_counter()
        self.wait_for_group_ready(
            step=manifest.step,
            group_index=group_index,
            cancelled=cancelled,
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

        acknowledgement_started = time.perf_counter()
        self.acknowledge_group(step=manifest.step, group_index=group_index)
        acknowledgement_seconds = time.perf_counter() - acknowledgement_started

        decode_started = torch.cuda.Event(enable_timing=True)
        decode_finished = torch.cuda.Event(enable_timing=True)
        previous_decode_event = decode_slot_events[slot]
        if previous_decode_event is not None:
            previous_decode_event.synchronize()
        with torch.cuda.stream(self.delta_prefetch_stream):
            decode_started.record(self.delta_prefetch_stream)
            decoded = self.decode_delta_frames(pulled, decode_slot=slot)
            decode_finished.record(self.delta_prefetch_stream)
        receive_slot_events[slot] = decode_finished

        return PreparedDeltaGroup(
            group_index=group_index,
            decode_slot=slot,
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
                routed_bytes=self.delta_group_plans[id(transfer_group)].required_nbytes,
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

    def ensure_delta_decode_arenas(self, required_bytes: int, count: int) -> None:
        required_bytes = max(NVCOMP_FRAME_ALIGNMENT, align_nvcomp_nbytes(required_bytes))
        if len(self.delta_decode_arenas) == count and self.delta_decode_slot_bytes >= required_bytes:
            return
        self.delta_xor_graphs.clear()
        self.delta_decode_slot_bytes = required_bytes
        with use_cuda_malloc_pool():
            self.delta_decode_arenas = [
                torch.empty(required_bytes, dtype=torch.uint8, device=self.device) for _ in range(count)
            ]

    def select_delta_frames(
        self,
        transfer_group: WeightTransferGroup,
        frames: tuple[NIXLDeltaFrame, ...],
    ) -> list[NIXLDeltaFrame]:
        required = self.delta_group_plans[id(transfer_group)].required_sources
        return [frame for frame in frames if any((frame.agent, tensor.name) in required for tensor in frame.tensors)]

    def select_resident_delta_frames(self, frames: tuple[NIXLDeltaFrame, ...]) -> list[NIXLDeltaFrame]:
        inference_rank = self.model_express.rank
        selected: list[NIXLDeltaFrame] = []
        for frame in frames:
            targets = {parse_resident_tensor_name(tensor.name)[0] for tensor in frame.tensors}
            if targets == {inference_rank}:
                selected.append(frame)
            elif inference_rank in targets:
                raise RuntimeError("an FP8 resident delta frame spans multiple inference ranks")
        return selected

    @staticmethod
    def packed_delta_frame_bytes(frames: list[NIXLDeltaFrame]) -> int:
        used = 0
        for frame in frames:
            used = align_nvcomp_nbytes(used)
            used += frame.compressed_nbytes
        return align_nvcomp_nbytes(used)

    @staticmethod
    def packed_delta_decoded_frame_bytes(frames: list[NIXLDeltaFrame]) -> int:
        used = 0
        for frame in frames:
            used = align_nvcomp_nbytes(used)
            used += frame.uncompressed_nbytes
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
        graph_captures = sum(group.graph_captures for group in groups)
        graph_replays = sum(group.graph_replays for group in groups)
        graphed_routes = sum(group.graphed_routes for group in groups)
        eager_routes = sum(group.eager_routes for group in groups)
        ownership_filter = published_bytes / pulled_bytes if pulled_bytes else float("inf")
        frame_inflation = pulled_uncompressed_bytes / routed_bytes if routed_bytes else float("inf")
        route_filter = published_uncompressed_bytes / routed_bytes if routed_bytes else float("inf")
        logger.info(
            "NIXL XOR rank %d policy v%d metrics: groups=%d, frames=%d/%d, "
            "compressed=%.2f/%.2f MiB, ownership_filter=%.2fx, wait=%.3fs, pull=%.3fs, "
            "source=%.2f/%.2f/%.2f MiB, frame_inflation=%.2fx, route_filter=%.2fx, "
            "acknowledge=%.3fs, decode=%.3fs, apply=%.3fs, final_sync=%.3fs, pipeline=%.3fs, "
            "graphs=%d/%d, routes=%d/%d, receive_arenas=%.2f MiB, decode_arenas=%.2f MiB, "
            "peak_allocated=%.2f GiB",
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
            graph_captures,
            graph_replays,
            graphed_routes,
            eager_routes,
            sum(arena.numel() for arena in self.delta_receive_arenas) / 2**20,
            sum(arena.numel() for arena in self.delta_decode_arenas) / 2**20,
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
        *,
        decode_slot: int,
    ) -> dict[tuple[int, str], tuple[torch.Tensor, DeltaTensorMetadata]]:
        if not pulled:
            return {}
        assert self.delta_codec is not None
        arena = self.delta_decode_arenas[decode_slot]
        decoded_frames: list[torch.Tensor] = []
        used = 0
        for frame, _payload in pulled:
            used = align_nvcomp_nbytes(used)
            decoded_frames.append(arena.narrow(0, used, frame.uncompressed_nbytes))
            used += frame.uncompressed_nbytes
        self.delta_codec.decode_into(
            [payload for _frame, payload in pulled],
            decoded_frames,
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

    def warm_delta_xor_graphs(self) -> None:
        if self.delta_xor_graph_warmup is not None:
            return
        self.delta_xor_graph_warmup = torch.zeros(1, dtype=torch.uint8, device=self.device)
        with torch.cuda.stream(self.delta_apply_stream):
            self.delta_xor_graph_warmup.bitwise_xor_(self.delta_xor_graph_warmup)
        self.delta_apply_stream.synchronize()

    def apply_decoded_delta_group(
        self,
        transfer_group: WeightTransferGroup,
        decoded: dict[tuple[int, str], tuple[torch.Tensor, DeltaTensorMetadata]],
        *,
        group_index: int,
        decode_slot: int,
        allow_capture: bool,
    ) -> DeltaXorGraphMetrics:
        if self.weight_transfer_plan is not None and self.weight_transfer_plan.table.representation == "fp8_kernel":
            return self.apply_fp8_resident_delta_group(
                decoded,
                group_index=group_index,
                decode_slot=decode_slot,
                allow_capture=allow_capture,
            )
        self.validate_decoded_delta_sources(transfer_group, decoded)
        operations: list[tuple[TensorCopyPlan, DeltaCopyPlan, torch.Tensor]] = []
        for layer in transfer_group.layers:
            for copy_plan in (*layer.copies, *layer.persistent_copies):
                delta_copy = self.delta_copy_plans[id(copy_plan)]
                required = delta_copy.required_sources
                present = required.intersection(decoded)
                if not present:
                    continue
                if present != required:
                    raise RuntimeError(
                        f"NIXL XOR update has incomplete shards for {copy_plan.recorded_copy.source_name!r}"
                    )
                copy = copy_plan.recorded_copy
                parameter = getattr(copy.destination_module, copy.destination_name)
                destination = parameter.as_strided(
                    copy.destination_shape,
                    copy.destination_stride,
                    copy.destination_offset,
                )
                operations.append((copy_plan, delta_copy, destination))

        metrics = DeltaXorGraphMetrics()
        direct_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        segment_index = 0
        for copy_plan, delta_copy, destination in operations:
            if delta_copy.direct_xor:
                direct_pairs.extend(self.direct_delta_route_pairs(delta_copy, destination, decoded))
                continue
            metrics += self.apply_delta_xor_graph_segment(
                direct_pairs,
                group_index=group_index,
                decode_slot=decode_slot,
                segment_index=segment_index,
                allow_capture=allow_capture,
            )
            direct_pairs.clear()
            segment_index += 1
            self.apply_replayed_delta(copy_plan, destination, decoded)
            metrics += DeltaXorGraphMetrics(eager_routes=len(delta_copy.routes))
        metrics += self.apply_delta_xor_graph_segment(
            direct_pairs,
            group_index=group_index,
            decode_slot=decode_slot,
            segment_index=segment_index,
            allow_capture=allow_capture,
        )
        return metrics

    def apply_fp8_resident_delta_group(
        self,
        decoded: dict[tuple[int, str], tuple[torch.Tensor, DeltaTensorMetadata]],
        *,
        group_index: int,
        decode_slot: int,
        allow_capture: bool,
    ) -> DeltaXorGraphMetrics:
        resident = self.get_fp8_resident_tensors()
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _key, (delta, metadata) in decoded.items():
            inference_rank, name = parse_resident_tensor_name(metadata.name)
            if inference_rank != self.model_express.rank:
                raise RuntimeError(
                    f"received resident FP8 delta for inference rank {inference_rank} on rank {self.model_express.rank}"
                )
            destination = resident.get(name)
            if destination is None:
                raise RuntimeError(f"vLLM resident tensor {name!r} was not found")
            if destination.dtype != delta.dtype or tuple(destination.shape) != tuple(delta.shape):
                raise RuntimeError(
                    f"resident FP8 delta {metadata.name!r} has {delta.dtype}{tuple(delta.shape)}; "
                    f"vLLM has {destination.dtype}{tuple(destination.shape)}"
                )
            pairs.append((destination.view(torch.uint8).reshape(-1), delta.view(torch.uint8).reshape(-1)))
        return self.apply_delta_xor_graph_segment(
            pairs,
            group_index=group_index,
            decode_slot=decode_slot,
            segment_index=0,
            allow_capture=allow_capture,
        )

    def get_fp8_resident_tensors(self) -> dict[str, torch.Tensor]:
        if self.fp8_resident_tensors is not None:
            return self.fp8_resident_tensors
        from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

        tensors: dict[str, torch.Tensor] = {}
        for module_name, module in self.raw_model.named_modules():
            for name, tensor in get_layer_tensors(module).items():
                if tensor.is_meta:
                    continue
                full_name = f"{module_name}.{name}" if module_name else name
                previous = tensors.setdefault(full_name, tensor)
                if previous is not tensor:
                    raise RuntimeError(f"duplicate vLLM resident tensor name {full_name!r}")
        self.fp8_resident_tensors = tensors
        logger.info("Indexed %d TP-local vLLM resident tensors for direct FP8 XOR", len(tensors))
        return tensors

    def apply_delta_xor_graph_segment(
        self,
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        group_index: int,
        decode_slot: int,
        segment_index: int,
        allow_capture: bool,
    ) -> DeltaXorGraphMetrics:
        if not pairs:
            return DeltaXorGraphMetrics()
        signature = tuple(
            (destination.data_ptr(), source.data_ptr(), destination.numel()) for destination, source in pairs
        )
        key = (group_index, decode_slot, segment_index)
        cached = self.delta_xor_graphs.get(key)
        if cached is not None and cached.signature == signature:
            cached.graph.replay()
            return DeltaXorGraphMetrics(replays=1, graphed_routes=len(pairs))
        if cached is not None:
            for destination, source in pairs:
                destination.bitwise_xor_(source)
            self.delta_xor_graphs_invalidated = True
            return DeltaXorGraphMetrics(eager_routes=len(pairs))
        if not allow_capture:
            for destination, source in pairs:
                destination.bitwise_xor_(source)
            self.delta_xor_graphs_invalidated = self.delta_cuda_graphs
            return DeltaXorGraphMetrics(eager_routes=len(pairs))

        # CUDA capture must begin after the decode dependency and earlier applies complete.
        self.delta_apply_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin(capture_error_mode="thread_local")
        for destination, source in pairs:
            destination.bitwise_xor_(source)
        graph.capture_end()
        graph.replay()
        self.delta_xor_graphs[key] = DeltaXorGraph(graph=graph, signature=signature)
        return DeltaXorGraphMetrics(captures=1, graphed_routes=len(pairs))

    def direct_delta_route_pairs(
        self,
        delta_copy: DeltaCopyPlan,
        destination: torch.Tensor,
        decoded: dict[tuple[int, str], tuple[torch.Tensor, DeltaTensorMetadata]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        destination_bytes = destination.view(torch.uint8).reshape(-1)
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for binding in delta_copy.routes:
            source = decoded[binding.source.key][0].view(torch.uint8).reshape(-1)
            pairs.append(
                (
                    destination_bytes.narrow(0, binding.staging_offset, binding.nbytes),
                    source.narrow(0, binding.source_offset, binding.nbytes),
                )
            )
        return pairs

    def apply_replayed_delta(
        self,
        copy_plan: TensorCopyPlan,
        destination: torch.Tensor,
        decoded: dict[tuple[int, str], tuple[torch.Tensor, DeltaTensorMetadata]],
    ) -> None:
        self.populate_delta_staging(copy_plan, decoded)
        value = apply_chain(copy_plan.staging_tensor, copy_plan.replay_ops)
        copy = copy_plan.recorded_copy
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

    def validate_decoded_delta_sources(
        self,
        transfer_group: WeightTransferGroup,
        decoded: dict[tuple[int, str], tuple[torch.Tensor, DeltaTensorMetadata]],
    ) -> None:
        for key, spec in self.delta_group_plans[id(transfer_group)].source_specs.items():
            item = decoded.get(key)
            if item is None:
                continue
            value, metadata = item
            if metadata.dtype != spec.dtype:
                raise RuntimeError(
                    f"NIXL XOR source dtype {metadata.dtype} does not match wire dtype "
                    f"{spec.dtype} for {metadata.name!r}"
                )
            if metadata.resolved_global_shape != spec.global_shape:
                raise RuntimeError(
                    f"NIXL XOR source shape {metadata.resolved_global_shape} does not match "
                    f"the transfer table shape {spec.global_shape} for {metadata.name!r}"
                )
            if metadata.nbytes != spec.nbytes or value.numel() * value.element_size() != spec.nbytes:
                raise RuntimeError(
                    f"NIXL XOR shard for {metadata.name!r} has {metadata.nbytes} bytes; "
                    f"the transfer table requires {spec.nbytes}"
                )

    def apply_direct_delta_routes(
        self,
        copy_plan: TensorCopyPlan,
        destination: torch.Tensor,
        decoded: dict[tuple[int, str], tuple[torch.Tensor, DeltaTensorMetadata]],
    ) -> None:
        delta_copy = self.delta_copy_plans[id(copy_plan)]
        for destination_slice, source_slice in self.direct_delta_route_pairs(delta_copy, destination, decoded):
            destination_slice.bitwise_xor_(source_slice)

    def populate_delta_staging(
        self,
        copy_plan: TensorCopyPlan,
        decoded: dict[tuple[int, str], tuple[torch.Tensor, DeltaTensorMetadata]],
    ) -> None:
        destination_bytes = copy_plan.staging_tensor.view(torch.uint8).reshape(-1)
        for binding in self.delta_copy_plans[id(copy_plan)].routes:
            source_bytes = decoded[binding.source.key][0].view(torch.uint8).reshape(-1)
            destination_bytes.narrow(0, binding.staging_offset, binding.nbytes).copy_(
                source_bytes.narrow(0, binding.source_offset, binding.nbytes)
            )
