"""XOR-compressed trainer-side NIXL weight broadcasting."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from modelexpress import p2p_pb2

from prime_rl.configs.trainer import NIXLWeightBroadcastConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.nixl.cuda_malloc_memory import use_cuda_malloc_pool
from prime_rl.trainer.rl.broadcast.nixl.delta_manifest import (
    NIXL_DELTA_PROTOCOL_VERSION,
    NIXLDeltaAgent,
    NIXLDeltaFrame,
    NIXLDeltaGroup,
    NIXLDeltaManifest,
    NIXLPolicyMetadata,
    build_local_delta_groups,
    merge_delta_manifest_fragments,
)
from prime_rl.trainer.rl.broadcast.nixl.nixl import NIXLWeightBroadcast
from prime_rl.weight_sync.xor_delta import NVCOMP_FRAME_ALIGNMENT, DeltaUpdate, align_nvcomp_nbytes


class NIXLXorWeightBroadcast(NIXLWeightBroadcast):
    def __init__(
        self,
        output_dir: Path,
        config: NIXLWeightBroadcastConfig,
        parallel_dims: ParallelDims,
    ) -> None:
        if config.delta_mode != "xor":
            raise ValueError(f"NIXLXorWeightBroadcast requires delta_mode='xor', got {config.delta_mode!r}")
        super().__init__(
            output_dir,
            config,
            parallel_dims,
            retain_fp8_resident=True,
        )
        self.delta_staging_arena: torch.Tensor | None = None
        self.delta_staging_registration: object | None = None
        self.delta_staging_slot_bytes = 0
        self.delta_group_copies: list[list[tuple[torch.Tensor, torch.Tensor]]] = []

    def resolve_wire_dtype(
        self,
        tensor_name: str,
        value: torch.Tensor,
        keep_in_fp32: Callable[[str], bool],
    ) -> torch.dtype:
        if self.config.delta_representation == "source":
            return value.dtype
        return super().resolve_wire_dtype(tensor_name, value, keep_in_fp32)

    def release_full_transfer_resources(self) -> None:
        if not self.full_transfer_resources_active:
            return

        source_bytes = sum(
            shard.source_tensor.numel() * shard.source_tensor.element_size() for shard in self.staged_shards
        )
        staging_bytes = sum(arena.numel() * arena.element_size() for arena in self.staging_arenas.values())
        allocated_before = torch.cuda.memory_allocated()
        free_before, _ = torch.cuda.mem_get_info()

        self.deregister_full_staging_arenas()
        self.staging_registrations.clear()
        self.staged_shards_by_group.clear()
        self.staged_shards.clear()
        self.staging_arenas.clear()
        self.trainer_table = None
        self.full_transfer_resources_active = False
        torch.cuda.empty_cache()

        allocated_after = torch.cuda.memory_allocated()
        free_after, _ = torch.cuda.mem_get_info()
        self.logger.info(
            f"Released FP8 startup full-transfer resources on trainer rank {self.world.rank}: "
            f"checkpoint_sources={source_bytes / 2**30:.2f} GiB, "
            f"staging_arenas={staging_bytes / 2**30:.2f} GiB, "
            f"allocated={allocated_before / 2**30:.2f}->{allocated_after / 2**30:.2f} GiB, "
            f"device_free={free_before / 2**30:.2f}->{free_after / 2**30:.2f} GiB"
        )

    def after_full_transfer(self) -> None:
        if self.fp8_producer is not None:
            self.release_full_transfer_resources()

    @torch.no_grad()
    def broadcast_weights(
        self,
        model: nn.Module,
        step: int,
        delta_update: DeltaUpdate | None = None,
    ) -> None:
        ready_runs = list(self.multi_run_manager.ready_to_update_idxs)
        if self.fp8_producer is not None:
            transfer_groups = self.build_transfer_group_index(model.state_dict())
            transfer_tensors = self.fp8_producer.build_transfer_tensors(
                model,
                include_checkpoint=not self.initialized,
            )
            if not self.initialized:
                if transfer_tensors.resident:
                    self.fp8_producer.initialize_tensors(transfer_tensors.resident, step=step)
                self.initialize_transfer(model, transfer_tensors.checkpoint, transfer_groups)
            else:
                if self.last_broadcast_step is None:
                    raise RuntimeError("FP8 resident transfer was initialized without a policy version")
                if transfer_tensors.resident:
                    _snapshot, delta_update = self.fp8_producer.advance_tensors(
                        transfer_tensors.resident,
                        base_step=self.last_broadcast_step,
                        step=step,
                    )
                if transfer_tensors.checkpoint:
                    self.update_staged_sources(transfer_tensors.checkpoint)
            del transfer_tensors
        else:
            self.initialize_transfer(model)
        delta_available = torch.tensor(
            delta_update is not None,
            dtype=torch.uint8,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        availability_op = dist.ReduceOp.MAX if self.fp8_producer is not None else dist.ReduceOp.MIN
        dist.all_reduce(delta_available, op=availability_op)
        if bool(delta_available.item()):
            if self.fp8_producer is None:
                assert delta_update is not None
            self.broadcast_delta(delta_update, step, ready_runs)
            return
        reason = None
        if self.last_broadcast_step is not None:
            reason = "at least one trainer rank rejected its XOR payload"
        self.broadcast_full(step, ready_runs, reason=reason)

    def build_local_delta_manifest(self, update: DeltaUpdate) -> NIXLDeltaManifest:
        groups = build_local_delta_groups(update, self.transfer_group_names)
        frame_payloads = {payload.data_ptr(): payload for payload in update.frame_payloads()}
        group_sizes = [sum(align_nvcomp_nbytes(frame.compressed_nbytes) for frame in frames) for frames in groups]
        required_slot_bytes = max(group_sizes, default=0)
        self.ensure_delta_staging_arena(required_slot_bytes)
        assert self.delta_staging_arena is not None

        staged_groups: list[list[NIXLDeltaFrame]] = []
        self.delta_group_copies = []
        for group_index, frames in enumerate(groups):
            slot_offset = (group_index % self.staging_buffer_count) * self.delta_staging_slot_bytes
            offset = 0
            staged_frames: list[NIXLDeltaFrame] = []
            copies: list[tuple[torch.Tensor, torch.Tensor]] = []
            for frame in frames:
                offset = align_nvcomp_nbytes(offset)
                destination = self.delta_staging_arena.narrow(
                    0,
                    slot_offset + offset,
                    frame.compressed_nbytes,
                )
                staged_frames.append(
                    NIXLDeltaFrame(
                        agent=frame.agent,
                        addr=destination.data_ptr(),
                        compressed_nbytes=frame.compressed_nbytes,
                        uncompressed_nbytes=frame.uncompressed_nbytes,
                        tensors=frame.tensors,
                    )
                )
                copies.append((destination, frame_payloads[frame.addr]))
                offset += frame.compressed_nbytes
            staged_groups.append(staged_frames)
            self.delta_group_copies.append(copies)
        return NIXLDeltaManifest(
            protocol_version=NIXL_DELTA_PROTOCOL_VERSION,
            base_step=update.base_step,
            step=update.step,
            agents=(
                NIXLDeltaAgent(
                    name=self.nixl_agent.name,
                    metadata=self.nixl_agent.get_metadata(),
                    device_id=torch.cuda.current_device(),
                ),
            ),
            groups=tuple(
                NIXLDeltaGroup(name=name, frames=tuple(frames))
                for name, frames in zip(self.transfer_group_names, staged_groups, strict=True)
            ),
            representation=self.config.delta_representation,
            fp8_scale_format=(
                self.config.delta_fp8_scale_format if self.config.delta_representation == "fp8_kernel" else ""
            ),
        )

    def ensure_delta_staging_arena(self, slot_bytes: int) -> None:
        slot_bytes = max(NVCOMP_FRAME_ALIGNMENT, align_nvcomp_nbytes(slot_bytes))
        if self.delta_staging_arena is not None and slot_bytes <= self.delta_staging_slot_bytes:
            return
        if self.delta_staging_registration is not None:
            self.nixl_agent.deregister_tensor(self.delta_staging_registration)
        self.delta_staging_slot_bytes = slot_bytes
        with use_cuda_malloc_pool():
            self.delta_staging_arena = torch.empty(
                self.staging_buffer_count * slot_bytes,
                dtype=torch.uint8,
                device=torch.cuda.current_device(),
            )
        self.delta_staging_registration = self.nixl_agent.register_tensor(self.delta_staging_arena)

    def gather_delta_manifest(self, update: DeltaUpdate | None) -> NIXLDeltaManifest | None:
        if update is None:
            self.delta_group_copies = [[] for _ in self.transfer_group_names]
        fragment = (
            self.build_local_delta_manifest(update).encode() if self.is_serving_rank and update is not None else None
        )
        gathered: list[bytes | None] | None = [None] * self.world.world_size if self.world.is_master else None
        dist.gather_object(fragment, gathered, dst=0)
        if gathered is None:
            return None
        return merge_delta_manifest_fragments(
            [NIXLDeltaManifest.decode(value) for value in gathered if value is not None]
        )

    @torch.no_grad()
    def broadcast_delta(self, update: DeltaUpdate | None, step: int, ready_runs: list[int]) -> None:
        if update is not None and update.step != step:
            raise ValueError(f"delta step {update.step} does not match broadcast step {step}")
        if update is not None and update.base_step != self.last_broadcast_step:
            raise ValueError(
                f"delta base step {update.base_step} does not match last broadcast {self.last_broadcast_step}"
            )
        start = time.perf_counter()
        manifest = self.gather_delta_manifest(update)
        fallback_to_full = False
        if manifest is not None:
            compressed_nbytes = sum(frame.compressed_nbytes for group in manifest.groups for frame in group.frames)
            uncompressed_nbytes = sum(frame.uncompressed_nbytes for group in manifest.groups for frame in group.frames)
            fallback_to_full = (
                self.config.delta_representation != "fp8_kernel" and compressed_nbytes >= uncompressed_nbytes
            )
        decision = [fallback_to_full]
        dist.broadcast_object_list(decision, src=0)
        if decision[0]:
            self.broadcast_full(step, ready_runs, reason="XOR payload was not smaller than full weights")
            return

        if manifest is not None:
            self.update_session.publish(
                nixl_metadata=NIXLPolicyMetadata(
                    kind="xor",
                    payload=manifest.encode(),
                    base_step=manifest.base_step,
                    step=manifest.step,
                ).encode()
            )
            self.update_session.set_status(p2p_pb2.SOURCE_STATUS_READY)
            self.logger.info(
                f"NIXL XOR policy v{step}: {uncompressed_nbytes / compressed_nbytes:.2f}x compression "
                f"({uncompressed_nbytes / 2**30:.2f} GiB raw, "
                f"{compressed_nbytes / 2**30:.2f} GiB compressed)"
            )

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

        for group_index, group_name in enumerate(self.transfer_group_names):
            buffer_index = group_index % self.staging_buffer_count
            if group_index >= self.staging_buffer_count:
                self.finish_staging_buffer_transfer(step, group_index - self.staging_buffer_count)
            if self.is_serving_rank:
                for destination, source in self.delta_group_copies[group_index]:
                    destination.copy_(source)
                torch.cuda.synchronize()
            dist.barrier()
            if self.world.is_master:
                self.publish_group_ready(step, group_index)
                self.logger.debug(f"NIXL XOR policy v{step} group {group_name} ready in buffer {buffer_index}")

        first_pending_group = max(0, len(self.transfer_group_names) - self.staging_buffer_count)
        for group_index in range(first_pending_group, len(self.transfer_group_names)):
            self.finish_staging_buffer_transfer(step, group_index)

        self.finish_policy_transfer()
        self.last_broadcast_step = step

        for run_index in ready_runs:
            self.multi_run_manager.ready_to_update[run_index] = False
        self.logger.info(f"NIXL XOR policy v{step} synchronized in {time.perf_counter() - start:.2f}s")
