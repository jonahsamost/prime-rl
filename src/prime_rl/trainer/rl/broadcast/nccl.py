import os
import pickle
import time
from math import prod
from pathlib import Path
from typing import Callable, Generator, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.configs.trainer import NCCLWeightBroadcastConfig
from prime_rl.trainer.conversion_utils import get_max_layer_num
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.utils.client import NCCL_READY_MARKER
from prime_rl.utils.logger import get_logger
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable
from prime_rl.utils.pathing import sync_wait_for_path
from prime_rl.utils.utils import get_broadcast_dir, get_step_path
from prime_rl.utils.vlm import get_layer_prefix
from prime_rl.weight_sync.bf16_delta import (
    BF16DeltaUpdate,
    CompressedDeltaFrame,
    DeltaTensorMetadata,
    ShardedBF16DeltaUpdate,
    WeightUpdateHeader,
    WeightUpdateKind,
    encode_weight_update_header,
    packed_delta_nbytes,
    validate_sharded_delta_update,
)
from prime_rl.weight_sync.profiling import (
    PhaseProfile,
    PhaseProfiler,
    WeightSyncMetrics,
    cuda_event_pair,
    elapsed_cuda_ms,
)


def broadcast_integer(
    integer: int,
    communicator: PyNcclCommunicator,
    profile: WeightSyncMetrics | None = None,
) -> None:
    """Broadcast an integer to a process group using NCCL communicator."""
    integer_tensor = torch.tensor([integer], dtype=torch.long).cuda()
    _broadcast_tensor(integer_tensor, communicator, profile)


def broadcast_update_header(
    header: WeightUpdateHeader,
    communicator: PyNcclCommunicator,
    profile: WeightSyncMetrics | None = None,
) -> None:
    values = encode_weight_update_header(header, device=communicator.device)
    _broadcast_tensor(values, communicator, profile)


def _broadcast_tensor(
    tensor: Tensor,
    communicator: PyNcclCommunicator,
    profile: WeightSyncMetrics | None,
) -> None:
    if profile is not None:
        profile.wire_bytes += tensor.numel() * tensor.element_size()
        profile.nccl_call_count += 1
    events = cuda_event_pair() if profile is not None else None
    if events is not None:
        events[0].record()
    communicator.broadcast(tensor, src=0)
    if events is not None:
        events[1].record()
        profile.sender_nccl_ms += elapsed_cuda_ms(events)


def _stage_bytes(
    payload: bytes,
    communicator: PyNcclCommunicator,
    profile: WeightSyncMetrics | None,
) -> Tensor:
    events = cuda_event_pair() if profile is not None else None
    if events is not None:
        events[0].record()
    values = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(communicator.device)
    if events is not None:
        events[1].record()
        profile.sender_stage_h2d_ms += elapsed_cuda_ms(events)
    return values


def broadcast_bytes(
    payload: bytes,
    communicator: PyNcclCommunicator,
    profile: WeightSyncMetrics | None = None,
) -> None:
    size = torch.tensor([len(payload)], dtype=torch.long, device=communicator.device)
    _broadcast_tensor(size, communicator, profile)
    values = _stage_bytes(payload, communicator, profile)
    _broadcast_tensor(values, communicator, profile)


def broadcast_compressed_delta(
    update: ShardedBF16DeltaUpdate,
    communicator: PyNcclCommunicator,
    profile: WeightSyncMetrics | None = None,
) -> None:
    metadata = pickle.dumps(
        tuple((shard.tensors, shard.frames, shard.compressed_nbytes) for shard in update.shards)
    )
    broadcast_bytes(metadata, communicator, profile)
    for rank, shard in enumerate(update.shards):
        if shard.payload.device != communicator.device:
            raise ValueError(
                f"BF16 delta payload for trainer rank {rank} is on {shard.payload.device}; "
                f"NCCL communicator uses {communicator.device}"
            )
        _broadcast_tensor(shard.payload, communicator, profile)


def _serialize_local_delta_metadata(update: BF16DeltaUpdate) -> bytes:
    return pickle.dumps((update.base_step, update.step, update.tensors, update.frames))


def _deserialize_local_delta_metadata(payload: Tensor, compressed_payload: Tensor) -> BF16DeltaUpdate:
    decoded = pickle.loads(payload.cpu().numpy().tobytes())
    if not isinstance(decoded, tuple) or len(decoded) != 4:
        raise ValueError("invalid trainer BF16 delta metadata envelope")
    base_step, step, tensors, frames = decoded
    if not isinstance(base_step, int) or not isinstance(step, int):
        raise ValueError("trainer BF16 delta metadata has invalid policy versions")
    if not isinstance(tensors, tuple) or not all(isinstance(item, DeltaTensorMetadata) for item in tensors):
        raise ValueError("trainer BF16 delta metadata has invalid tensor manifest")
    if not isinstance(frames, tuple) or not all(isinstance(item, CompressedDeltaFrame) for item in frames):
        raise ValueError("trainer BF16 delta metadata has invalid frame manifest")
    if compressed_payload.numel() != packed_delta_nbytes(frames):
        raise ValueError(
            f"trainer BF16 delta payload has {compressed_payload.numel()} bytes; "
            f"metadata requires {packed_delta_nbytes(frames)}"
        )
    return BF16DeltaUpdate(
        base_step=base_step,
        step=step,
        tensors=tensors,
        frames=frames,
        payload=compressed_payload,
    )


def _gather_variable_cuda_tensor_to_rank_zero(
    local_value: Tensor,
    sizes: list[int],
) -> Tensor | None:
    """Collect exact-sized CUDA byte segments on rank 0 through one NCCL collective."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if len(sizes) != world_size or local_value.numel() != sizes[rank]:
        raise ValueError(
            f"invalid variable gather sizes {sizes} for rank {rank} tensor with {local_value.numel()} elements"
        )
    input_splits = [0] * world_size
    input_splits[0] = local_value.numel()
    output_splits = sizes if rank == 0 else [0] * world_size
    output = torch.empty(
        sum(output_splits),
        dtype=local_value.dtype,
        device=local_value.device,
    )
    dist.all_to_all_single(
        output,
        local_value,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
    )
    return output if rank == 0 else None


def _split_gathered_tensor(value: Tensor, sizes: list[int]) -> list[Tensor]:
    pieces: list[Tensor] = []
    offset = 0
    for size in sizes:
        pieces.append(value.narrow(0, offset, size))
        offset += size
    if offset != value.numel():
        raise ValueError(f"variable gather describes {offset} elements; received {value.numel()}")
    return pieces


def _delta_gather_debug(rank: int, stage: str, **details: object) -> None:
    if os.environ.get("PRIME_RL_DELTA_GATHER_DEBUG") != "1":
        return
    local_rank = os.environ.get("LOCAL_RANK", "unknown")
    device = f"cuda:{local_rank}" if local_rank != "unknown" else "cuda:unknown"
    suffix = " ".join(f"{key}={value}" for key, value in details.items())
    print(
        f"[{time.time():.6f}] [bf16-delta-gather pid={os.getpid()} rank={rank} "
        f"local_rank={local_rank} device={device}] {stage} {suffix}".rstrip(),
        flush=True,
    )


def gather_compressed_delta_updates(
    local_update: BF16DeltaUpdate | None,
    *,
    profile: WeightSyncMetrics | None = None,
) -> tuple[ShardedBF16DeltaUpdate | None, bool]:
    """Gather rank-local compressed FSDP deltas onto trainer rank 0."""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if world_size == 1:
        if local_update is None:
            return None, False
        sharded = ShardedBF16DeltaUpdate(
            base_step=local_update.base_step,
            step=local_update.step,
            shards=(local_update,),
            profile=profile,
        )
        validate_sharded_delta_update(sharded)
        if profile is not None:
            profile.trainer_shard_count = 1
            profile.min_rank_compressed_bytes = local_update.compressed_nbytes
            profile.max_rank_compressed_bytes = local_update.compressed_nbytes
            profile.rank_compressed_bytes = [local_update.compressed_nbytes]
        return sharded, True

    device = (
        local_update.payload.device
        if local_update is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    available = torch.tensor([local_update is not None], dtype=torch.uint8, device=device)
    _delta_gather_debug(rank, "before availability all_reduce", local_available=local_update is not None)
    dist.all_reduce(available, op=dist.ReduceOp.MIN)
    _delta_gather_debug(rank, "after availability all_reduce", globally_available=bool(available.item()))
    if not bool(available.item()):
        return None, False
    assert local_update is not None

    metadata_bytes = _serialize_local_delta_metadata(local_update)
    metadata = torch.frombuffer(bytearray(metadata_bytes), dtype=torch.uint8).to(device)
    local_sizes = torch.tensor(
        [metadata.numel(), local_update.payload.numel()],
        dtype=torch.long,
        device=device,
    )
    gathered_sizes = [torch.empty_like(local_sizes) for _ in range(world_size)]
    gather_start = time.perf_counter()
    gather_events = cuda_event_pair() if profile is not None else None
    if gather_events is not None:
        gather_events[0].record()
    _delta_gather_debug(
        rank,
        "before size all_gather",
        metadata_bytes=metadata.numel(),
        payload_bytes=local_update.payload.numel(),
    )
    dist.all_gather(gathered_sizes, local_sizes)
    sizes = [tuple(int(value) for value in item.tolist()) for item in gathered_sizes]
    _delta_gather_debug(rank, "after size all_gather", gathered_sizes=sizes)

    metadata_sizes = [metadata_size for metadata_size, _payload_size in sizes]
    payload_sizes = [payload_size for _metadata_size, payload_size in sizes]
    _delta_gather_debug(
        rank,
        "before metadata all_to_all_single",
        local_bytes=metadata.numel(),
        input_splits=[metadata.numel(), *([0] * (world_size - 1))],
        output_splits=metadata_sizes if rank == 0 else [0] * world_size,
    )
    gathered_metadata = _gather_variable_cuda_tensor_to_rank_zero(metadata, metadata_sizes)
    _delta_gather_debug(
        rank,
        "after metadata all_to_all_single",
        received_bytes=gathered_metadata.numel() if gathered_metadata is not None else 0,
    )
    _delta_gather_debug(
        rank,
        "before payload all_to_all_single",
        local_bytes=local_update.payload.numel(),
        input_splits=[local_update.payload.numel(), *([0] * (world_size - 1))],
        output_splits=payload_sizes if rank == 0 else [0] * world_size,
    )
    gathered_payload = _gather_variable_cuda_tensor_to_rank_zero(local_update.payload, payload_sizes)
    _delta_gather_debug(
        rank,
        "after payload all_to_all_single",
        received_bytes=gathered_payload.numel() if gathered_payload is not None else 0,
    )
    metadata_by_rank: list[Tensor] | None = None
    payload_by_rank: list[Tensor] | None = None
    if rank == 0:
        assert gathered_metadata is not None and gathered_payload is not None
        metadata_by_rank = _split_gathered_tensor(gathered_metadata, metadata_sizes)
        # all_to_all_single packs variable-size rank segments back-to-back.
        # Segment offsets therefore need not preserve nvCOMP's 256-byte input
        # alignment. Clone each compressed segment into its own CUDA allocation;
        # this remains a compressed-only operation and gives every rank payload
        # an aligned base address for decoding and downstream NCCL broadcast.
        payload_by_rank = [piece.clone() for piece in _split_gathered_tensor(gathered_payload, payload_sizes)]
    _delta_gather_debug(rank, "before final gather barrier")
    dist.barrier()
    _delta_gather_debug(rank, "after final gather barrier")
    if gather_events is not None:
        gather_events[1].record()
        profile.trainer_gather_gpu_ms += elapsed_cuda_ms(gather_events)
    if profile is not None:
        profile.trainer_gather_wall_ms += (time.perf_counter() - gather_start) * 1000
        profile.trainer_shard_count = world_size
        profile.trainer_gather_bytes += sum(
            metadata_size + payload_size for metadata_size, payload_size in sizes[1:]
        )
        profile.min_rank_compressed_bytes = min(payload_sizes)
        profile.max_rank_compressed_bytes = max(payload_sizes)
        profile.rank_compressed_bytes = payload_sizes

    if rank != 0:
        return None, True
    assert metadata_by_rank is not None and payload_by_rank is not None
    shards = [
        _deserialize_local_delta_metadata(metadata_by_rank[index], payload_by_rank[index])
        for index in range(world_size)
    ]
    sharded = ShardedBF16DeltaUpdate(
        base_step=local_update.base_step,
        step=local_update.step,
        shards=tuple(shards),
        profile=profile,
    )
    validate_sharded_delta_update(sharded)
    return sharded, True


def broadcast_state_dict(
    state_dict: dict[str, Tensor],
    communicator: PyNcclCommunicator,
    profile: WeightSyncMetrics | None = None,
) -> None:
    """Broadcast a state dict to NCCL process group using the PyNcclCommunicator."""
    # Group tensors by dtype
    dtype_groups: dict[torch.dtype, list[tuple[str, Tensor]]] = {}
    for key, value in state_dict.items():
        assert not isinstance(value, DTensor), (
            "DTensor is not supported for broadcast, should have been converted to tensor already"
        )
        dtype = value.dtype
        if dtype not in dtype_groups:
            dtype_groups[dtype] = []
        dtype_groups[dtype].append((key, value))
        if profile is not None:
            nbytes = value.numel() * value.element_size()
            profile.raw_bytes += nbytes
            profile.compressed_bytes += nbytes
            profile.tensor_count += 1
            profile.largest_tensor_bytes = max(profile.largest_tensor_bytes, nbytes)

    # Build metadata: for each dtype group, store keys and shapes
    metadata = {}
    for dtype, items in dtype_groups.items():
        metadata[dtype] = [(key, value.shape, value.numel()) for key, value in items]

    # Send metadata
    state = pickle.dumps(metadata)
    size_tensor = torch.tensor([len(state)], dtype=torch.long).cuda()
    _broadcast_tensor(size_tensor, communicator, profile)
    state_tensor = _stage_bytes(state, communicator, profile)
    _broadcast_tensor(state_tensor, communicator, profile)

    # Concatenate and broadcast tensors grouped by dtype
    for dtype, items in dtype_groups.items():
        # Flatten all tensors and concatenate
        flat_tensors = [value.flatten() for _, value in items]
        pack_events = cuda_event_pair() if profile is not None else None
        if pack_events is not None:
            pack_events[0].record()
        concatenated = torch.cat(flat_tensors)
        if pack_events is not None:
            pack_events[1].record()
            profile.full_pack_ms += elapsed_cuda_ms(pack_events)
        _broadcast_tensor(concatenated, communicator, profile)
        if profile is not None:
            profile.frame_count += 1
        del concatenated
        # Clean up individual tensors
        for _, value in items:
            del value


def filter_state_dict_by_layers(
    state_dict: dict[str, torch.Tensor], num_layers: int, layer_prefix: str
) -> Generator[tuple[int, dict[str, torch.Tensor]], None, None]:
    """Yield non-layer weights first, then each layer's weights.

    Yields (layer_idx, layer_state_dict) where layer_idx is -1 for the non-layer
    dict and the actual layer index (0, 1, ...) for layer dicts.
    """
    yield -1, {key: value for key, value in state_dict.items() if not key.startswith(layer_prefix)}

    for i in range(num_layers):
        yield (
            i,
            {key: value for key, value in state_dict.items() if key.startswith(f"{layer_prefix}{i}.")},
        )


def preprocess_layer_checkpoint(
    model: nn.Module,
    layer_state_dict: dict[str, Tensor],
    layer_idx: int,
) -> dict[str, Tensor]:
    if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(layer_state_dict):
        model.convert_layer_to_hf(layer_state_dict, layer_idx)
        return layer_state_dict

    from transformers.core_model_loading import revert_weight_conversion

    return revert_weight_conversion(model, layer_state_dict)


def preprocess_layer_quantized(
    model: nn.Module,
    layer_state_dict: dict[str, Tensor],
    layer_idx: int,
) -> dict[str, Tensor]:
    if layer_idx < 0:
        return layer_state_dict
    return model.convert_layer_to_vllm_kernel(layer_state_dict, layer_idx, quantize_fp8=True)


class NCCLWeightBroadcastSender:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
        dtype: torch.dtype = torch.bfloat16,
        quantize_in_weight_transfer: bool = False,
        delta_mode: str = "none",
        profiling_sample_interval_ms: float | None = None,
    ):
        self.logger = get_logger()
        self.world = get_world()
        self.dtype = dtype
        self.quantize_in_weight_transfer = quantize_in_weight_transfer
        self.delta_mode = delta_mode
        self.last_broadcast_step: int | None = None
        self.last_profile: WeightSyncMetrics | None = None
        self._optimizer_step_profile: tuple[float, float, PhaseProfile] | None = None
        if self.delta_mode != "none" and self.quantize_in_weight_transfer:
            raise ValueError("BF16 XOR delta mode is incompatible with quantize_in_weight_transfer")

        if self.world.is_master:
            disable_nccl_p2p_if_unavailable()
            # Trainer is on rank 0 in process group with all inference GPUs
            pg = StatelessProcessGroup.create(
                host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout
            )
            self.communicator = PyNcclCommunicator(pg, device=device)
            self.logger.debug("NCCL broadcast initialized on master rank")
        else:
            self.logger.debug("NCCL broadcast initialized on non-master rank (no communicator)")
        self.profiler = PhaseProfiler(
            enabled=profiling_sample_interval_ms is not None,
            device=device,
            sample_interval_ms=profiling_sample_interval_ms or 5.0,
        )

    @torch.no_grad()
    def broadcast_weights(
        self,
        model: nn.Module,
        step: int,
        delta_update: BF16DeltaUpdate | None = None,
    ) -> None:
        """Broadcast the state dict of a model into the inference pool using NCCL."""
        profile = delta_update.profile if delta_update is not None else None
        if self.profiler.enabled and profile is None:
            profile = WeightSyncMetrics()
        phase_name = "delta_send" if delta_update is not None else "full_send"
        with self.profiler.measure(phase_name) as phase:
            self._broadcast_weights(model, step, delta_update, profile)
        if profile is not None:
            if self._optimizer_step_profile is not None:
                wall_ms, gpu_ms, optimizer_phase = self._optimizer_step_profile
                profile.optimizer_step_wall_ms = wall_ms
                profile.optimizer_step_gpu_ms = gpu_ms
                profile.phases[optimizer_phase.name] = optimizer_phase
                self._optimizer_step_profile = None
            profile.phases[phase.name] = phase
            self.last_profile = profile
            self.logger.info(profile.structured_log(event="weight_sync_profile", step=step, role="trainer"))

    def set_optimizer_step_profile(self, *, wall_ms: float, gpu_ms: float, phase: PhaseProfile) -> None:
        self._optimizer_step_profile = (wall_ms, gpu_ms, phase)

    def _broadcast_weights(
        self,
        model: nn.Module,
        step: int,
        delta_update: BF16DeltaUpdate | None,
        profile: WeightSyncMetrics | None,
    ) -> None:
        sharded_delta: ShardedBF16DeltaUpdate | None = None
        is_delta_update = False
        if self.delta_mode == "bf16_xor":
            sharded_delta, is_delta_update = gather_compressed_delta_updates(delta_update, profile=profile)
        if is_delta_update:
            assert delta_update is not None
            if self.delta_mode != "bf16_xor":
                raise ValueError("received a BF16 delta update while delta mode is disabled")
            if delta_update.step != step:
                raise ValueError(f"delta step {delta_update.step} does not match broadcast step {step}")
            if delta_update.base_step != self.last_broadcast_step:
                raise ValueError(
                    f"delta base step {delta_update.base_step} does not match last broadcast {self.last_broadcast_step}"
                )
            if getattr(model.config, "model_type", None) != "qwen3":
                raise ValueError("BF16 XOR delta broadcast currently supports Qwen3 only")
            header = WeightUpdateHeader(WeightUpdateKind.BF16_XOR, delta_update.base_step, step)
        else:
            header = WeightUpdateHeader(WeightUpdateKind.FULL, -1, step)
            state_dict = model.state_dict()

        if self.world.is_master and self.delta_mode != "none":
            broadcast_update_header(header, self.communicator, profile)

        if is_delta_update:
            if self.world.is_master:
                assert sharded_delta is not None
                if profile is not None:
                    profile.raw_bytes = sharded_delta.uncompressed_nbytes
                    profile.compressed_bytes = sharded_delta.compressed_nbytes
                    profile.tensor_count = sharded_delta.tensor_count
                    profile.frame_count = sharded_delta.frame_count
                    profile.largest_tensor_bytes = max(
                        prod(item.resolved_global_shape) * 2 for item in sharded_delta.shards[0].tensors
                    )
                broadcast_compressed_delta(sharded_delta, self.communicator, profile)
            self.last_broadcast_step = step
            return

        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)
        num_state_dict_to_send = num_layers + 1  # we send all layer plus the remaining weights

        if self.world.is_master:
            broadcast_integer(num_state_dict_to_send, self.communicator, profile)

        self.logger.debug(f"Broadcasting {num_state_dict_to_send} layer state dicts")
        preprocess_fn: Callable[[nn.Module, dict[str, Tensor], int], dict[str, Tensor]]
        if self.quantize_in_weight_transfer:
            preprocess_fn = preprocess_layer_quantized
        else:
            preprocess_fn = preprocess_layer_checkpoint

        for layer_id, layer_state_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
            layer_state_dict = self._resolve_dtensors(layer_state_dict)
            layer_state_dict = preprocess_fn(model, layer_state_dict, layer_id)
            if self.world.is_master:
                broadcast_state_dict(layer_state_dict, self.communicator, profile)
        self.last_broadcast_step = step

    def _resolve_dtensors(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        for key, value in list(state_dict.items()):
            if isinstance(value, DTensor):
                state_dict[key] = cast(DTensor, value.to(self.dtype)).full_tensor()
        return state_dict


class NCCLWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine using NCCL."""

    def __init__(
        self,
        output_dir: Path,
        config: NCCLWeightBroadcastConfig,
        device: int | str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(output_dir)
        self.logger = get_logger()
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        self.nccl_broadcast_sender = NCCLWeightBroadcastSender(
            config.host,
            config.port,
            0,
            config.inference_world_size + 1,
            device,
            config.timeout,
            dtype,
            quantize_in_weight_transfer=config.quantize_in_weight_transfer,
            delta_mode=config.delta_mode,
            profiling_sample_interval_ms=(
                config.profiling.sample_interval_ms if config.profiling is not None else None
            ),
        )

    @torch.no_grad()
    def broadcast_weights(
        self,
        model: nn.Module,
        step: int,
        delta_update: BF16DeltaUpdate | None = None,
    ) -> None:
        """Broadcast the state dict of a model into the inference pool using NCCL and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting weights to inference engine via NCCL")
        start_time = time.perf_counter()
        # `_compute_notified_runs` is a pure function of SPMD-replicated state on
        # multi_run_manager, so every trainer rank derives the same list. Only
        # the master touches the filesystem to notify the orchestrator, but all
        # ranks must wait for the inference pool before entering the broadcast path:
        # the broadcast preparation (DTensor resolution, quantization) enqueues
        # collectives on non-master ranks, and if those ranks start prep before
        # the orchestrator has paused inference, the collectives sit unmatched
        # until NCCL's watchdog kills the process after 10 min.
        notified_runs = self._compute_notified_runs()
        if self.world.is_master:
            self._notify_orchestrator(notified_runs)
            self._wait_for_nccl_ready(notified_runs)
        if self.world.world_size > 1:
            dist.barrier()
        self.nccl_broadcast_sender.broadcast_weights(model, step, delta_update)
        self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def _compute_notified_runs(self) -> list[tuple[int, Path]]:
        """Derive the list of (run_idx, save_dir) pairs that need broadcasting.

        Pure function of `multi_run_manager` state, which is replicated across
        trainer ranks (SPMD). Returns the same list on every rank so master and
        non-master ranks agree on which NCCL_READY markers to wait for.
        """
        notified_runs: list[tuple[int, Path]] = []
        for idx in self.multi_run_manager.used_idxs:
            if not self.multi_run_manager.ready_to_update[idx]:
                continue
            try:
                # pack() already advanced progress to the next step, so the model we just
                # trained — policy v(step-1) — broadcasts to broadcasts/step_{step-1}.
                save_dir = get_step_path(
                    get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                    self.multi_run_manager.progress[idx].step - 1,
                )
                notified_runs.append((idx, save_dir))
            except FileNotFoundError:
                self.logger.warning(f"Run {idx} is deleted, skipping")
            except Exception as e:
                self.logger.error(f"Error resolving broadcast dir for run {idx}: {e}")
        return notified_runs

    def _notify_orchestrator(self, notified_runs: list[tuple[int, Path]]) -> None:
        """Create STABLE markers for each notified run and clear their ready flags.

        Master-only side effects (filesystem writes + state mutation). Called
        after `_compute_notified_runs`; non-master ranks skip this entirely.
        """
        for idx, save_dir in notified_runs:
            try:
                save_dir.mkdir(parents=True, exist_ok=True)
                stable_file = save_dir / "STABLE"
                stable_file.touch()
            except FileNotFoundError:
                self.logger.warning(f"Run {idx} is deleted, skipping")
            except Exception as e:
                self.logger.error(f"Error broadcasting weights for run {idx}: {e}")
            finally:
                self.multi_run_manager.ready_to_update[idx] = False

    def _wait_for_nccl_ready(self, notified_runs: list[tuple[int, Path]]):
        """Wait for inference workers to signal they are ready to receive NCCL broadcast."""
        for idx, save_dir in notified_runs:
            nccl_ready_file = save_dir / NCCL_READY_MARKER
            self.logger.debug(f"Waiting for NCCL_READY marker at {nccl_ready_file}")
            sync_wait_for_path(nccl_ready_file, interval=0.1, log_interval=10)
            self.logger.debug(f"Inference workers ready for NCCL broadcast (run {idx})")
