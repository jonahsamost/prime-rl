import pickle
import re
import time
from typing import TYPE_CHECKING, Generator, cast

import torch
from torch.nn import Module
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.qwen3_bf16_delta import apply_qwen3_bf16_source_deltas_
from prime_rl.inference.vllm.worker.weight_transfer import (
    load_weights_checkpoint_layerwise,
    load_weights_kernel,
    update_mla_absorbed_weights,
)
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable
from prime_rl.weight_sync.bf16_delta import (
    BF16DeltaUpdate,
    CompressedDeltaFrame,
    DeltaTensorMetadata,
    NvcompLZ4Codec,
    WeightUpdateHeader,
    WeightUpdateKind,
    decode_delta_tensors,
    decode_weight_update_header,
    packed_delta_nbytes,
    validate_delta_update,
)
from prime_rl.weight_sync.profiling import (
    PhaseProfiler,
    WeightSyncMetrics,
    cuda_event_pair,
    elapsed_cuda_ms,
)

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nccl")


def _receive_tensor(
    tensor: torch.Tensor,
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
        profile.receiver_nccl_ms += elapsed_cuda_ms(events)


def receive_integer(
    communicator: PyNcclCommunicator,
    profile: WeightSyncMetrics | None = None,
) -> int:
    """Receive an integer from the trainer master rank using NCCL communicator."""
    integer_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    _receive_tensor(integer_tensor, communicator, profile)
    return cast(int, integer_tensor.item())


def receive_update_header(
    communicator: PyNcclCommunicator,
    profile: WeightSyncMetrics | None = None,
) -> WeightUpdateHeader:
    values = torch.empty(5, dtype=torch.long, device=communicator.device)
    _receive_tensor(values, communicator, profile)
    try:
        return decode_weight_update_header(values)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def receive_bytes(
    communicator: PyNcclCommunicator,
    profile: WeightSyncMetrics | None = None,
) -> bytes:
    size = receive_integer(communicator, profile)
    values = torch.empty(size, dtype=torch.uint8, device=communicator.device)
    _receive_tensor(values, communicator, profile)
    events = cuda_event_pair() if profile is not None else None
    if events is not None:
        events[0].record()
    cpu_values = values.cpu()
    if events is not None:
        events[1].record()
        profile.receiver_stage_d2h_ms += elapsed_cuda_ms(events)
    return cpu_values.numpy().tobytes()


def receive_compressed_delta(
    communicator: PyNcclCommunicator,
    *,
    base_step: int,
    step: int,
    profile: WeightSyncMetrics | None = None,
) -> BF16DeltaUpdate:
    tensors, frame_sizes = pickle.loads(receive_bytes(communicator, profile))
    if not isinstance(tensors, tuple) or not all(isinstance(item, DeltaTensorMetadata) for item in tensors):
        raise RuntimeError("invalid BF16 delta tensor metadata")
    frames = tuple(
        CompressedDeltaFrame(
            first_tensor_index=first_tensor_index,
            tensor_count=tensor_count,
            uncompressed_nbytes=uncompressed_nbytes,
            compressed_nbytes=compressed_nbytes,
        )
        for first_tensor_index, tensor_count, uncompressed_nbytes, compressed_nbytes in frame_sizes
    )
    payload = torch.empty(
        packed_delta_nbytes(frames),
        dtype=torch.uint8,
        device=communicator.device,
    )
    _receive_tensor(payload, communicator, profile)
    update = BF16DeltaUpdate(
        base_step=base_step,
        step=step,
        tensors=tensors,
        frames=frames,
        payload=payload,
        profile=profile,
    )
    try:
        validate_delta_update(update)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    return update


_QWEN_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")


def qwen_layer_index(name: str) -> int:
    match = _QWEN_LAYER_PATTERN.match(name)
    return int(match.group(1)) if match else -1


def receive_state_dict(
    communicator: PyNcclCommunicator,
    profile: WeightSyncMetrics | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Stream tensors in a state dict broadcasted over NCCL."""
    size_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    _receive_tensor(size_tensor, communicator, profile)
    state_tensor = torch.empty(cast(int, size_tensor.item()), dtype=torch.uint8).to(communicator.device)
    _receive_tensor(state_tensor, communicator, profile)

    metadata_events = cuda_event_pair() if profile is not None else None
    if metadata_events is not None:
        metadata_events[0].record()
    state_cpu = state_tensor.cpu()
    if metadata_events is not None:
        metadata_events[1].record()
        profile.receiver_stage_d2h_ms += elapsed_cuda_ms(metadata_events)
    metadata = pickle.loads(bytes(state_cpu.numpy()))

    # Receive concatenated tensors per dtype and split them back
    for dtype, tensor_info_list in metadata.items():
        # Receive concatenated tensor for this dtype
        total_elements = sum(numel for _, _, numel in tensor_info_list)
        concatenated = torch.empty(total_elements, dtype=dtype, device=communicator.device)
        _receive_tensor(concatenated, communicator, profile)
        if profile is not None:
            nbytes = concatenated.numel() * concatenated.element_size()
            profile.raw_bytes += nbytes
            profile.compressed_bytes += nbytes
            profile.tensor_count += len(tensor_info_list)
            profile.frame_count += 1
            profile.largest_tensor_bytes = max(
                profile.largest_tensor_bytes,
                max(numel * dtype.itemsize for _, _, numel in tensor_info_list),
            )

        # Split concatenated tensor back into individual tensors
        offset = 0
        for key, shape, numel in tensor_info_list:
            tensor = concatenated[offset : offset + numel].view(shape).clone()
            offset += numel
            try:
                yield key, tensor
            finally:
                del tensor

        del concatenated


class NCCLWeightBroadcastReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
        delta_mode: str = "none",
        profiling_sample_interval_ms: float | None = None,
    ):
        logger.info(f"Initializing NCCL broadcast receiver ({host}:{port}, rank={rank}, world_size={world_size})")
        disable_nccl_p2p_if_unavailable()

        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout)
        self.communicator = PyNcclCommunicator(pg, device=device)
        self.delta_mode = delta_mode
        self.current_step: int | None = None
        self.delta_codec = NvcompLZ4Codec(device) if delta_mode == "bf16_xor" else None
        self.profiler = PhaseProfiler(
            enabled=profiling_sample_interval_ms is not None,
            device=device,
            sample_interval_ms=profiling_sample_interval_ms or 5.0,
        )

    @torch.no_grad()
    def receive_state_dict(self, profile: WeightSyncMetrics | None = None):
        """Receives the state dict of a model from the trainer master rank using NCCL communicator."""
        logger.info("Receiving weights from trainer")
        num_state_dict_to_receive = receive_integer(self.communicator, profile)
        logger.info(f"Receiving {num_state_dict_to_receive} layer state dicts")
        for layer_id in range(num_state_dict_to_receive):
            logger.info(f"Receiving state dict {layer_id + 1}/{num_state_dict_to_receive}")
            for key, value in receive_state_dict(self.communicator, profile):
                yield key, value

    def receive_update_header(self, profile: WeightSyncMetrics | None = None) -> WeightUpdateHeader | None:
        if self.delta_mode == "none":
            return None
        return receive_update_header(self.communicator, profile)


class NCCLWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using NCCL."""

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
        profiling_sample_interval_ms: float | None = None,
    ) -> None:
        """Initialize the NCCL broadcast receiver.

        Args:
            rank_offset: Starting GPU offset for this server in the global inference group.
            inference_world_size: Total number of inference GPUs across all servers.
        """
        del session_id
        self.quantize_in_weight_transfer = quantize_in_weight_transfer
        self.delta_mode = delta_mode
        if self.delta_mode != "none" and self.quantize_in_weight_transfer:
            raise ValueError("BF16 XOR delta mode is incompatible with quantize_in_weight_transfer")
        # Use the worker's device index directly as the local rank.
        # The previous dp_group-based computation broke in vLLM v1 multiprocess
        # DP mode where each worker is a separate process with a singleton
        # DP group (rank_in_group is always 0).
        local_rank = self.device.index
        global_rank_inference = rank_offset + local_rank

        logger.info(
            f"Worker [local_rank={local_rank} rank_offset={rank_offset}] "
            f"-> [global_rank={global_rank_inference} inference_world_size={inference_world_size}]"
        )

        self.nccl_broadcast_receiver = NCCLWeightBroadcastReceiver(
            host=host,
            port=port,
            rank=global_rank_inference + 1,  # +1 as the trainer broadcaster is on rank 0
            world_size=inference_world_size + 1,  # +1 as the trainer broadcaster is on rank 0
            device=self.device,
            timeout=timeout,
            delta_mode=delta_mode,
            profiling_sample_interval_ms=profiling_sample_interval_ms,
        )

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    def update_weights_from_path(self, weight_dir: str, requested_step: int = -1) -> None:
        """Update weights with the nccl communicator."""
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        profile = WeightSyncMetrics() if self.nccl_broadcast_receiver.profiler.enabled else None
        header = self.nccl_broadcast_receiver.receive_update_header(profile)
        if header is not None and header.kind == WeightUpdateKind.BF16_XOR:
            if header.base_step != self.nccl_broadcast_receiver.current_step:
                raise RuntimeError(
                    f"cannot apply BF16 delta for step {header.step}: base step {header.base_step} "
                    f"does not match resident step {self.nccl_broadcast_receiver.current_step}"
                )
            with self.nccl_broadcast_receiver.profiler.measure("delta_receive_apply") as phase:
                update = receive_compressed_delta(
                    self.nccl_broadcast_receiver.communicator,
                    base_step=header.base_step,
                    step=header.step,
                    profile=profile,
                )
                if profile is not None:
                    profile.raw_bytes = update.uncompressed_nbytes
                    profile.compressed_bytes = update.compressed_nbytes
                    profile.tensor_count = len(update.tensors)
                    profile.frame_count = len(update.frames)
                    profile.largest_tensor_bytes = max(item.nbytes for item in update.tensors)
                logger.info(
                    "Received nvCOMP LZ4 BF16 XOR update: %d tensors in %d frames, "
                    "%.2f MiB compressed from %.2f MiB (%.1fx)",
                    len(update.tensors),
                    len(update.frames),
                    update.compressed_nbytes / (1024 * 1024),
                    update.uncompressed_nbytes / (1024 * 1024),
                    update.uncompressed_nbytes / update.compressed_nbytes,
                )
                apply_start = time.perf_counter()
                codec = self.nccl_broadcast_receiver.delta_codec
                assert codec is not None
                frame_payloads = list(update.frame_payloads())
                decode_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
                for frame, frame_payload in zip(update.frames, frame_payloads, strict=True):
                    decoded, events = decode_delta_tensors(
                        codec,
                        update.tensors,
                        [frame],
                        [frame_payload],
                        profile=profile is not None,
                    )
                    if events is not None:
                        decode_events.append(events)
                    group_start = 0
                    while group_start < len(decoded):
                        layer_index = qwen_layer_index(decoded[group_start][0])
                        group_end = group_start + 1
                        while group_end < len(decoded) and qwen_layer_index(decoded[group_end][0]) == layer_index:
                            group_end += 1
                        apply_qwen3_bf16_source_deltas_(
                            model,
                            dict(decoded[group_start:group_end]),
                            layer_index=layer_index,
                            profile=profile,
                        )
                        group_start = group_end
                torch.cuda.synchronize(self.device)
                if profile is not None:
                    profile.nvcomp_decompress_gpu_ms += sum(start.elapsed_time(end) for start, end in decode_events)
            logger.info(
                "Applied nvCOMP LZ4 BF16 XOR update for policy v%d in %.2fs",
                header.step,
                time.perf_counter() - apply_start,
            )
            if profile is not None:
                profile.phases[phase.name] = phase
                logger.info(profile.structured_log(event="weight_sync_profile", step=header.step, role="inference"))
            self.nccl_broadcast_receiver.current_step = header.step
            return

        with self.nccl_broadcast_receiver.profiler.measure("full_receive_apply") as phase:
            state_iter = self.nccl_broadcast_receiver.receive_state_dict(profile)
            if self.quantize_in_weight_transfer:
                load_weights_kernel(model, state_iter)
                update_mla_absorbed_weights(model)
            else:
                load_weights_checkpoint_layerwise(
                    model,
                    state_iter,
                    self.model_runner.model_config,
                    self.vllm_config,
                )
            if profile is not None:
                torch.cuda.synchronize(self.device)
        if profile is not None:
            profile.phases[phase.name] = phase
            profile.full_apply_ms = max(
                0.0,
                phase.wall_ms - profile.receiver_nccl_ms - profile.receiver_stage_d2h_ms,
            )
            step = header.step if header is not None else requested_step
            logger.info(profile.structured_log(event="weight_sync_profile", step=step, role="inference"))
        if header is not None:
            self.nccl_broadcast_receiver.current_step = header.step
