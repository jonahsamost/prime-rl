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
    ShardedBF16DeltaUpdate,
    WeightUpdateHeader,
    WeightUpdateKind,
    decode_delta_tensors,
    decode_weight_update_header,
    packed_delta_nbytes,
    reconstruct_delta_tensors,
    validate_sharded_delta_update,
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
) -> None:
    communicator.broadcast(tensor, src=0)


def receive_integer(
    communicator: PyNcclCommunicator,
) -> int:
    """Receive an integer from the trainer master rank using NCCL communicator."""
    integer_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    _receive_tensor(integer_tensor, communicator)
    return cast(int, integer_tensor.item())


def receive_update_header(
    communicator: PyNcclCommunicator,
) -> WeightUpdateHeader:
    values = torch.empty(5, dtype=torch.long, device=communicator.device)
    _receive_tensor(values, communicator)
    try:
        return decode_weight_update_header(values)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def receive_bytes(
    communicator: PyNcclCommunicator,
) -> bytes:
    size = receive_integer(communicator)
    values = torch.empty(size, dtype=torch.uint8, device=communicator.device)
    _receive_tensor(values, communicator)
    cpu_values = values.cpu()
    return cpu_values.numpy().tobytes()


def receive_compressed_delta(
    communicator: PyNcclCommunicator,
    *,
    base_step: int,
    step: int,
) -> ShardedBF16DeltaUpdate:
    shard_metadata = pickle.loads(receive_bytes(communicator))
    if not isinstance(shard_metadata, tuple) or not shard_metadata:
        raise RuntimeError("invalid distributed BF16 delta metadata")
    shards: list[BF16DeltaUpdate] = []
    for rank, item in enumerate(shard_metadata):
        if not isinstance(item, tuple) or len(item) != 3:
            raise RuntimeError(f"invalid BF16 delta metadata for trainer shard {rank}")
        tensors, frames, compressed_nbytes = item
        if not isinstance(tensors, tuple) or not all(isinstance(tensor, DeltaTensorMetadata) for tensor in tensors):
            raise RuntimeError(f"invalid BF16 delta tensor metadata for trainer shard {rank}")
        if not isinstance(frames, tuple) or not all(isinstance(frame, CompressedDeltaFrame) for frame in frames):
            raise RuntimeError(f"invalid BF16 delta frame metadata for trainer shard {rank}")
        if compressed_nbytes != packed_delta_nbytes(frames):
            raise RuntimeError(
                f"trainer shard {rank} metadata describes {packed_delta_nbytes(frames)} compressed bytes; "
                f"sender announced {compressed_nbytes}"
            )
        payload = torch.empty(compressed_nbytes, dtype=torch.uint8, device=communicator.device)
        _receive_tensor(payload, communicator)
        shards.append(
            BF16DeltaUpdate(
                base_step=base_step,
                step=step,
                tensors=tensors,
                frames=frames,
                payload=payload,
            )
        )
    update = ShardedBF16DeltaUpdate(
        base_step=base_step,
        step=step,
        shards=tuple(shards),
    )
    try:
        validate_sharded_delta_update(update)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    return update


_QWEN_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")


def qwen_layer_index(name: str) -> int:
    match = _QWEN_LAYER_PATTERN.match(name)
    return int(match.group(1)) if match else -1


def receive_state_dict(
    communicator: PyNcclCommunicator,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Stream tensors in a state dict broadcasted over NCCL."""
    size_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    _receive_tensor(size_tensor, communicator)
    state_tensor = torch.empty(cast(int, size_tensor.item()), dtype=torch.uint8).to(communicator.device)
    _receive_tensor(state_tensor, communicator)

    state_cpu = state_tensor.cpu()
    metadata = pickle.loads(bytes(state_cpu.numpy()))

    # Receive concatenated tensors per dtype and split them back
    for dtype, tensor_info_list in metadata.items():
        # Receive concatenated tensor for this dtype
        total_elements = sum(numel for _, _, numel in tensor_info_list)
        concatenated = torch.empty(total_elements, dtype=dtype, device=communicator.device)
        _receive_tensor(concatenated, communicator)

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
    ):
        logger.info(f"Initializing NCCL broadcast receiver ({host}:{port}, rank={rank}, world_size={world_size})")
        disable_nccl_p2p_if_unavailable()

        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout)
        self.communicator = PyNcclCommunicator(pg, device=device)
        self.delta_mode = delta_mode
        self.current_step: int | None = None
        self.delta_codec = NvcompLZ4Codec(device) if delta_mode == "bf16_xor" else None

    @torch.no_grad()
    def receive_state_dict(self):
        """Receives the state dict of a model from the trainer master rank using NCCL communicator."""
        logger.info("Receiving weights from trainer")
        num_state_dict_to_receive = receive_integer(self.communicator)
        logger.info(f"Receiving {num_state_dict_to_receive} layer state dicts")
        for layer_id in range(num_state_dict_to_receive):
            logger.info(f"Receiving state dict {layer_id + 1}/{num_state_dict_to_receive}")
            for key, value in receive_state_dict(self.communicator):
                yield key, value

    def receive_update_header(self) -> WeightUpdateHeader | None:
        return receive_update_header(self.communicator)


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
        )

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    def update_weights_from_path(self, weight_dir: str) -> None:
        """Update weights with the nccl communicator."""
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        del weight_dir
        header = self.nccl_broadcast_receiver.receive_update_header()
        if header is not None and header.kind == WeightUpdateKind.BF16_XOR:
            if header.base_step != self.nccl_broadcast_receiver.current_step:
                raise RuntimeError(
                    f"cannot apply BF16 delta for step {header.step}: base step {header.base_step} "
                    f"does not match resident step {self.nccl_broadcast_receiver.current_step}"
                )
            update = receive_compressed_delta(
                self.nccl_broadcast_receiver.communicator,
                base_step=header.base_step,
                step=header.step,
            )
            logger.info(
                "Received nvCOMP LZ4 BF16 XOR update: %d tensors in %d frames from %d trainer shards, "
                "%.2f MiB compressed from %.2f MiB (%.1fx)",
                update.tensor_count,
                update.frame_count,
                len(update.shards),
                update.compressed_nbytes / (1024 * 1024),
                update.uncompressed_nbytes / (1024 * 1024),
                update.uncompressed_nbytes / update.compressed_nbytes,
            )
            codec = self.nccl_broadcast_receiver.delta_codec
            assert codec is not None
            frame_payloads = [list(shard.frame_payloads()) for shard in update.shards]
            for frame_index in range(update.frame_count):
                decoded_shards: list[list[tuple[str, torch.Tensor]]] = []
                metadata_shards: list[tuple[DeltaTensorMetadata, ...]] = []
                for shard_index, shard in enumerate(update.shards):
                    frame = shard.frames[frame_index]
                    decoded = decode_delta_tensors(
                        codec,
                        shard.tensors,
                        [frame],
                        [frame_payloads[shard_index][frame_index]],
                    )
                    decoded_shards.append(decoded)
                    metadata_shards.append(
                        shard.tensors[frame.first_tensor_index : frame.first_tensor_index + frame.tensor_count]
                    )
                reference_metadata = metadata_shards[0]
                group_start = 0
                while group_start < len(reference_metadata):
                    layer_index = qwen_layer_index(reference_metadata[group_start].name)
                    group_end = group_start + 1
                    while (
                        group_end < len(reference_metadata)
                        and qwen_layer_index(reference_metadata[group_end].name) == layer_index
                    ):
                        group_end += 1
                    decoded_group = reconstruct_delta_tensors(
                        [values[group_start:group_end] for values in decoded_shards],
                        [metadata[group_start:group_end] for metadata in metadata_shards],
                    )
                    apply_qwen3_bf16_source_deltas_(
                        model,
                        dict(decoded_group),
                        layer_index=layer_index,
                    )
                    del decoded_group
                    group_start = group_end
                del decoded_shards, metadata_shards
            torch.cuda.synchronize(self.device)
            optimizer_to_apply_ms = (
                (time.perf_counter_ns() - header.optimizer_start_ns) / 1_000_000 if header.optimizer_start_ns else 0.0
            )
            if header.optimizer_start_ns:
                logger.info(
                    "Policy v%d optimizer-start to inference-apply: %.2f ms",
                    header.step,
                    optimizer_to_apply_ms,
                )
            self.nccl_broadcast_receiver.current_step = header.step
            return

        state_iter = self.nccl_broadcast_receiver.receive_state_dict()
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
        torch.cuda.synchronize(self.device)
        optimizer_to_apply_ms = (
            (time.perf_counter_ns() - header.optimizer_start_ns) / 1_000_000
            if header is not None and header.optimizer_start_ns
            else 0.0
        )
        if header is not None and header.optimizer_start_ns:
            logger.info(
                "Policy v%d optimizer-start to inference-apply: %.2f ms",
                header.step,
                optimizer_to_apply_ms,
            )
        if header is not None:
            self.nccl_broadcast_receiver.current_step = header.step
