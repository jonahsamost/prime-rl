import pickle
import time
from pathlib import Path
from typing import Callable, Generator, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.configs.trainer import NCCLWeightBroadcastConfig
from prime_rl.trainer.conversion_utils import get_max_layer_num
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.broadcast.nccl_delta import (
    broadcast_compressed_delta,
    gather_compressed_delta_updates,
)
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.utils.client import NCCL_READY_MARKER
from prime_rl.utils.logger import get_logger
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable
from prime_rl.utils.pathing import sync_wait_for_path
from prime_rl.utils.utils import get_broadcast_dir, get_step_path
from prime_rl.utils.vlm import get_layer_prefix
from prime_rl.weight_sync.xor_delta import (
    DeltaUpdate,
    ShardedDeltaUpdate,
    WeightUpdateHeader,
    WeightUpdateKind,
    encode_weight_update_header,
)


def broadcast_integer(
    integer: int,
    communicator: PyNcclCommunicator,
) -> None:
    """Broadcast an integer to a process group using NCCL communicator."""
    integer_tensor = torch.tensor([integer], dtype=torch.long).cuda()
    _broadcast_tensor(integer_tensor, communicator)


def broadcast_update_header(
    header: WeightUpdateHeader,
    communicator: PyNcclCommunicator,
) -> None:
    values = encode_weight_update_header(header, device=communicator.device)
    _broadcast_tensor(values, communicator)


def _broadcast_tensor(
    tensor: Tensor,
    communicator: PyNcclCommunicator,
) -> None:
    communicator.broadcast(tensor, src=0)


def _stage_bytes(
    payload: bytes,
    communicator: PyNcclCommunicator,
) -> Tensor:
    values = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(communicator.device)
    return values


def broadcast_bytes(
    payload: bytes,
    communicator: PyNcclCommunicator,
) -> None:
    size = torch.tensor([len(payload)], dtype=torch.long, device=communicator.device)
    _broadcast_tensor(size, communicator)
    values = _stage_bytes(payload, communicator)
    _broadcast_tensor(values, communicator)


def broadcast_state_dict(
    state_dict: dict[str, Tensor],
    communicator: PyNcclCommunicator,
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

    # Build metadata: for each dtype group, store keys and shapes
    metadata = {}
    for dtype, items in dtype_groups.items():
        metadata[dtype] = [(key, value.shape, value.numel()) for key, value in items]

    # Send metadata
    state = pickle.dumps(metadata)
    size_tensor = torch.tensor([len(state)], dtype=torch.long).cuda()
    _broadcast_tensor(size_tensor, communicator)
    state_tensor = _stage_bytes(state, communicator)
    _broadcast_tensor(state_tensor, communicator)

    # Concatenate and broadcast tensors grouped by dtype
    for dtype, items in dtype_groups.items():
        # Flatten all tensors and concatenate
        flat_tensors = [value.flatten() for _, value in items]
        concatenated = torch.cat(flat_tensors)
        _broadcast_tensor(concatenated, communicator)
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
    ):
        self.logger = get_logger()
        self.world = get_world()
        self.dtype = dtype
        self.quantize_in_weight_transfer = quantize_in_weight_transfer
        self.delta_mode = delta_mode
        self.last_broadcast_step: int | None = None
        self._optimizer_start_ns = 0
        if self.delta_mode != "none" and self.quantize_in_weight_transfer:
            raise ValueError("XOR delta mode is incompatible with quantize_in_weight_transfer")

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

    @torch.no_grad()
    def broadcast_weights(
        self,
        model: nn.Module,
        step: int,
        delta_update: DeltaUpdate | None = None,
    ) -> None:
        """Broadcast the state dict of a model into the inference pool using NCCL."""
        self._broadcast_weights(model, step, delta_update)
        self._optimizer_start_ns = 0

    def set_optimizer_start_ns(self, optimizer_start_ns: int) -> None:
        if optimizer_start_ns <= 0:
            raise ValueError(f"optimizer_start_ns must be positive, got {optimizer_start_ns}")
        self._optimizer_start_ns = optimizer_start_ns

    def _broadcast_weights(
        self,
        model: nn.Module,
        step: int,
        delta_update: DeltaUpdate | None,
    ) -> None:
        sharded_delta: ShardedDeltaUpdate | None = None
        is_delta_update = False
        if self.delta_mode == "xor":
            sharded_delta, is_delta_update = gather_compressed_delta_updates(delta_update)
        if is_delta_update:
            assert delta_update is not None
            if self.delta_mode != "xor":
                raise ValueError("received an XOR delta update while delta mode is disabled")
            if delta_update.step != step:
                raise ValueError(f"delta step {delta_update.step} does not match broadcast step {step}")
            if delta_update.base_step != self.last_broadcast_step:
                raise ValueError(
                    f"delta base step {delta_update.base_step} does not match last broadcast {self.last_broadcast_step}"
                )
            header = WeightUpdateHeader(
                WeightUpdateKind.XOR,
                delta_update.base_step,
                step,
                self._optimizer_start_ns,
            )
        else:
            header = WeightUpdateHeader(WeightUpdateKind.FULL, -1, step, self._optimizer_start_ns)
            state_dict = model.state_dict()

        if self.world.is_master:
            broadcast_update_header(header, self.communicator)

        if is_delta_update:
            if self.world.is_master:
                assert sharded_delta is not None
                broadcast_compressed_delta(
                    sharded_delta,
                    self.communicator,
                    broadcast_tensor=_broadcast_tensor,
                    broadcast_bytes=broadcast_bytes,
                )
            self.last_broadcast_step = step
            return

        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)
        num_state_dict_to_send = num_layers + 1  # we send all layer plus the remaining weights

        if self.world.is_master:
            broadcast_integer(num_state_dict_to_send, self.communicator)

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
                broadcast_state_dict(layer_state_dict, self.communicator)
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
        )

    @torch.no_grad()
    def broadcast_weights(
        self,
        model: nn.Module,
        step: int,
        delta_update: DeltaUpdate | None = None,
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
