from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from math import prod
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor

from prime_rl.weight_sync.profiling import WeightSyncMetrics, cuda_event_pair

NCCL_DELTA_PROTOCOL_MAGIC = 0x50524C44  # "PRLD"
NCCL_DELTA_PROTOCOL_VERSION = 7
NVCOMP_FRAME_ALIGNMENT = 256
NVCOMP_PIPELINE_DEPTH = 6


class WeightUpdateKind(IntEnum):
    FULL = 0
    BF16_XOR = 1


@dataclass(frozen=True)
class WeightUpdateHeader:
    kind: WeightUpdateKind
    base_step: int
    step: int
    optimizer_start_ns: int = 0


def encode_weight_update_header(header: WeightUpdateHeader, *, device: torch.device | str | int) -> Tensor:
    if header.kind == WeightUpdateKind.BF16_XOR and header.step != header.base_step + 1:
        raise ValueError(
            f"BF16 XOR header must name consecutive versions: base_step={header.base_step}, step={header.step}"
        )
    if header.kind == WeightUpdateKind.FULL and header.base_step != -1:
        raise ValueError(f"full update header must use base_step=-1, got {header.base_step}")
    if header.optimizer_start_ns < 0:
        raise ValueError(f"optimizer_start_ns must be non-negative, got {header.optimizer_start_ns}")
    return torch.tensor(
        [
            NCCL_DELTA_PROTOCOL_MAGIC,
            NCCL_DELTA_PROTOCOL_VERSION,
            int(header.kind),
            header.base_step,
            header.step,
            header.optimizer_start_ns,
        ],
        dtype=torch.long,
        device=device,
    )


def decode_weight_update_header(values: Tensor) -> WeightUpdateHeader:
    if values.dtype != torch.long or values.shape != (6,):
        raise ValueError(f"invalid NCCL update header tensor: dtype={values.dtype}, shape={tuple(values.shape)}")
    magic, version, kind, base_step, step, optimizer_start_ns = (int(value) for value in values.tolist())
    if magic != NCCL_DELTA_PROTOCOL_MAGIC:
        raise ValueError(f"invalid NCCL delta protocol magic: {magic:#x}")
    if version != NCCL_DELTA_PROTOCOL_VERSION:
        raise ValueError(f"unsupported NCCL delta protocol version {version}; expected {NCCL_DELTA_PROTOCOL_VERSION}")
    try:
        update_kind = WeightUpdateKind(kind)
    except ValueError as error:
        raise ValueError(f"unsupported NCCL weight update kind: {kind}") from error
    header = WeightUpdateHeader(update_kind, base_step, step, optimizer_start_ns)
    encode_weight_update_header(header, device="cpu")
    return header


@dataclass(frozen=True)
class DeltaTensorMetadata:
    name: str
    shape: tuple[int, ...]
    nbytes: int
    global_shape: tuple[int, ...] | None = None
    shard_dim: int | None = None
    shard_index: int = 0
    shard_count: int = 1

    @property
    def resolved_global_shape(self) -> tuple[int, ...]:
        return self.shape if self.global_shape is None else self.global_shape

    @property
    def local_shape(self) -> tuple[int, ...]:
        return self.shape


@dataclass(frozen=True)
class CompressedDeltaFrame:
    first_tensor_index: int
    tensor_count: int
    uncompressed_nbytes: int
    compressed_nbytes: int


@dataclass(frozen=True)
class NvcompEncodeEvents:
    encode: tuple[torch.cuda.Event, torch.cuda.Event]
    clone: tuple[torch.cuda.Event, torch.cuda.Event]


@dataclass(frozen=True)
class PendingNvcompEncode:
    output_buffers: tuple[Tensor, ...]
    encoded: tuple[Any, ...]
    events: NvcompEncodeEvents | None


@dataclass(frozen=True)
class PendingDeltaFrame:
    first_tensor_index: int
    tensor_count: int
    uncompressed_nbytes: int
    encode: PendingNvcompEncode


@dataclass(frozen=True)
class BF16DeltaUpdate:
    """A GPU-resident nvCOMP LZ4 source-layout XOR update."""

    base_step: int
    step: int
    tensors: tuple[DeltaTensorMetadata, ...]
    frames: tuple[CompressedDeltaFrame, ...]
    payload: Tensor
    profile: WeightSyncMetrics | None = field(default=None, compare=False, repr=False)

    @property
    def uncompressed_nbytes(self) -> int:
        return sum(frame.uncompressed_nbytes for frame in self.frames)

    @property
    def compressed_nbytes(self) -> int:
        return self.payload.numel() * self.payload.element_size()

    def frame_payloads(self) -> Iterator[Tensor]:
        offset = 0
        for frame in self.frames:
            offset = _align_up(offset, NVCOMP_FRAME_ALIGNMENT)
            yield self.payload.narrow(0, offset, frame.compressed_nbytes)
            offset += frame.compressed_nbytes
        if offset != self.compressed_nbytes:
            raise ValueError(
                f"compressed frame metadata describes {offset} bytes but payload has {self.compressed_nbytes}"
            )


@dataclass(frozen=True)
class ShardedBF16DeltaUpdate:
    """Rank-local compressed FSDP shards for one logical policy update."""

    base_step: int
    step: int
    shards: tuple[BF16DeltaUpdate, ...]
    profile: WeightSyncMetrics | None = field(default=None, compare=False, repr=False)

    @property
    def uncompressed_nbytes(self) -> int:
        return sum(shard.uncompressed_nbytes for shard in self.shards)

    @property
    def compressed_nbytes(self) -> int:
        return sum(shard.compressed_nbytes for shard in self.shards)

    @property
    def tensor_count(self) -> int:
        return len(self.shards[0].tensors) if self.shards else 0

    @property
    def frame_count(self) -> int:
        return len(self.shards[0].frames) if self.shards else 0


def local_tensor(tensor: Tensor) -> Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _load_nvcomp():
    try:
        from nvidia import nvcomp
    except ImportError as error:
        raise RuntimeError(
            "BF16 XOR weight synchronization requires nvidia-nvcomp-cu12 on a CUDA-capable Linux host"
        ) from error
    return nvcomp


class NvcompLZ4Codec:
    """nvCOMP LZ4 bound to one CUDA device and a dedicated stream."""

    def __init__(self, device: torch.device | str | int) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"nvCOMP LZ4 requires a CUDA device, got {self.device}")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.nvcomp = _load_nvcomp()
        self.stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.device(self.device):
            self.codec = self.nvcomp.Codec(
                algorithm="LZ4",
                device_id=self.device.index,
                cuda_stream=self.stream.cuda_stream,
            )
        self._compression_configs: dict[tuple[int, ...], Any] = {}
        self._decompression_configs: dict[tuple[int, ...], Any] = {}

    def encode(
        self,
        values: Sequence[Tensor],
        *,
        profile: WeightSyncMetrics | None = None,
    ) -> tuple[list[Tensor], NvcompEncodeEvents | None]:
        if not values:
            return [], None
        pending = self.submit_encode(values, profile=profile)
        return self.finalize_encode(pending, profile=profile)

    def submit_encode(
        self,
        values: Sequence[Tensor],
        *,
        profile: WeightSyncMetrics | None = None,
    ) -> PendingNvcompEncode:
        if not values:
            raise ValueError("cannot submit an empty nvCOMP encode batch")
        byte_views = [self._byte_view(value) for value in values]
        sizes = tuple(value.numel() for value in byte_views)
        producer_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(producer_stream)
        for value in values:
            value.record_stream(self.stream)

        events = (
            NvcompEncodeEvents(
                encode=cuda_event_pair(),
                clone=cuda_event_pair(),
            )
            if profile is not None
            else None
        )
        with torch.cuda.stream(self.stream):
            sources = self.nvcomp.as_arrays(byte_views, cuda_stream=self.stream.cuda_stream)
            output_alloc_start = time.perf_counter()
            output_buffers = [
                torch.empty(
                    self.codec.get_max_comp_buffer_size(source),
                    dtype=torch.uint8,
                    device=self.device,
                )
                for source in sources
            ]
            if profile is not None:
                profile.nvcomp_output_alloc_wall_ms += (time.perf_counter() - output_alloc_start) * 1000
            destinations = self.nvcomp.as_arrays(output_buffers, cuda_stream=self.stream.cuda_stream)
            compression_config = self._compression_config(sizes)
            if events is not None:
                events.encode[0].record(self.stream)
            encode_call_start = time.perf_counter()
            encoded = self.codec.encode(
                sources,
                out=destinations,
                compression_config=compression_config,
            )
            if events is not None:
                events.encode[1].record(self.stream)
            if profile is not None:
                profile.nvcomp_encode_call_wall_ms += (time.perf_counter() - encode_call_start) * 1000

        return PendingNvcompEncode(
            output_buffers=tuple(output_buffers),
            encoded=tuple(encoded),
            events=events,
        )

    def finalize_encode(
        self,
        pending: PendingNvcompEncode,
        *,
        profile: WeightSyncMetrics | None = None,
    ) -> tuple[list[Tensor], NvcompEncodeEvents | None]:
        events = pending.events
        with torch.cuda.stream(self.stream):
            size_read_start = time.perf_counter()
            compressed_sizes = [item.buffer_size for item in pending.encoded]
            if profile is not None:
                profile.nvcomp_buffer_size_read_wall_ms += (time.perf_counter() - size_read_start) * 1000
            # Keep Torch as the allocator/owner. Converting nvCOMP-owned output
            # arrays through DLPack and then dropping their wrappers can leave
            # Torch tensors referring to released storage. Clone only the valid
            # compressed range so the worst-case output buffers remain bounded
            # to the current Adam bucket instead of accumulating for the model.
            clone_enqueue_start = time.perf_counter()
            if events is not None:
                events.clone[0].record(self.stream)
            payloads = [
                buffer.narrow(0, 0, compressed_size).clone()
                for buffer, compressed_size in zip(pending.output_buffers, compressed_sizes, strict=True)
            ]
            if events is not None:
                events.clone[1].record(self.stream)
            if profile is not None:
                profile.nvcomp_clone_enqueue_wall_ms += (time.perf_counter() - clone_enqueue_start) * 1000
        return payloads, events

    def pack(
        self,
        payloads: Sequence[Tensor],
        *,
        profile: bool = False,
    ) -> tuple[Tensor, tuple[torch.cuda.Event, torch.cuda.Event] | None]:
        if not payloads:
            return torch.empty(0, dtype=torch.uint8, device=self.device), None
        events = cuda_event_pair() if profile else None
        with torch.cuda.stream(self.stream):
            if events is not None:
                events[0].record(self.stream)
            packed = torch.empty(_packed_nbytes(payloads), dtype=torch.uint8, device=self.device)
            offset = 0
            for payload in payloads:
                offset = _align_up(offset, NVCOMP_FRAME_ALIGNMENT)
                packed.narrow(0, offset, payload.numel()).copy_(payload)
                offset += payload.numel()
            if events is not None:
                events[1].record(self.stream)
        return packed, events

    def decode(
        self,
        payloads: Sequence[Tensor],
        uncompressed_nbytes: Sequence[int],
        *,
        profile: bool = False,
    ) -> tuple[list[Tensor], tuple[torch.cuda.Event, torch.cuda.Event] | None]:
        if not payloads:
            return [], None
        sizes = tuple(uncompressed_nbytes)
        if len(payloads) != len(sizes):
            raise ValueError(f"received {len(payloads)} compressed frames for {len(sizes)} output sizes")
        consumer_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(consumer_stream)
        for payload in payloads:
            payload.record_stream(self.stream)

        events = cuda_event_pair() if profile else None
        with torch.cuda.stream(self.stream):
            outputs = [torch.empty(size, dtype=torch.uint8, device=self.device) for size in sizes]
            sources = self.nvcomp.as_arrays(payloads, cuda_stream=self.stream.cuda_stream)
            destinations = self.nvcomp.as_arrays(outputs, cuda_stream=self.stream.cuda_stream)
            decompression_config = self._decompression_config(sizes, sources)
            if events is not None:
                events[0].record(self.stream)
            self.codec.decode(
                sources,
                out=destinations,
                decompression_config=decompression_config,
            )
            if events is not None:
                events[1].record(self.stream)
        consumer_stream.wait_stream(self.stream)
        return outputs, events

    def synchronize(self) -> None:
        self.stream.synchronize()

    def _compression_config(self, sizes: tuple[int, ...]):
        config = self._compression_configs.get(sizes)
        if config is None:
            config = self.codec.compression_config(list(sizes))
            self._compression_configs[sizes] = config
        return config

    def _decompression_config(self, sizes: tuple[int, ...], sources):
        config = self._decompression_configs.get(sizes)
        if config is None:
            # CompressionConfig-derived decode configs are documented for the
            # same Codec object. Trainer and inference necessarily own separate
            # codecs, so parse the first received batch and cache that config for
            # subsequent policy versions with the same tensor shapes.
            config = self.codec.decompression_config(sources)
            self._decompression_configs[sizes] = config
        return config

    def _byte_view(self, value: Tensor) -> Tensor:
        if value.device != self.device:
            raise ValueError(f"nvCOMP tensor is on {value.device}; expected {self.device}")
        if not value.is_contiguous():
            raise ValueError(f"nvCOMP tensor is non-contiguous with stride {value.stride()}")
        return value.view(torch.uint8).reshape(-1)


class BF16DeltaEncoder:
    """Compress contiguous BF16 XOR parameter buckets through nvCOMP LZ4 on GPU."""

    def __init__(
        self,
        *,
        base_step: int,
        step: int,
        codec: NvcompLZ4Codec,
        profile: WeightSyncMetrics | None = None,
    ) -> None:
        if step != base_step + 1:
            raise ValueError(f"BF16 delta updates must be consecutive: base_step={base_step}, step={step}")
        self.base_step = base_step
        self.step = step
        self.codec = codec
        self.profile = profile
        self._tensors: list[DeltaTensorMetadata] = []
        self._frames: list[CompressedDeltaFrame] = []
        self._payloads: list[Tensor] = []
        self._pending: deque[PendingDeltaFrame] = deque()
        self._compression_events: list[NvcompEncodeEvents] = []
        self._compression_started_at: float | None = None
        self._finished = False

    def append(self, name: str, delta: Tensor) -> None:
        self.append_batch([(name, delta)])

    def append_batch(self, values: Sequence[tuple[str, Tensor]]) -> None:
        if self._finished:
            raise RuntimeError("cannot append to a finished BF16 delta encoder")
        normalized = self._normalize_values(values)
        if not normalized:
            return
        device = normalized[0][1].device
        bucket = torch.empty(
            sum(value.numel() for _, value in normalized),
            dtype=torch.bfloat16,
            device=device,
        )
        bucket_values: list[tuple[str, Tensor]] = []
        offset = 0
        for name, value in normalized:
            destination = bucket.narrow(0, offset, value.numel()).view(value.shape)
            destination.copy_(value)
            bucket_values.append((name, destination))
            offset += value.numel()
        self.append_bucket(bucket_values, bucket)

    def append_bucket(self, values: Sequence[tuple[str, Tensor]], bucket: Tensor) -> None:
        self.append_sharded_bucket(values, bucket)

    def append_sharded_bucket(
        self,
        values: Sequence[tuple[str, Tensor]],
        bucket: Tensor,
        *,
        global_shapes: Sequence[tuple[int, ...]] | None = None,
        shard_descriptors: Sequence[tuple[int | None, int, int]] | None = None,
        shard_dim: int | None = None,
        shard_index: int = 0,
        shard_count: int = 1,
    ) -> None:
        if self._finished:
            raise RuntimeError("cannot append to a finished BF16 delta encoder")
        normalized = self._normalize_values(values)
        if not normalized:
            return
        bucket = local_tensor(bucket.detach())
        if bucket.dtype != torch.bfloat16 or bucket.device.type != "cuda" or not bucket.is_contiguous():
            raise ValueError(
                "BF16 delta bucket must be a contiguous CUDA BF16 tensor; "
                f"got dtype={bucket.dtype}, device={bucket.device}, stride={bucket.stride()}"
            )
        expected_elements = sum(value.numel() for _, value in normalized)
        if bucket.numel() != expected_elements:
            raise ValueError(f"BF16 delta bucket has {bucket.numel()} elements; tensors require {expected_elements}")
        offset = 0
        for name, value in normalized:
            if value.device != bucket.device:
                raise ValueError(f"{name} is on {value.device}; bucket is on {bucket.device}")
            expected_pointer = bucket.data_ptr() + offset * bucket.element_size()
            if value.data_ptr() != expected_pointer:
                raise ValueError(f"{name} is not the expected contiguous view into its BF16 delta bucket")
            offset += value.numel()

        if global_shapes is None:
            global_shapes = [tuple(value.shape) for _, value in normalized]
        if len(global_shapes) != len(normalized):
            raise ValueError(
                f"received {len(global_shapes)} global shapes for {len(normalized)} BF16 delta tensors"
            )
        if shard_descriptors is None:
            shard_descriptors = [(shard_dim, shard_index, shard_count)] * len(normalized)
        if len(shard_descriptors) != len(normalized):
            raise ValueError(
                f"received {len(shard_descriptors)} shard descriptors for {len(normalized)} BF16 delta tensors"
            )
        metadata = tuple(
            DeltaTensorMetadata(
                name=name,
                shape=tuple(value.shape),
                nbytes=value.numel() * value.element_size(),
                global_shape=tuple(global_shape),
                shard_dim=descriptor[0],
                shard_index=descriptor[1],
                shard_count=descriptor[2],
            )
            for (name, value), global_shape, descriptor in zip(
                normalized,
                global_shapes,
                shard_descriptors,
                strict=True,
            )
        )
        first_tensor_index = len(self._tensors)
        self._tensors.extend(metadata)
        if self.profile is not None:
            largest_metadata_bytes = max(item.nbytes for item in metadata)
            self.profile.largest_tensor_bytes = max(
                self.profile.largest_tensor_bytes,
                largest_metadata_bytes,
            )

        if self._compression_started_at is None:
            self._compression_started_at = time.perf_counter()
        self._pending.append(
            PendingDeltaFrame(
                first_tensor_index=first_tensor_index,
                tensor_count=len(metadata),
                uncompressed_nbytes=bucket.numel() * bucket.element_size(),
                encode=self.codec.submit_encode([bucket], profile=self.profile),
            )
        )
        if self.profile is not None:
            self.profile.nvcomp_batch_count += 1
            self.profile.nvcomp_peak_pending_batches = max(
                self.profile.nvcomp_peak_pending_batches,
                len(self._pending),
            )
        if len(self._pending) >= NVCOMP_PIPELINE_DEPTH:
            self._finalize_oldest()

    def finish(self) -> BF16DeltaUpdate:
        if self._finished:
            raise RuntimeError("BF16 delta encoder is already finished")
        self._finished = True
        while self._pending:
            self._finalize_oldest()
        packed, pack_events = self.codec.pack(self._payloads, profile=self.profile is not None)
        self.codec.synchronize()
        update = BF16DeltaUpdate(
            base_step=self.base_step,
            step=self.step,
            tensors=tuple(self._tensors),
            frames=tuple(self._frames),
            payload=packed,
            profile=self.profile,
        )
        validate_delta_update(update)
        if self.profile is not None:
            encode_gpu_ms = _sum_completed_events(
                events.encode for events in self._compression_events
            )
            clone_gpu_ms = _sum_completed_events(
                events.clone for events in self._compression_events
            )
            self.profile.nvcomp_encode_gpu_ms += encode_gpu_ms
            self.profile.nvcomp_clone_gpu_ms += clone_gpu_ms
            self.profile.nvcomp_compress_gpu_ms += encode_gpu_ms + clone_gpu_ms
            if pack_events is not None:
                self.profile.delta_gpu_pack_ms += pack_events[0].elapsed_time(pack_events[1])
            if self._compression_started_at is not None:
                self.profile.nvcomp_compress_wall_ms += (time.perf_counter() - self._compression_started_at) * 1000
            self.profile.raw_bytes = update.uncompressed_nbytes
            self.profile.compressed_bytes = update.compressed_nbytes
            self.profile.tensor_count = len(update.tensors)
            self.profile.frame_count = len(update.frames)
        return update

    def abort(self) -> None:
        if not self._finished:
            self._finished = True
            self.codec.synchronize()

    def _finalize_oldest(self) -> None:
        pending = self._pending.popleft()
        payloads, events = self.codec.finalize_encode(pending.encode, profile=self.profile)
        if len(payloads) != 1:
            raise RuntimeError(f"nvCOMP returned {len(payloads)} frames for one BF16 delta bucket")
        payload = payloads[0]
        self._frames.append(
            CompressedDeltaFrame(
                first_tensor_index=pending.first_tensor_index,
                tensor_count=pending.tensor_count,
                uncompressed_nbytes=pending.uncompressed_nbytes,
                compressed_nbytes=payload.numel() * payload.element_size(),
            )
        )
        self._payloads.append(payload)
        if events is not None:
            self._compression_events.append(events)

    @staticmethod
    def _normalize_values(values: Sequence[tuple[str, Tensor]]) -> list[tuple[str, Tensor]]:
        normalized: list[tuple[str, Tensor]] = []
        for name, delta in values:
            value = local_tensor(delta.detach())
            if value.dtype != torch.bfloat16:
                raise TypeError(f"{name} has dtype {value.dtype}; BF16 delta mode requires BF16 parameters")
            if value.device.type != "cuda":
                raise ValueError(f"{name} is on {value.device}; BF16 delta mode requires CUDA tensors")
            if not value.is_contiguous():
                raise ValueError(f"{name} is non-contiguous with stride {value.stride()}")
            if value.numel() == 0:
                raise ValueError(f"{name} is empty; BF16 delta tensors must contain at least one element")
            normalized.append((name, value))
        return normalized


def decode_delta_tensors(
    codec: NvcompLZ4Codec,
    tensors: Sequence[DeltaTensorMetadata],
    frames: Sequence[CompressedDeltaFrame],
    payloads: Sequence[Tensor],
    *,
    profile: bool = False,
) -> tuple[list[tuple[str, Tensor]], tuple[torch.cuda.Event, torch.cuda.Event] | None]:
    if len(frames) != len(payloads):
        raise ValueError(
            f"delta decode length mismatch: frames={len(frames)}, payloads={len(payloads)}"
        )
    decoded, events = codec.decode(
        payloads,
        [frame.uncompressed_nbytes for frame in frames],
        profile=profile,
    )
    values: list[tuple[str, Tensor]] = []
    for frame, raw in zip(frames, decoded, strict=True):
        offset = 0
        frame_tensors = tensors[frame.first_tensor_index : frame.first_tensor_index + frame.tensor_count]
        for metadata in frame_tensors:
            tensor_bytes = raw.narrow(0, offset, metadata.nbytes)
            values.append((metadata.name, tensor_bytes.view(torch.bfloat16).view(metadata.shape)))
            offset += metadata.nbytes
        if offset != frame.uncompressed_nbytes:
            raise ValueError(
                f"decoded BF16 delta frame describes {offset} tensor bytes; expected {frame.uncompressed_nbytes}"
            )
    return values, events


def validate_delta_update(update: BF16DeltaUpdate) -> None:
    valid_payload = (
        update.payload.dtype == torch.uint8 and update.payload.device.type == "cuda" and update.payload.is_contiguous()
    )
    if not valid_payload:
        raise ValueError(
            "BF16 delta payload must be a contiguous CUDA uint8 tensor; "
            f"got dtype={update.payload.dtype}, device={update.payload.device}, stride={update.payload.stride()}"
        )
    if update.payload.data_ptr() % NVCOMP_FRAME_ALIGNMENT != 0:
        raise ValueError(
            f"BF16 delta payload address {update.payload.data_ptr():#x} is not "
            f"{NVCOMP_FRAME_ALIGNMENT}-byte aligned"
        )
    names: set[str] = set()
    for tensor in update.tensors:
        if not tensor.name or tensor.name in names:
            raise ValueError(f"invalid or duplicate BF16 delta tensor name: {tensor.name!r}")
        names.add(tensor.name)
        expected = prod(tensor.shape) * 2
        if tensor.nbytes != expected:
            raise ValueError(f"{tensor.name} metadata has {tensor.nbytes} bytes; shape requires {expected}")
        global_shape = tensor.resolved_global_shape
        if len(global_shape) != len(tensor.shape):
            raise ValueError(
                f"{tensor.name} local rank {len(tensor.shape)} does not match global rank {len(global_shape)}"
            )
        if tensor.shard_count <= 0 or not 0 <= tensor.shard_index < tensor.shard_count:
            raise ValueError(
                f"{tensor.name} has invalid shard {tensor.shard_index}/{tensor.shard_count}"
            )
        if tensor.shard_count == 1:
            if tensor.shard_dim is not None or tensor.shape != global_shape:
                raise ValueError(f"unsharded tensor {tensor.name} must have identical local and global shapes")
        elif tensor.shard_dim != 0:
            raise ValueError(f"{tensor.name} uses unsupported shard dimension {tensor.shard_dim}; expected 0")
    next_tensor_index = 0
    for frame_index, frame in enumerate(update.frames):
        if frame.first_tensor_index != next_tensor_index:
            raise ValueError(
                f"BF16 delta frame {frame_index} starts at tensor {frame.first_tensor_index}; "
                f"expected {next_tensor_index}"
            )
        if frame.tensor_count <= 0 or frame.first_tensor_index + frame.tensor_count > len(update.tensors):
            raise ValueError(f"BF16 delta frame {frame_index} has invalid tensor count {frame.tensor_count}")
        frame_tensors = update.tensors[frame.first_tensor_index : frame.first_tensor_index + frame.tensor_count]
        expected_uncompressed = sum(tensor.nbytes for tensor in frame_tensors)
        if frame.uncompressed_nbytes != expected_uncompressed:
            raise ValueError(
                f"BF16 delta frame {frame_index} has {frame.uncompressed_nbytes} raw bytes; "
                f"its tensors require {expected_uncompressed}"
            )
        if frame.compressed_nbytes <= 0:
            raise ValueError(f"BF16 delta frame {frame_index} has invalid compressed size {frame.compressed_nbytes}")
        next_tensor_index += frame.tensor_count
    if next_tensor_index != len(update.tensors):
        raise ValueError(f"BF16 delta frames cover {next_tensor_index} of {len(update.tensors)} tensors")
    packed_bytes = packed_delta_nbytes(update.frames)
    if packed_bytes != update.compressed_nbytes:
        raise ValueError(
            f"aligned compressed frame metadata describes {packed_bytes} bytes "
            f"but payload has {update.compressed_nbytes}"
        )


def validate_sharded_delta_update(update: ShardedBF16DeltaUpdate) -> None:
    if not update.shards:
        raise ValueError("distributed BF16 delta update has no trainer shards")
    shard_count = len(update.shards)
    reference = update.shards[0]
    for rank, shard in enumerate(update.shards):
        validate_delta_update(shard)
        if shard.base_step != update.base_step or shard.step != update.step:
            raise ValueError(
                f"trainer shard {rank} names policy {shard.base_step}->{shard.step}; "
                f"expected {update.base_step}->{update.step}"
            )
        if len(shard.tensors) != len(reference.tensors) or len(shard.frames) != len(reference.frames):
            raise ValueError(f"trainer shard {rank} has an incompatible tensor/frame manifest")
        for tensor_index, (candidate, expected) in enumerate(zip(shard.tensors, reference.tensors, strict=True)):
            if candidate.name != expected.name or candidate.resolved_global_shape != expected.resolved_global_shape:
                raise ValueError(f"trainer shard {rank} tensor {tensor_index} does not match the rank-0 manifest")
            if expected.shard_count == 1:
                if (
                    candidate.shard_count != 1
                    or candidate.shard_index != 0
                    or candidate.shard_dim is not None
                    or candidate.shape != candidate.resolved_global_shape
                ):
                    raise ValueError(f"trainer shard {rank} has incompatible unsharded metadata for {candidate.name}")
            elif candidate.shard_index != rank or candidate.shard_count != shard_count:
                raise ValueError(
                    f"{candidate.name} identifies shard {candidate.shard_index}/{candidate.shard_count}; "
                    f"expected {rank}/{shard_count}"
                )
        for frame_index, (candidate, expected) in enumerate(zip(shard.frames, reference.frames, strict=True)):
            if (candidate.first_tensor_index, candidate.tensor_count) != (
                expected.first_tensor_index,
                expected.tensor_count,
            ):
                raise ValueError(f"trainer shard {rank} frame {frame_index} has an incompatible tensor manifest")

    for tensor_index in range(len(reference.tensors)):
        pieces = [shard.tensors[tensor_index] for shard in update.shards]
        global_shape = pieces[0].resolved_global_shape
        if pieces[0].shard_count == 1:
            if any(piece.shard_count != 1 or piece.shape != global_shape for piece in pieces):
                raise ValueError(f"{pieces[0].name} has inconsistent unsharded trainer copies")
            continue
        if any(piece.shard_dim != 0 for piece in pieces):
            raise ValueError(f"{pieces[0].name} is not sharded along source dimension 0")
        if any(piece.shape[1:] != global_shape[1:] for piece in pieces):
            raise ValueError(f"{pieces[0].name} shard trailing dimensions do not match its global shape")
        reconstructed = (sum(piece.shape[0] for piece in pieces), *global_shape[1:])
        if reconstructed != global_shape:
            raise ValueError(
                f"{pieces[0].name} shards reconstruct shape {reconstructed}; expected {global_shape}"
            )


def reconstruct_delta_tensors(
    shard_values: Sequence[Sequence[tuple[str, Tensor]]],
    shard_metadata: Sequence[Sequence[DeltaTensorMetadata]],
) -> list[tuple[str, Tensor]]:
    """Reconstruct full source-layout tensors from rank-ordered dimension-0 shards."""
    if not shard_values or len(shard_values) != len(shard_metadata):
        raise ValueError("BF16 delta reconstruction requires matching non-empty values and metadata")
    tensor_count = len(shard_values[0])
    if any(len(values) != tensor_count for values in shard_values) or any(
        len(metadata) != tensor_count for metadata in shard_metadata
    ):
        raise ValueError("BF16 delta shard frames contain different tensor counts")
    reconstructed: list[tuple[str, Tensor]] = []
    for tensor_index in range(tensor_count):
        pieces = [values[tensor_index][1] for values in shard_values]
        metadata = [items[tensor_index] for items in shard_metadata]
        names = [values[tensor_index][0] for values in shard_values]
        if any(name != names[0] for name in names) or any(item.name != names[0] for item in metadata):
            raise ValueError(f"BF16 delta shard tensor {tensor_index} has inconsistent names")
        global_shape = metadata[0].resolved_global_shape
        if any(item.resolved_global_shape != global_shape for item in metadata):
            raise ValueError(f"{names[0]} has inconsistent global shapes")
        if metadata[0].shard_count == 1:
            if any(
                item.shard_count != 1
                or item.shard_index != 0
                or item.shard_dim is not None
                or item.shape != global_shape
                for item in metadata
            ):
                raise ValueError(f"{names[0]} has inconsistent unsharded metadata")
        else:
            expected_indices = list(range(metadata[0].shard_count))
            actual_indices = sorted(item.shard_index for item in metadata)
            if (
                len(metadata) != metadata[0].shard_count
                or actual_indices != expected_indices
                or any(item.shard_count != len(metadata) or item.shard_dim != 0 for item in metadata)
            ):
                raise ValueError(
                    f"{names[0]} has malformed dimension-0 shards: "
                    f"indices={actual_indices}, expected={expected_indices}"
                )
        ordered = sorted(zip(metadata, pieces, strict=True), key=lambda item: item[0].shard_index)
        if ordered[0][0].shard_count == 1:
            value = ordered[0][1]
        else:
            value = torch.cat([piece for _, piece in ordered], dim=0)
        if tuple(value.shape) != global_shape:
            raise ValueError(f"{names[0]} reconstructed shape {tuple(value.shape)}; expected {global_shape}")
        reconstructed.append((names[0], value))
    return reconstructed


def _sum_completed_events(events: Iterable[tuple[torch.cuda.Event, torch.cuda.Event]]) -> float:
    return sum(start.elapsed_time(end) for start, end in events)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def packed_delta_nbytes(frames: Sequence[CompressedDeltaFrame]) -> int:
    """Return the CUDA payload size including inter-frame alignment padding."""
    size = 0
    for frame in frames:
        size = _align_up(size, NVCOMP_FRAME_ALIGNMENT) + frame.compressed_nbytes
    return size


def _packed_nbytes(payloads: Sequence[Tensor]) -> int:
    size = 0
    for payload in payloads:
        size = _align_up(size, NVCOMP_FRAME_ALIGNMENT) + payload.numel()
    return size


__all__ = [
    "BF16DeltaEncoder",
    "BF16DeltaUpdate",
    "ShardedBF16DeltaUpdate",
    "CompressedDeltaFrame",
    "DeltaTensorMetadata",
    "NvcompLZ4Codec",
    "WeightUpdateHeader",
    "WeightUpdateKind",
    "decode_delta_tensors",
    "decode_weight_update_header",
    "encode_weight_update_header",
    "local_tensor",
    "packed_delta_nbytes",
    "reconstruct_delta_tensors",
    "validate_delta_update",
    "validate_sharded_delta_update",
]
