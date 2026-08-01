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
NCCL_DELTA_PROTOCOL_VERSION = 4
NVCOMP_FRAME_ALIGNMENT = 256
NVCOMP_PIPELINE_DEPTH = 2


class WeightUpdateKind(IntEnum):
    FULL = 0
    BF16_XOR = 1


@dataclass(frozen=True)
class WeightUpdateHeader:
    kind: WeightUpdateKind
    base_step: int
    step: int


def encode_weight_update_header(header: WeightUpdateHeader, *, device: torch.device | str | int) -> Tensor:
    if header.kind == WeightUpdateKind.BF16_XOR and header.step != header.base_step + 1:
        raise ValueError(
            f"BF16 XOR header must name consecutive versions: base_step={header.base_step}, step={header.step}"
        )
    if header.kind == WeightUpdateKind.FULL and header.base_step != -1:
        raise ValueError(f"full update header must use base_step=-1, got {header.base_step}")
    return torch.tensor(
        [
            NCCL_DELTA_PROTOCOL_MAGIC,
            NCCL_DELTA_PROTOCOL_VERSION,
            int(header.kind),
            header.base_step,
            header.step,
        ],
        dtype=torch.long,
        device=device,
    )


def decode_weight_update_header(values: Tensor) -> WeightUpdateHeader:
    if values.dtype != torch.long or values.shape != (5,):
        raise ValueError(f"invalid NCCL update header tensor: dtype={values.dtype}, shape={tuple(values.shape)}")
    magic, version, kind, base_step, step = (int(value) for value in values.tolist())
    if magic != NCCL_DELTA_PROTOCOL_MAGIC:
        raise ValueError(f"invalid NCCL delta protocol magic: {magic:#x}")
    if version != NCCL_DELTA_PROTOCOL_VERSION:
        raise ValueError(f"unsupported NCCL delta protocol version {version}; expected {NCCL_DELTA_PROTOCOL_VERSION}")
    try:
        update_kind = WeightUpdateKind(kind)
    except ValueError as error:
        raise ValueError(f"unsupported NCCL weight update kind: {kind}") from error
    header = WeightUpdateHeader(update_kind, base_step, step)
    encode_weight_update_header(header, device="cpu")
    return header


@dataclass(frozen=True)
class DeltaTensorMetadata:
    name: str
    shape: tuple[int, ...]
    nbytes: int


@dataclass(frozen=True)
class CompressedDeltaFrame:
    uncompressed_nbytes: int
    compressed_nbytes: int


@dataclass(frozen=True)
class NvcompEncodeEvents:
    total: tuple[torch.cuda.Event, torch.cuda.Event]
    encode: tuple[torch.cuda.Event, torch.cuda.Event]
    clone: tuple[torch.cuda.Event, torch.cuda.Event]


@dataclass(frozen=True)
class PendingNvcompEncode:
    output_buffers: tuple[Tensor, ...]
    encoded: tuple[Any, ...]
    events: NvcompEncodeEvents | None


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
                total=cuda_event_pair(),
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
                events.total[0].record(self.stream)
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
            if events is not None:
                events.total[1].record(self.stream)
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
    """Batch BF16 XOR tensors through nvCOMP LZ4 without leaving the GPU."""

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
        self._pending: deque[tuple[tuple[int, ...], PendingNvcompEncode]] = deque()
        self._compression_events: list[NvcompEncodeEvents] = []
        self._compression_started_at: float | None = None
        self._finished = False

    def append(self, name: str, delta: Tensor) -> None:
        self.append_batch([(name, delta)])

    def append_batch(self, values: Sequence[tuple[str, Tensor]]) -> None:
        if self._finished:
            raise RuntimeError("cannot append to a finished BF16 delta encoder")
        if not values:
            return
        deltas: list[Tensor] = []
        for name, delta in values:
            value = local_tensor(delta.detach())
            if value.dtype != torch.bfloat16:
                raise TypeError(f"{name} has dtype {value.dtype}; BF16 delta mode requires BF16 parameters")
            if value.device.type != "cuda":
                raise ValueError(f"{name} is on {value.device}; BF16 delta mode requires CUDA tensors")
            if not value.is_contiguous():
                raise ValueError(f"{name} is non-contiguous with stride {value.stride()}")
            nbytes = value.numel() * value.element_size()
            self._tensors.append(DeltaTensorMetadata(name=name, shape=tuple(value.shape), nbytes=nbytes))
            if self.profile is not None:
                self.profile.largest_tensor_bytes = max(self.profile.largest_tensor_bytes, nbytes)
            deltas.append(value)

        if self._compression_started_at is None:
            self._compression_started_at = time.perf_counter()
        pending = self.codec.submit_encode(deltas, profile=self.profile)
        self._pending.append(
            (
                tuple(delta.numel() * delta.element_size() for delta in deltas),
                pending,
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
            self.profile.nvcomp_compress_gpu_ms += _sum_completed_events(
                events.total for events in self._compression_events
            )
            self.profile.nvcomp_encode_gpu_ms += _sum_completed_events(
                events.encode for events in self._compression_events
            )
            self.profile.nvcomp_clone_gpu_ms += _sum_completed_events(
                events.clone for events in self._compression_events
            )
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
        uncompressed_sizes, pending = self._pending.popleft()
        payloads, events = self.codec.finalize_encode(pending, profile=self.profile)
        if len(payloads) != len(uncompressed_sizes):
            raise RuntimeError(
                f"nvCOMP returned {len(payloads)} frames for {len(uncompressed_sizes)} tensors"
            )
        self._frames.extend(
            CompressedDeltaFrame(
                uncompressed_nbytes=uncompressed_nbytes,
                compressed_nbytes=payload.numel() * payload.element_size(),
            )
            for uncompressed_nbytes, payload in zip(uncompressed_sizes, payloads, strict=True)
        )
        self._payloads.extend(payloads)
        if events is not None:
            self._compression_events.append(events)


def decode_delta_tensors(
    codec: NvcompLZ4Codec,
    tensors: Sequence[DeltaTensorMetadata],
    frames: Sequence[CompressedDeltaFrame],
    payloads: Sequence[Tensor],
    *,
    profile: bool = False,
) -> tuple[list[tuple[str, Tensor]], tuple[torch.cuda.Event, torch.cuda.Event] | None]:
    if not (len(tensors) == len(frames) == len(payloads)):
        raise ValueError(
            f"delta decode length mismatch: tensors={len(tensors)}, frames={len(frames)}, payloads={len(payloads)}"
        )
    decoded, events = codec.decode(
        payloads,
        [frame.uncompressed_nbytes for frame in frames],
        profile=profile,
    )
    values = [
        (metadata.name, raw.view(torch.bfloat16).view(metadata.shape))
        for metadata, raw in zip(tensors, decoded, strict=True)
    ]
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
    if len(update.tensors) != len(update.frames):
        raise ValueError(f"BF16 delta has {len(update.tensors)} tensors but {len(update.frames)} frames")
    names: set[str] = set()
    for tensor, frame in zip(update.tensors, update.frames, strict=True):
        if not tensor.name or tensor.name in names:
            raise ValueError(f"invalid or duplicate BF16 delta tensor name: {tensor.name!r}")
        names.add(tensor.name)
        expected = prod(tensor.shape) * 2
        if tensor.nbytes != expected:
            raise ValueError(f"{tensor.name} metadata has {tensor.nbytes} bytes; shape requires {expected}")
        if frame.uncompressed_nbytes != tensor.nbytes:
            raise ValueError(f"{tensor.name} frame has {frame.uncompressed_nbytes} raw bytes; expected {tensor.nbytes}")
        if frame.compressed_nbytes <= 0:
            raise ValueError(f"{tensor.name} has invalid compressed size {frame.compressed_nbytes}")
    packed_bytes = packed_delta_nbytes(update.frames)
    if packed_bytes != update.compressed_nbytes:
        raise ValueError(
            f"aligned compressed frame metadata describes {packed_bytes} bytes "
            f"but payload has {update.compressed_nbytes}"
        )
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
    "validate_delta_update",
]
