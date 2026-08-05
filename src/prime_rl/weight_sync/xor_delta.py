from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from math import prod
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor

NCCL_DELTA_PROTOCOL_MAGIC = 0x50524C44  # "PRLD"
NVCOMP_FRAME_ALIGNMENT = 256
DEFAULT_NVCOMP_PIPELINE_DEPTH = 2
SUPPORTED_DELTA_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

_DTYPE_NAMES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}
_DTYPES_BY_NAME = {name: dtype for dtype, name in _DTYPE_NAMES.items()}
_INTEGER_DTYPES = {
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
    torch.float32: torch.int32,
}
_DTYPE_NBYTES = {
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.float32: 4,
}


class WeightUpdateKind(IntEnum):
    FULL = 0
    XOR = 1


@dataclass(frozen=True)
class WeightUpdateHeader:
    kind: WeightUpdateKind
    base_step: int
    step: int
    optimizer_start_ns: int = 0


def encode_weight_update_header(header: WeightUpdateHeader, *, device: torch.device | str | int) -> Tensor:
    if header.kind == WeightUpdateKind.XOR and header.step != header.base_step + 1:
        raise ValueError(f"XOR header must name consecutive versions: base_step={header.base_step}, step={header.step}")
    if header.kind == WeightUpdateKind.FULL and header.base_step != -1:
        raise ValueError(f"full update header must use base_step=-1, got {header.base_step}")
    if header.optimizer_start_ns < 0:
        raise ValueError(f"optimizer_start_ns must be non-negative, got {header.optimizer_start_ns}")
    return torch.tensor(
        [
            NCCL_DELTA_PROTOCOL_MAGIC,
            int(header.kind),
            header.base_step,
            header.step,
            header.optimizer_start_ns,
        ],
        dtype=torch.long,
        device=device,
    )


def decode_weight_update_header(values: Tensor) -> WeightUpdateHeader:
    if values.dtype != torch.long or values.shape != (5,):
        raise ValueError(f"invalid NCCL update header tensor: dtype={values.dtype}, shape={tuple(values.shape)}")
    magic, kind, base_step, step, optimizer_start_ns = (int(value) for value in values.tolist())
    if magic != NCCL_DELTA_PROTOCOL_MAGIC:
        raise ValueError(f"invalid NCCL delta protocol magic: {magic:#x}")
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
    dtype: str
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
class PendingNvcompEncode:
    output_buffers: tuple[Tensor, ...]
    encoded: tuple[Any, ...]


@dataclass(frozen=True)
class PendingDeltaFrame:
    first_tensor_index: int
    tensor_count: int
    uncompressed_nbytes: int
    encode: PendingNvcompEncode


@dataclass(frozen=True)
class DeltaUpdate:
    """A GPU-resident nvCOMP LZ4 source-layout XOR update."""

    base_step: int
    step: int
    tensors: tuple[DeltaTensorMetadata, ...]
    frames: tuple[CompressedDeltaFrame, ...]
    payload: Tensor

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
        packed_nbytes = _align_up(offset, NVCOMP_FRAME_ALIGNMENT)
        if packed_nbytes != self.compressed_nbytes:
            raise ValueError(
                f"compressed frame metadata describes {packed_nbytes} aligned bytes "
                f"but payload has {self.compressed_nbytes}"
            )


@dataclass(frozen=True)
class ShardedDeltaUpdate:
    """Rank-local compressed FSDP shards for one logical policy update."""

    base_step: int
    step: int
    shards: tuple[DeltaUpdate, ...]

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


def delta_dtype_name(dtype: torch.dtype) -> str:
    try:
        return _DTYPE_NAMES[dtype]
    except KeyError as error:
        raise TypeError(f"unsupported XOR delta dtype {dtype}; expected one of {SUPPORTED_DELTA_DTYPES}") from error


def delta_dtype_from_name(name: str) -> torch.dtype:
    try:
        return _DTYPES_BY_NAME[name]
    except KeyError as error:
        raise ValueError(f"unsupported XOR delta dtype name {name!r}") from error


def integer_view(tensor: Tensor) -> Tensor:
    try:
        dtype = _INTEGER_DTYPES[tensor.dtype]
    except KeyError as error:
        raise TypeError(
            f"unsupported XOR delta dtype {tensor.dtype}; expected one of {SUPPORTED_DELTA_DTYPES}"
        ) from error
    return tensor.view(dtype)


def delta_dtype_nbytes(dtype: torch.dtype) -> int:
    try:
        return _DTYPE_NBYTES[dtype]
    except KeyError as error:
        raise TypeError(f"unsupported XOR delta dtype {dtype}; expected one of {SUPPORTED_DELTA_DTYPES}") from error


def _load_nvcomp():
    try:
        from nvidia import nvcomp
    except ImportError as error:
        raise RuntimeError(
            "XOR weight synchronization requires nvidia-nvcomp-cu12 on a CUDA-capable Linux host"
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
        self._packed_capacity_hint = 0

    def encode(
        self,
        values: Sequence[Tensor],
    ) -> list[Tensor]:
        if not values:
            return []
        return self.finalize_encode(self.submit_encode(values))

    def submit_encode(
        self,
        values: Sequence[Tensor],
    ) -> PendingNvcompEncode:
        if not values:
            raise ValueError("cannot submit an empty nvCOMP encode batch")
        byte_views = [self._byte_view(value) for value in values]
        sizes = tuple(value.numel() for value in byte_views)
        producer_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(producer_stream)
        for value in values:
            value.record_stream(self.stream)

        with torch.cuda.stream(self.stream):
            sources = self.nvcomp.as_arrays(byte_views, cuda_stream=self.stream.cuda_stream)
            output_buffers = [
                torch.empty(
                    self.codec.get_max_comp_buffer_size(source),
                    dtype=torch.uint8,
                    device=self.device,
                )
                for source in sources
            ]
            destinations = self.nvcomp.as_arrays(output_buffers, cuda_stream=self.stream.cuda_stream)
            compression_config = self._compression_config(sizes)
            encoded = self.codec.encode(
                sources,
                out=destinations,
                compression_config=compression_config,
            )

        return PendingNvcompEncode(
            output_buffers=tuple(output_buffers),
            encoded=tuple(encoded),
        )

    def finalize_encode(
        self,
        pending: PendingNvcompEncode,
    ) -> list[Tensor]:
        with torch.cuda.stream(self.stream):
            compressed_sizes = [item.buffer_size for item in pending.encoded]
            # Keep Torch as the allocator/owner. Converting nvCOMP-owned output
            # arrays through DLPack and then dropping their wrappers can leave
            # Torch tensors referring to released storage. Clone only the valid
            # compressed range so the worst-case output buffers remain bounded
            # to the current Adam bucket instead of accumulating for the model.
            payloads = [
                buffer.narrow(0, 0, compressed_size).clone()
                for buffer, compressed_size in zip(pending.output_buffers, compressed_sizes, strict=True)
            ]
        return payloads

    def copy_encoded(
        self,
        pending: PendingNvcompEncode,
        destinations: Sequence[Tensor],
        compressed_sizes: Sequence[int],
    ) -> None:
        if len(pending.encoded) != len(destinations) or len(destinations) != len(compressed_sizes):
            raise ValueError(
                f"received {len(destinations)} destinations and {len(compressed_sizes)} sizes "
                f"for {len(pending.encoded)} encoded values"
            )
        with torch.cuda.stream(self.stream):
            for buffer, compressed_size, destination in zip(
                pending.output_buffers,
                compressed_sizes,
                destinations,
                strict=True,
            ):
                if destination.dtype != torch.uint8 or destination.numel() != compressed_size:
                    raise ValueError(
                        f"encoded destination has dtype={destination.dtype}, nbytes={destination.numel()}; "
                        f"expected uint8 with {compressed_size} bytes"
                    )
                destination.copy_(buffer.narrow(0, 0, compressed_size))

    def decode(
        self,
        payloads: Sequence[Tensor],
        uncompressed_nbytes: Sequence[int],
    ) -> list[Tensor]:
        if not payloads:
            return []
        sizes = tuple(uncompressed_nbytes)
        if len(payloads) != len(sizes):
            raise ValueError(f"received {len(payloads)} compressed frames for {len(sizes)} output sizes")
        consumer_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(consumer_stream)
        for payload in payloads:
            payload.record_stream(self.stream)

        with torch.cuda.stream(self.stream):
            outputs = [torch.empty(size, dtype=torch.uint8, device=self.device) for size in sizes]
            sources = self.nvcomp.as_arrays(payloads, cuda_stream=self.stream.cuda_stream)
            destinations = self.nvcomp.as_arrays(outputs, cuda_stream=self.stream.cuda_stream)
            decompression_config = self._decompression_config(sizes, sources)
            self.codec.decode(
                sources,
                out=destinations,
                decompression_config=decompression_config,
            )
        consumer_stream.wait_stream(self.stream)
        return outputs

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


class DeltaEncoder:
    """Compress contiguous floating-point XOR parameter buckets through nvCOMP LZ4 on GPU."""

    def __init__(
        self,
        *,
        base_step: int,
        step: int,
        codec: NvcompLZ4Codec,
        pipeline_depth: int = DEFAULT_NVCOMP_PIPELINE_DEPTH,
    ) -> None:
        if step != base_step + 1:
            raise ValueError(f"XOR delta updates must be consecutive: base_step={base_step}, step={step}")
        self.base_step = base_step
        self.step = step
        self.codec = codec
        if pipeline_depth < 1:
            raise ValueError(f"delta pipeline depth must be positive, got {pipeline_depth}")
        self.pipeline_depth = pipeline_depth
        self._tensors: list[DeltaTensorMetadata] = []
        self._frames: list[CompressedDeltaFrame] = []
        self._payload: Tensor | None = None
        self._payload_size = 0
        self._pending: deque[PendingDeltaFrame] = deque()
        self._finished = False

    def append(self, name: str, delta: Tensor) -> None:
        self.append_batch([(name, delta)])

    def append_batch(self, values: Sequence[tuple[str, Tensor]]) -> None:
        if self._finished:
            raise RuntimeError("cannot append to a finished XOR delta encoder")
        normalized = self._normalize_values(values)
        if not normalized:
            return
        device = normalized[0][1].device
        dtype = normalized[0][1].dtype
        if any(value.dtype != dtype for _, value in normalized):
            raise TypeError("an XOR delta batch must contain a single dtype")
        bucket = torch.empty(sum(value.numel() for _, value in normalized), dtype=dtype, device=device)
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
            raise RuntimeError("cannot append to a finished XOR delta encoder")
        normalized = self._normalize_values(values)
        if not normalized:
            return
        bucket = local_tensor(bucket.detach())
        if bucket.dtype not in SUPPORTED_DELTA_DTYPES or bucket.device.type != "cuda" or not bucket.is_contiguous():
            raise ValueError(
                "XOR delta bucket must be a contiguous CUDA tensor with a supported dtype; "
                f"got dtype={bucket.dtype}, device={bucket.device}, stride={bucket.stride()}"
            )
        expected_elements = sum(value.numel() for _, value in normalized)
        if bucket.numel() != expected_elements:
            raise ValueError(f"XOR delta bucket has {bucket.numel()} elements; tensors require {expected_elements}")
        offset = 0
        for name, value in normalized:
            if value.device != bucket.device:
                raise ValueError(f"{name} is on {value.device}; bucket is on {bucket.device}")
            if value.dtype != bucket.dtype:
                raise TypeError(f"{name} has dtype {value.dtype}; bucket has dtype {bucket.dtype}")
            expected_pointer = bucket.data_ptr() + offset * bucket.element_size()
            if value.data_ptr() != expected_pointer:
                raise ValueError(f"{name} is not the expected contiguous view into its XOR delta bucket")
            offset += value.numel()

        if global_shapes is None:
            global_shapes = [tuple(value.shape) for _, value in normalized]
        if len(global_shapes) != len(normalized):
            raise ValueError(f"received {len(global_shapes)} global shapes for {len(normalized)} XOR delta tensors")
        if shard_descriptors is None:
            shard_descriptors = [(shard_dim, shard_index, shard_count)] * len(normalized)
        if len(shard_descriptors) != len(normalized):
            raise ValueError(
                f"received {len(shard_descriptors)} shard descriptors for {len(normalized)} XOR delta tensors"
            )
        metadata = tuple(
            DeltaTensorMetadata(
                name=name,
                shape=tuple(value.shape),
                dtype=delta_dtype_name(value.dtype),
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
        self._pending.append(
            PendingDeltaFrame(
                first_tensor_index=first_tensor_index,
                tensor_count=len(metadata),
                uncompressed_nbytes=bucket.numel() * bucket.element_size(),
                encode=self.codec.submit_encode([bucket]),
            )
        )
        if len(self._pending) >= self.pipeline_depth:
            self._finalize_oldest()

    def finish(self) -> DeltaUpdate:
        if self._finished:
            raise RuntimeError("XOR delta encoder is already finished")
        self._finished = True
        while self._pending:
            self._finalize_oldest()
        packed_nbytes = _align_up(self._payload_size, NVCOMP_FRAME_ALIGNMENT)
        self._reserve_payload(packed_nbytes)
        assert self._payload is not None
        packed = self._payload.narrow(0, 0, packed_nbytes)
        self.codec._packed_capacity_hint = _align_up(
            packed_nbytes + packed_nbytes // 8,
            NVCOMP_FRAME_ALIGNMENT,
        )
        self.codec.synchronize()
        update = DeltaUpdate(
            base_step=self.base_step,
            step=self.step,
            tensors=tuple(self._tensors),
            frames=tuple(self._frames),
            payload=packed,
        )
        validate_delta_update(update)
        return update

    def abort(self) -> None:
        if not self._finished:
            self._finished = True
            self.codec.synchronize()

    def _finalize_oldest(self) -> None:
        pending = self._pending.popleft()
        compressed_sizes = tuple(item.buffer_size for item in pending.encode.encoded)
        if len(compressed_sizes) != 1:
            raise RuntimeError(f"nvCOMP returned {len(compressed_sizes)} frames for one XOR delta bucket")
        compressed_nbytes = compressed_sizes[0]
        offset = _align_up(self._payload_size, NVCOMP_FRAME_ALIGNMENT)
        self._reserve_payload(offset + compressed_nbytes)
        assert self._payload is not None
        destination = self._payload.narrow(0, offset, compressed_nbytes)
        self.codec.copy_encoded(pending.encode, [destination], compressed_sizes)
        self._payload_size = offset + compressed_nbytes
        self._frames.append(
            CompressedDeltaFrame(
                first_tensor_index=pending.first_tensor_index,
                tensor_count=pending.tensor_count,
                uncompressed_nbytes=pending.uncompressed_nbytes,
                compressed_nbytes=compressed_nbytes,
            )
        )

    def _reserve_payload(self, required: int) -> None:
        if self._payload is not None and self._payload.numel() >= required:
            return
        current_capacity = 0 if self._payload is None else self._payload.numel()
        capacity = _align_up(
            max(required, self.codec._packed_capacity_hint, max(NVCOMP_FRAME_ALIGNMENT, current_capacity * 2)),
            NVCOMP_FRAME_ALIGNMENT,
        )
        replacement = torch.empty(capacity, dtype=torch.uint8, device=self.codec.device)
        if self._payload is not None and self._payload_size:
            self._payload.record_stream(self.codec.stream)
            with torch.cuda.stream(self.codec.stream):
                replacement.narrow(0, 0, self._payload_size).copy_(
                    self._payload.narrow(0, 0, self._payload_size)
                )
        self._payload = replacement

    @staticmethod
    def _normalize_values(values: Sequence[tuple[str, Tensor]]) -> list[tuple[str, Tensor]]:
        normalized: list[tuple[str, Tensor]] = []
        for name, delta in values:
            value = local_tensor(delta.detach())
            if value.dtype not in SUPPORTED_DELTA_DTYPES:
                raise TypeError(f"{name} has dtype {value.dtype}; XOR delta mode supports {SUPPORTED_DELTA_DTYPES}")
            if value.device.type != "cuda":
                raise ValueError(f"{name} is on {value.device}; XOR delta mode requires CUDA tensors")
            if not value.is_contiguous():
                raise ValueError(f"{name} is non-contiguous with stride {value.stride()}")
            if value.numel() == 0:
                raise ValueError(f"{name} is empty; XOR delta tensors must contain at least one element")
            normalized.append((name, value))
        return normalized


def decode_delta_tensors(
    codec: NvcompLZ4Codec,
    tensors: Sequence[DeltaTensorMetadata],
    frames: Sequence[CompressedDeltaFrame],
    payloads: Sequence[Tensor],
) -> list[tuple[str, Tensor]]:
    if len(frames) != len(payloads):
        raise ValueError(f"delta decode length mismatch: frames={len(frames)}, payloads={len(payloads)}")
    decoded = codec.decode(
        payloads,
        [frame.uncompressed_nbytes for frame in frames],
    )
    values: list[tuple[str, Tensor]] = []
    for frame, raw in zip(frames, decoded, strict=True):
        values.extend(unpack_delta_frame(raw, tensors, frame))
    return values


def unpack_delta_frame(
    raw: Tensor,
    tensors: Sequence[DeltaTensorMetadata],
    frame: CompressedDeltaFrame,
) -> list[tuple[str, Tensor]]:
    """Split one decoded frame into its typed tensor views."""
    if raw.dtype != torch.uint8 or raw.numel() != frame.uncompressed_nbytes:
        raise ValueError(
            f"decoded XOR delta frame has dtype={raw.dtype}, nbytes={raw.numel()}; "
            f"expected uint8 with {frame.uncompressed_nbytes} bytes"
        )
    values: list[tuple[str, Tensor]] = []
    offset = 0
    frame_tensors = tensors[frame.first_tensor_index : frame.first_tensor_index + frame.tensor_count]
    for metadata in frame_tensors:
        tensor_bytes = raw.narrow(0, offset, metadata.nbytes)
        dtype = delta_dtype_from_name(metadata.dtype)
        values.append((metadata.name, tensor_bytes.view(dtype).view(metadata.shape)))
        offset += metadata.nbytes
    if offset != frame.uncompressed_nbytes:
        raise ValueError(
            f"decoded XOR delta frame describes {offset} tensor bytes; expected {frame.uncompressed_nbytes}"
        )
    return values


def validate_delta_update(update: DeltaUpdate) -> None:
    valid_payload = (
        update.payload.dtype == torch.uint8 and update.payload.device.type == "cuda" and update.payload.is_contiguous()
    )
    if not valid_payload:
        raise ValueError(
            "XOR delta payload must be a contiguous CUDA uint8 tensor; "
            f"got dtype={update.payload.dtype}, device={update.payload.device}, stride={update.payload.stride()}"
        )
    if update.payload.data_ptr() % NVCOMP_FRAME_ALIGNMENT != 0:
        raise ValueError(
            f"XOR delta payload address {update.payload.data_ptr():#x} is not {NVCOMP_FRAME_ALIGNMENT}-byte aligned"
        )
    names: set[str] = set()
    for tensor in update.tensors:
        if not tensor.name or tensor.name in names:
            raise ValueError(f"invalid or duplicate XOR delta tensor name: {tensor.name!r}")
        names.add(tensor.name)
        dtype = delta_dtype_from_name(tensor.dtype)
        expected = prod(tensor.shape) * delta_dtype_nbytes(dtype)
        if tensor.nbytes != expected:
            raise ValueError(f"{tensor.name} metadata has {tensor.nbytes} bytes; shape requires {expected}")
        global_shape = tensor.resolved_global_shape
        if len(global_shape) != len(tensor.shape):
            raise ValueError(
                f"{tensor.name} local rank {len(tensor.shape)} does not match global rank {len(global_shape)}"
            )
        if tensor.shard_count <= 0 or not 0 <= tensor.shard_index < tensor.shard_count:
            raise ValueError(f"{tensor.name} has invalid shard {tensor.shard_index}/{tensor.shard_count}")
        if tensor.shard_count == 1:
            if tensor.shard_dim is not None or tensor.shape != global_shape:
                raise ValueError(f"unsharded tensor {tensor.name} must have identical local and global shapes")
        elif tensor.shard_dim != 0:
            raise ValueError(f"{tensor.name} uses unsupported shard dimension {tensor.shard_dim}; expected 0")
    next_tensor_index = 0
    for frame_index, frame in enumerate(update.frames):
        if frame.first_tensor_index != next_tensor_index:
            raise ValueError(
                f"XOR delta frame {frame_index} starts at tensor {frame.first_tensor_index}; "
                f"expected {next_tensor_index}"
            )
        if frame.tensor_count <= 0 or frame.first_tensor_index + frame.tensor_count > len(update.tensors):
            raise ValueError(f"XOR delta frame {frame_index} has invalid tensor count {frame.tensor_count}")
        frame_tensors = update.tensors[frame.first_tensor_index : frame.first_tensor_index + frame.tensor_count]
        frame_dtypes = {tensor.dtype for tensor in frame_tensors}
        if len(frame_dtypes) != 1:
            raise ValueError(f"XOR delta frame {frame_index} mixes tensor dtypes: {sorted(frame_dtypes)}")
        expected_uncompressed = sum(tensor.nbytes for tensor in frame_tensors)
        if frame.uncompressed_nbytes != expected_uncompressed:
            raise ValueError(
                f"XOR delta frame {frame_index} has {frame.uncompressed_nbytes} raw bytes; "
                f"its tensors require {expected_uncompressed}"
            )
        if frame.compressed_nbytes <= 0:
            raise ValueError(f"XOR delta frame {frame_index} has invalid compressed size {frame.compressed_nbytes}")
        next_tensor_index += frame.tensor_count
    if next_tensor_index != len(update.tensors):
        raise ValueError(f"XOR delta frames cover {next_tensor_index} of {len(update.tensors)} tensors")
    packed_bytes = packed_delta_nbytes(update.frames)
    if packed_bytes != update.compressed_nbytes:
        raise ValueError(
            f"aligned compressed frame metadata describes {packed_bytes} bytes "
            f"but payload has {update.compressed_nbytes}"
        )


def validate_sharded_delta_update(update: ShardedDeltaUpdate) -> None:
    if not update.shards:
        raise ValueError("distributed XOR delta update has no trainer shards")
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
            if (
                candidate.name != expected.name
                or candidate.dtype != expected.dtype
                or candidate.resolved_global_shape != expected.resolved_global_shape
            ):
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
            raise ValueError(f"{pieces[0].name} shards reconstruct shape {reconstructed}; expected {global_shape}")


def reconstruct_delta_tensors(
    shard_values: Sequence[Sequence[tuple[str, Tensor]]],
    shard_metadata: Sequence[Sequence[DeltaTensorMetadata]],
) -> list[tuple[str, Tensor]]:
    """Reconstruct full source-layout tensors from rank-ordered dimension-0 shards."""
    if not shard_values or len(shard_values) != len(shard_metadata):
        raise ValueError("XOR delta reconstruction requires matching non-empty values and metadata")
    tensor_count = len(shard_values[0])
    if any(len(values) != tensor_count for values in shard_values) or any(
        len(metadata) != tensor_count for metadata in shard_metadata
    ):
        raise ValueError("XOR delta shard frames contain different tensor counts")
    reconstructed: list[tuple[str, Tensor]] = []
    for tensor_index in range(tensor_count):
        pieces = [values[tensor_index][1] for values in shard_values]
        metadata = [items[tensor_index] for items in shard_metadata]
        names = [values[tensor_index][0] for values in shard_values]
        if any(name != names[0] for name in names) or any(item.name != names[0] for item in metadata):
            raise ValueError(f"XOR delta shard tensor {tensor_index} has inconsistent names")
        global_shape = metadata[0].resolved_global_shape
        if any(item.resolved_global_shape != global_shape for item in metadata):
            raise ValueError(f"{names[0]} has inconsistent global shapes")
        dtype = metadata[0].dtype
        if any(item.dtype != dtype for item in metadata) or any(
            piece.dtype != delta_dtype_from_name(dtype) for piece in pieces
        ):
            raise ValueError(f"{names[0]} has inconsistent XOR delta dtypes")
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


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def packed_delta_nbytes(frames: Sequence[CompressedDeltaFrame]) -> int:
    """Return the CUDA payload size including frame and terminal alignment."""
    size = 0
    for frame in frames:
        size = _align_up(size, NVCOMP_FRAME_ALIGNMENT) + frame.compressed_nbytes
    return _align_up(size, NVCOMP_FRAME_ALIGNMENT)


__all__ = [
    "CompressedDeltaFrame",
    "DeltaEncoder",
    "DeltaTensorMetadata",
    "DeltaUpdate",
    "NvcompLZ4Codec",
    "SUPPORTED_DELTA_DTYPES",
    "ShardedDeltaUpdate",
    "WeightUpdateHeader",
    "WeightUpdateKind",
    "delta_dtype_from_name",
    "delta_dtype_name",
    "delta_dtype_nbytes",
    "decode_delta_tensors",
    "decode_weight_update_header",
    "encode_weight_update_header",
    "integer_view",
    "local_tensor",
    "packed_delta_nbytes",
    "reconstruct_delta_tensors",
    "unpack_delta_frame",
    "validate_delta_update",
    "validate_sharded_delta_update",
]
