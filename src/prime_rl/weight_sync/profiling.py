from __future__ import annotations

import json
import os
import resource
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class MemorySample:
    gpu_allocated_bytes: int
    gpu_reserved_bytes: int
    gpu_device_used_bytes: int
    pinned_allocated_bytes: int
    pinned_active_bytes: int
    rss_bytes: int


@dataclass(frozen=True)
class MemoryProfile:
    start: MemorySample
    peak: MemorySample
    end: MemorySample

    def flat_metrics(self, prefix: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for field_name in MemorySample.__dataclass_fields__:
            start = getattr(self.start, field_name)
            peak = getattr(self.peak, field_name)
            end = getattr(self.end, field_name)
            name = field_name.removesuffix("_bytes")
            metrics[f"{prefix}/memory/{name}_start_gib"] = start / 1024**3
            metrics[f"{prefix}/memory/{name}_peak_gib"] = peak / 1024**3
            metrics[f"{prefix}/memory/{name}_end_gib"] = end / 1024**3
            metrics[f"{prefix}/memory/{name}_peak_growth_gib"] = max(0, peak - start) / 1024**3
        return metrics


@dataclass
class PhaseProfile:
    name: str
    wall_ms: float = 0.0
    memory: MemoryProfile | None = None

    def flat_metrics(self, prefix: str = "weight_sync") -> dict[str, float]:
        metrics = {f"{prefix}/{self.name}/wall_ms": self.wall_ms}
        if self.memory is not None:
            metrics.update(self.memory.flat_metrics(f"{prefix}/{self.name}"))
        return metrics


@dataclass
class WeightSyncMetrics:
    optimizer_step_wall_ms: float = 0.0
    optimizer_step_gpu_ms: float = 0.0
    optimizer_state_h2d_ms: float = 0.0
    optimizer_state_d2h_ms: float = 0.0
    snapshot_gpu_ms: float = 0.0
    adam_gpu_ms: float = 0.0
    adam_bucket_count: int = 0
    max_adam_bucket_bytes: int = 0
    xor_gpu_ms: float = 0.0
    event_timing_resolve_wall_ms: float = 0.0
    nvcomp_compress_gpu_ms: float = 0.0
    nvcomp_encode_gpu_ms: float = 0.0
    nvcomp_clone_gpu_ms: float = 0.0
    nvcomp_compress_wall_ms: float = 0.0
    nvcomp_output_alloc_wall_ms: float = 0.0
    nvcomp_encode_call_wall_ms: float = 0.0
    nvcomp_buffer_size_read_wall_ms: float = 0.0
    nvcomp_clone_enqueue_wall_ms: float = 0.0
    nvcomp_decompress_gpu_ms: float = 0.0
    delta_gpu_pack_ms: float = 0.0
    nvcomp_batch_count: int = 0
    nvcomp_peak_pending_batches: int = 0
    trainer_shard_count: int = 0
    trainer_gather_wall_ms: float = 0.0
    trainer_gather_gpu_ms: float = 0.0
    trainer_gather_bytes: int = 0
    min_rank_compressed_bytes: int = 0
    max_rank_compressed_bytes: int = 0
    rank_compressed_bytes: list[int] = field(default_factory=list)
    sender_stage_h2d_ms: float = 0.0
    sender_nccl_ms: float = 0.0
    full_pack_ms: float = 0.0
    receiver_nccl_ms: float = 0.0
    receiver_stage_d2h_ms: float = 0.0
    vllm_route_ms: float = 0.0
    apply_xor_ms: float = 0.0
    full_apply_ms: float = 0.0
    raw_bytes: int = 0
    compressed_bytes: int = 0
    wire_bytes: int = 0
    nccl_call_count: int = 0
    tensor_count: int = 0
    frame_count: int = 0
    largest_tensor_bytes: int = 0
    phases: dict[str, PhaseProfile] = field(default_factory=dict)

    def flat_metrics(self, prefix: str = "weight_sync") -> dict[str, float]:
        metrics: dict[str, float] = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if field_name == "phases":
                continue
            if isinstance(value, (int, float)):
                metrics[f"{prefix}/{field_name}"] = float(value)
        for phase in self.phases.values():
            metrics.update(phase.flat_metrics(prefix))
        if self.compressed_bytes:
            metrics[f"{prefix}/compression_ratio"] = self.raw_bytes / self.compressed_bytes
        if self.wire_bytes:
            metrics[f"{prefix}/wire_reduction_ratio"] = self.raw_bytes / self.wire_bytes
        return metrics

    def structured_log(self, *, event: str, step: int, role: str) -> str:
        payload: dict[str, Any] = {
            "event": event,
            "role": role,
            "step": step,
            **asdict(self),
        }
        if self.compressed_bytes:
            payload["compression_ratio"] = self.raw_bytes / self.compressed_bytes
        if self.wire_bytes:
            payload["wire_reduction_ratio"] = self.raw_bytes / self.wire_bytes
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class PhaseProfiler:
    def __init__(
        self,
        *,
        enabled: bool,
        device: torch.device | str | int,
        sample_interval_ms: float = 5.0,
    ) -> None:
        if sample_interval_ms <= 0:
            raise ValueError(f"sample_interval_ms must be positive, got {sample_interval_ms}")
        self.enabled = enabled
        self.device = torch.device(device)
        self.sample_interval_seconds = sample_interval_ms / 1000

    def measure(self, name: str) -> _PhaseMeasurement:
        return _PhaseMeasurement(self, name)

    def sample(self) -> MemorySample:
        host_stats = torch.cuda.host_memory_stats()
        return MemorySample(
            gpu_allocated_bytes=torch.cuda.memory_allocated(self.device),
            gpu_reserved_bytes=torch.cuda.memory_reserved(self.device),
            gpu_device_used_bytes=torch.cuda.device_memory_used(self.device),
            pinned_allocated_bytes=int(host_stats.get("allocated_bytes.current", 0)),
            pinned_active_bytes=int(host_stats.get("active_bytes.current", 0)),
            rss_bytes=_process_rss_bytes(),
        )


class _PhaseMeasurement:
    def __init__(self, profiler: PhaseProfiler, name: str) -> None:
        self.profiler = profiler
        self.profile = PhaseProfile(name=name)
        self._sampler: _MemorySampler | None = None
        self._start_time = 0.0

    def __enter__(self) -> PhaseProfile:
        if not self.profiler.enabled:
            return self.profile
        torch.cuda.synchronize(self.profiler.device)
        start = self.profiler.sample()
        self._sampler = _MemorySampler(self.profiler, start)
        self._sampler.start()
        self._start_time = time.perf_counter()
        return self.profile

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self.profiler.enabled:
            return
        torch.cuda.synchronize(self.profiler.device)
        self.profile.wall_ms = (time.perf_counter() - self._start_time) * 1000
        assert self._sampler is not None
        self.profile.memory = self._sampler.finish()


class _MemorySampler:
    def __init__(self, profiler: PhaseProfiler, start: MemorySample) -> None:
        self.profiler = profiler
        self.start_sample = start
        self.peak_sample = start
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="weight-sync-memory-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> MemoryProfile:
        self._stop.set()
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("weight-sync memory sampler failed") from self._error
        end = self.profiler.sample()
        self._update_peak(end)
        return MemoryProfile(start=self.start_sample, peak=self.peak_sample, end=end)

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.profiler.sample_interval_seconds):
                self._update_peak(self.profiler.sample())
        except BaseException as error:
            self._error = error

    def _update_peak(self, sample: MemorySample) -> None:
        self.peak_sample = MemorySample(
            **{
                field_name: max(getattr(self.peak_sample, field_name), getattr(sample, field_name))
                for field_name in MemorySample.__dataclass_fields__
            }
        )


def cuda_event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def elapsed_cuda_ms(events: tuple[torch.cuda.Event, torch.cuda.Event]) -> float:
    start, end = events
    end.synchronize()
    return start.elapsed_time(end)


def _process_rss_bytes() -> int:
    statm = Path("/proc/self/statm")
    if statm.exists():
        resident_pages = int(statm.read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if os.uname().sysname == "Darwin" else usage * 1024)


__all__ = [
    "WeightSyncMetrics",
    "MemoryProfile",
    "MemorySample",
    "PhaseProfile",
    "PhaseProfiler",
    "cuda_event_pair",
    "elapsed_cuda_ms",
]
