from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.adam import adam
from torch.optim.optimizer import _use_grad_for_differentiable

from prime_rl.weight_sync.bf16_delta import (
    BF16DeltaEncoder,
    BF16DeltaUpdate,
    NvcompLZ4Codec,
    local_tensor,
)
from prime_rl.weight_sync.profiling import (
    PhaseProfiler,
    WeightSyncMetrics,
    cuda_event_pair,
)

_CHECKPOINT_WRAPPER_PREFIX = "_checkpoint_wrapped_module."
_DEFAULT_ADAM_BUCKET_BYTES = 256 * 1024 * 1024


def _canonical_parameter_name(name: str) -> str:
    return name.replace(_CHECKPOINT_WRAPPER_PREFIX, "")


class DeltaAdamW(AdamW):
    """AdamW that records each parameter's exact BF16 XOR immediately after updating it."""

    def __init__(
        self,
        params: Iterable[tuple[str, nn.Parameter]],
        *,
        profiling_sample_interval_ms: float | None = None,
        delta_adam_bucket_bytes: int = _DEFAULT_ADAM_BUCKET_BYTES,
        **kwargs: Any,
    ) -> None:
        if delta_adam_bucket_bytes <= 0:
            raise ValueError(f"delta_adam_bucket_bytes must be positive, got {delta_adam_bucket_bytes}")
        named_params = [
            (_canonical_parameter_name(name), parameter) for name, parameter in params if parameter.requires_grad
        ]
        names = [name for name, _ in named_params]
        if len(names) != len(set(names)):
            raise ValueError("parameter names are not unique after removing checkpoint-wrapper prefixes")
        super().__init__([parameter for _, parameter in named_params], **kwargs)
        self._parameter_names = {id(parameter): name for name, parameter in named_params}
        self._encoder: BF16DeltaEncoder | None = None
        self._pending_update: BF16DeltaUpdate | None = None
        self._delta_adam_bucket_bytes = delta_adam_bucket_bytes
        device = local_tensor(named_params[0][1]).device
        self._delta_codec = NvcompLZ4Codec(device)
        self._profiler = PhaseProfiler(
            enabled=profiling_sample_interval_ms is not None,
            device=device,
            sample_interval_ms=profiling_sample_interval_ms or 5.0,
        )

    def begin_delta(self, *, base_step: int, step: int) -> None:
        if self._encoder is not None:
            raise RuntimeError("a BF16 delta recording is already active")
        if self._pending_update is not None:
            raise RuntimeError("the previous BF16 delta update has not been consumed")
        profile = WeightSyncMetrics() if self._profiler.enabled else None
        self._encoder = BF16DeltaEncoder(
            base_step=base_step,
            step=step,
            codec=self._delta_codec,
            profile=profile,
        )

    def take_delta_update(self) -> BF16DeltaUpdate | None:
        update = self._pending_update
        self._pending_update = None
        return update

    def abort_delta(self) -> None:
        if self._encoder is not None:
            self._encoder.abort()
        self._encoder = None
        self._pending_update = None

    @property
    def profiling_enabled(self) -> bool:
        return self._profiler.enabled

    def add_optimizer_offload_timings(self, *, h2d_ms: float = 0.0, d2h_ms: float = 0.0) -> None:
        profile = self._pending_update.profile if self._pending_update is not None else None
        if profile is not None:
            profile.optimizer_state_h2d_ms += h2d_ms
            profile.optimizer_state_d2h_ms += d2h_ms

    @_use_grad_for_differentiable
    def step(self, closure: Callable[[], float] | None = None):
        if self._encoder is None:
            return super().step(closure)
        if hasattr(self, "_accelerator_graph_capture_health_check"):
            self._accelerator_graph_capture_health_check()
        else:
            self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        update: BF16DeltaUpdate | None = None
        snapshot_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        adam_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        xor_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        with self._profiler.measure("delta_encode") as phase:
            try:
                for group in self.param_groups:
                    params_with_grad: list[Tensor] = []
                    grads: list[Tensor] = []
                    exp_avgs: list[Tensor] = []
                    exp_avg_sqs: list[Tensor] = []
                    max_exp_avg_sqs: list[Tensor] = []
                    state_steps: list[Tensor] = []
                    beta1, beta2 = group["betas"]

                    has_complex = self._init_group(
                        group,
                        params_with_grad,
                        grads,
                        exp_avgs,
                        exp_avg_sqs,
                        max_exp_avg_sqs,
                        state_steps,
                    )
                    if has_complex and self._encoder is not None:
                        raise TypeError("BF16 delta mode does not support complex parameters")

                    for bucket_start, bucket_end, bucket_bytes in _parameter_buckets(
                        params_with_grad,
                        self._delta_adam_bucket_bytes,
                    ):
                        bucket_parameters = params_with_grad[bucket_start:bucket_end]
                        local_parameters: list[Tensor] = []
                        old_values: list[Tensor] = []

                        snapshot_events = cuda_event_pair() if self._profiler.enabled else None
                        if snapshot_events is not None:
                            snapshot_events[0].record()
                        for parameter in bucket_parameters:
                            local_parameter = local_tensor(parameter)
                            name = self._parameter_names[id(parameter)]
                            if local_parameter.dtype != torch.bfloat16:
                                raise TypeError(
                                    f"{name} has dtype {local_parameter.dtype}; "
                                    "BF16 delta mode requires BF16 parameters"
                                )
                            if not local_parameter.is_contiguous():
                                raise ValueError(f"{name} is non-contiguous with stride {local_parameter.stride()}")
                            local_parameters.append(local_parameter)
                            old_values.append(local_parameter.detach().clone())
                        if snapshot_events is not None:
                            snapshot_events[1].record()
                            snapshot_event_pairs.append(snapshot_events)

                        adam_events = cuda_event_pair() if self._profiler.enabled else None
                        if adam_events is not None:
                            adam_events[0].record()
                        adam(
                            bucket_parameters,
                            grads[bucket_start:bucket_end],
                            exp_avgs[bucket_start:bucket_end],
                            exp_avg_sqs[bucket_start:bucket_end],
                            max_exp_avg_sqs[bucket_start:bucket_end] if group["amsgrad"] else [],
                            state_steps[bucket_start:bucket_end],
                            amsgrad=group["amsgrad"],
                            has_complex=False,
                            beta1=beta1,
                            beta2=beta2,
                            lr=group["lr"],
                            weight_decay=group["weight_decay"],
                            eps=group["eps"],
                            maximize=group["maximize"],
                            foreach=group["foreach"],
                            capturable=group["capturable"],
                            differentiable=group["differentiable"],
                            fused=group["fused"],
                            grad_scale=getattr(self, "grad_scale", None),
                            found_inf=getattr(self, "found_inf", None),
                            decoupled_weight_decay=group["decoupled_weight_decay"],
                        )
                        if adam_events is not None:
                            adam_events[1].record()
                            adam_event_pairs.append(adam_events)

                        xor_events = cuda_event_pair() if self._profiler.enabled else None
                        if xor_events is not None:
                            xor_events[0].record()
                        for current_value, old_value in zip(local_parameters, old_values, strict=True):
                            old_value.view(torch.int16).bitwise_xor_(current_value.view(torch.int16))
                        if xor_events is not None:
                            xor_events[1].record()
                            xor_event_pairs.append(xor_events)
                        self._encoder.append_batch(
                            [
                                (self._parameter_names[id(parameter)], old_value)
                                for parameter, old_value in zip(bucket_parameters, old_values, strict=True)
                            ]
                        )
                        if update_profile := self._encoder.profile:
                            update_profile.adam_bucket_count += 1
                            update_profile.max_adam_bucket_bytes = max(
                                update_profile.max_adam_bucket_bytes,
                                bucket_bytes,
                            )
                        del local_parameters, old_values
                if self._encoder is not None:
                    update = self._encoder.finish()
                    self._encoder = None
            except Exception:
                self.abort_delta()
                raise

        if update is not None:
            if update.profile is not None:
                # finish() synchronized the nvCOMP stream, which waits on the
                # optimizer stream. All stage events are complete here, so query
                # them without synchronizing inside the parameter loop.
                resolve_start = time.perf_counter()
                update.profile.snapshot_gpu_ms += _sum_completed_cuda_events(snapshot_event_pairs)
                update.profile.adam_gpu_ms += _sum_completed_cuda_events(adam_event_pairs)
                update.profile.xor_gpu_ms += _sum_completed_cuda_events(xor_event_pairs)
                update.profile.event_timing_resolve_wall_ms += (time.perf_counter() - resolve_start) * 1000
                update.profile.phases[phase.name] = phase
            # A compressed payload that is no smaller than BF16 falls back to the
            # existing full-weight path for this version.
            self._pending_update = update if update.compressed_nbytes < update.uncompressed_nbytes else None
        return loss


def _sum_completed_cuda_events(
    event_pairs: Iterable[tuple[torch.cuda.Event, torch.cuda.Event]],
) -> float:
    """Sum elapsed GPU time after the caller has established event completion."""
    return sum(start.elapsed_time(end) for start, end in event_pairs)


def _parameter_buckets(
    parameters: list[Tensor],
    bucket_bytes: int,
) -> Iterator[tuple[int, int, int]]:
    """Yield contiguous parameter ranges capped by local parameter bytes."""
    start = 0
    used = 0
    for index, parameter in enumerate(parameters):
        value = local_tensor(parameter)
        parameter_bytes = value.numel() * value.element_size()
        if used and used + parameter_bytes > bucket_bytes:
            yield start, index, used
            start = index
            used = 0
        used += parameter_bytes
    if start < len(parameters):
        yield start, len(parameters), used


__all__ = ["DeltaAdamW"]
