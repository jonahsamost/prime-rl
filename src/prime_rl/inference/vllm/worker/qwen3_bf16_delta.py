from __future__ import annotations

import torch
from torch import nn

from prime_rl.inference.vllm.worker.bf16_delta import (
    BF16DeltaError,
    apply_bf16_deltas_,
    route_bf16_values_to_named_parameters,
    route_bf16_values_to_scratch,
)
from prime_rl.weight_sync.profiling import WeightSyncMetrics, cuda_event_pair, elapsed_cuda_ms


@torch.no_grad()
def apply_qwen3_bf16_source_deltas_(
    model: nn.Module,
    source_deltas: dict[str, torch.Tensor],
    *,
    layer_index: int,
    profile: WeightSyncMetrics | None = None,
) -> int:
    """Route one Qwen3 source-layout group and XOR it into live vLLM weights."""
    if getattr(model.config, "model_type", None) != "qwen3":
        raise BF16DeltaError(
            f"BF16 delta loading currently supports Qwen3 only, got model_type={model.config.model_type!r}"
        )
    if not source_deltas:
        return 0

    route_events = cuda_event_pair() if profile is not None else None
    if route_events is not None:
        route_events[0].record()
    load = lambda: model.load_weights(source_deltas.items())
    if layer_index >= 0:
        layer = model.model.layers[layer_index]
        routed = route_bf16_values_to_scratch(layer, load)
    else:
        model_parameters = dict(model.named_parameters(remove_duplicate=False))
        missing = source_deltas.keys() - model_parameters.keys()
        if missing:
            raise BF16DeltaError(f"non-layer source deltas have no direct vLLM destination: {sorted(missing)}")
        selected = [(name, model_parameters[name]) for name in source_deltas]
        routed = route_bf16_values_to_named_parameters(selected, load, context="Qwen3 non-layer parameters")
    if route_events is not None:
        route_events[1].record()
        profile.vllm_route_ms += elapsed_cuda_ms(route_events)

    xor_events = cuda_event_pair() if profile is not None else None
    if xor_events is not None:
        xor_events[0].record()
    apply_bf16_deltas_(routed)
    if xor_events is not None:
        xor_events[1].record()
        profile.apply_xor_ms += elapsed_cuda_ms(xor_events)
    return len(routed)


__all__ = ["apply_qwen3_bf16_source_deltas_"]
