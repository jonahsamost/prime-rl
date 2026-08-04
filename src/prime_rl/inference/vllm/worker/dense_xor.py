"""XOR-delta routing for dense vLLM models with Hugging Face-style layer names."""

from __future__ import annotations

import torch
from torch import nn

from prime_rl.inference.vllm.worker.xor_delta import (
    DeltaError,
    apply_deltas_,
    route_values_to_named_parameters,
    route_values_to_scratch,
)
from prime_rl.weight_sync.xor_delta import SUPPORTED_DELTA_DTYPES

_LAYER_CONTAINER_NAMES = frozenset({"block", "blocks", "h", "layers"})
_MOE_CONFIG_FIELDS = (
    "num_experts",
    "num_experts_per_tok",
    "num_local_experts",
    "n_routed_experts",
)


def validate_dense_delta_model(model: nn.Module) -> torch.dtype:
    config = getattr(model, "config", None)
    if config is None:
        raise DeltaError("XOR delta loading requires a model config")
    model_type = str(getattr(config, "model_type", ""))
    if "moe" in model_type.lower() or any(_enabled_expert_field(config, field) for field in _MOE_CONFIG_FIELDS):
        raise DeltaError(f"XOR delta loading does not support MoE model_type={model_type!r}")
    if getattr(config, "vision_config", None) is not None:
        raise DeltaError(f"XOR delta loading does not support multimodal model_type={model_type!r}")
    if not callable(getattr(model, "load_weights", None)):
        raise DeltaError(f"XOR delta loading requires {type(model).__name__}.load_weights()")
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if len(dtypes) != 1:
        raise DeltaError(f"XOR delta loading requires one model storage dtype, got {sorted(map(str, dtypes))}")
    dtype = dtypes.pop()
    if dtype not in SUPPORTED_DELTA_DTYPES:
        raise DeltaError(f"XOR delta loading does not support model storage dtype {dtype}")
    return dtype


@torch.no_grad()
def apply_dense_source_deltas_(model: nn.Module, source_deltas: dict[str, torch.Tensor]) -> int:
    """Route one source-layout parameter group and XOR it into live vLLM weights."""
    model_dtype = validate_dense_delta_model(model)
    if not source_deltas:
        return 0
    source_dtypes = {value.dtype for value in source_deltas.values()}
    if source_dtypes != {model_dtype}:
        raise DeltaError(
            f"source delta dtypes {sorted(map(str, source_dtypes))} do not match model storage dtype {model_dtype}"
        )

    layer_paths = {_layer_module_path(name) for name in source_deltas}
    if len(layer_paths) != 1:
        raise DeltaError(f"XOR delta group spans multiple layer modules: {sorted(layer_paths, key=str)}")

    load = lambda: model.load_weights(source_deltas.items())
    layer_path = layer_paths.pop()
    if layer_path is not None:
        try:
            layer = model.get_submodule(layer_path)
        except AttributeError as error:
            raise DeltaError(
                f"source delta names resolve to layer {layer_path!r}, but {type(model).__name__} has no such module"
            ) from error
        routed = route_values_to_scratch(layer, load)
    else:
        model_parameters = dict(model.named_parameters(remove_duplicate=False))
        missing = source_deltas.keys() - model_parameters.keys()
        if missing:
            raise DeltaError(f"non-layer source deltas have no direct vLLM destination: {sorted(missing)}")
        selected = [(name, model_parameters[name]) for name in source_deltas]
        routed = route_values_to_named_parameters(selected, load, context="dense non-layer parameters")

    apply_deltas_(routed)
    return len(routed)


def source_layer_module_path(name: str) -> str | None:
    """Return the destination module path for a conventional source parameter name."""
    return _layer_module_path(name)


def _layer_module_path(name: str) -> str | None:
    parts = name.split(".")
    for index in range(len(parts) - 1):
        if parts[index] in _LAYER_CONTAINER_NAMES and parts[index + 1].isdigit():
            return ".".join(parts[: index + 2])
    return None


def _enabled_expert_field(config, field: str) -> bool:
    value = getattr(config, field, None)
    return isinstance(value, int) and not isinstance(value, bool) and value > 1


__all__ = ["apply_dense_source_deltas_", "source_layer_module_path", "validate_dense_delta_model"]
