"""XOR-delta routing for dense vLLM models with Hugging Face-style layer names."""

from __future__ import annotations

import torch
from torch import nn

from prime_rl.inference.vllm.worker.xor_delta import (
    DeltaError,
    ParameterRoutingPlan,
    apply_deltas_,
    named_parameter_routing_plan,
    parameter_routing_plan,
    route_values_with_plan,
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
    # Intentionally conservative for now to require one storage dtype.
    # load_weights() may cast between src/dst dtypes but floating point casts dont preserve xor deltas
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if len(dtypes) != 1:
        raise DeltaError(f"XOR delta loading requires one model storage dtype, got {sorted(map(str, dtypes))}")
    dtype = dtypes.pop()
    if dtype not in SUPPORTED_DELTA_DTYPES:
        raise DeltaError(f"XOR delta loading does not support model storage dtype {dtype}")
    return dtype


class DenseDeltaRouter:
    """Cache model-layout routing plans while applying changing XOR values."""

    def __init__(self, model: nn.Module, model_dtype: torch.dtype) -> None:
        self.model = model
        self.model_dtype = model_dtype
        self._model_parameters = dict(model.named_parameters(remove_duplicate=False))
        self._plans: dict[tuple[str | None, tuple[str, ...]], ParameterRoutingPlan] = {}

    @torch.no_grad()
    def apply(self, source_deltas: dict[str, torch.Tensor], *, layer_path: str | None) -> int:
        if not source_deltas:
            return 0
        source_dtypes = {value.dtype for value in source_deltas.values()}
        if source_dtypes != {self.model_dtype}:
            raise DeltaError(
                f"source delta dtypes {sorted(map(str, source_dtypes))} "
                f"do not match model storage dtype {self.model_dtype}"
            )

        source_names = tuple(source_deltas)
        key = (layer_path, source_names)
        plan = self._plans.get(key)
        if plan is None:
            actual_paths = {_layer_module_path(name) for name in source_names}
            if actual_paths != {layer_path}:
                raise DeltaError(
                    f"XOR delta group expected layer {layer_path!r}, got "
                    f"{sorted(actual_paths, key=str)}"
                )
            plan = self._make_plan(source_names, layer_path)
            self._plans[key] = plan

        load = lambda: self.model.load_weights(source_deltas.items())
        routed = route_values_with_plan(plan, load)
        apply_deltas_(routed)
        return len(routed)

    def _make_plan(self, source_names: tuple[str, ...], layer_path: str | None) -> ParameterRoutingPlan:
        if layer_path is not None:
            try:
                layer = self.model.get_submodule(layer_path)
            except AttributeError as error:
                raise DeltaError(
                    f"source delta names resolve to layer {layer_path!r}, "
                    f"but {type(self.model).__name__} has no such module"
                ) from error
            return parameter_routing_plan(layer)

        missing = set(source_names) - self._model_parameters.keys()
        if missing:
            raise DeltaError(f"non-layer source deltas have no direct vLLM destination: {sorted(missing)}")
        selected = [(name, self._model_parameters[name]) for name in source_names]
        return named_parameter_routing_plan(selected, context="dense non-layer parameters")


@torch.no_grad()
def apply_dense_source_deltas_(
    model: nn.Module,
    source_deltas: dict[str, torch.Tensor],
    *,
    model_dtype: torch.dtype,
) -> int:
    """Route one source-layout parameter group and XOR it into live vLLM weights."""
    if not source_deltas:
        return 0
    layer_paths = {_layer_module_path(name) for name in source_deltas}
    if len(layer_paths) != 1:
        raise DeltaError(f"XOR delta group spans multiple layer modules: {sorted(layer_paths, key=str)}")
    layer_path = layer_paths.pop()
    return DenseDeltaRouter(model, model_dtype).apply(source_deltas, layer_path=layer_path)


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


__all__ = [
    "DenseDeltaRouter",
    "apply_dense_source_deltas_",
    "source_layer_module_path",
    "validate_dense_delta_model",
]
