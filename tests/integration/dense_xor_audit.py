"""Importable vLLM worker callbacks for dense-model XOR loader audits."""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any

import torch
from torch import nn

from prime_rl.inference.vllm.worker.dense_xor import apply_dense_source_deltas_, validate_dense_delta_model
from prime_rl.inference.vllm.worker.xor_delta import (
    DestinationDelta,
    route_values_to_named_parameters,
    route_values_to_scratch,
    xor_bits,
)
from prime_rl.weight_sync.xor_delta import integer_view

_STANDARD_MODEL_TYPES = {
    "qwen3": frozenset({"qwen3"}),
    "llama3": frozenset({"llama"}),
    "gemma": frozenset({"gemma2", "gemma3", "gemma3_text"}),
    # vLLM's Mistral-format loader exposes its runtime config as "transformer".
    "mistral": frozenset({"mistral", "transformer"}),
}


def audit_qwen3_xor(model: nn.Module) -> dict[str, Any]:
    return audit_standard_decoder_xor(model, family="qwen3")


def audit_llama3_xor(model: nn.Module) -> dict[str, Any]:
    return audit_standard_decoder_xor(model, family="llama3")


def audit_gemma_xor(model: nn.Module) -> dict[str, Any]:
    return audit_standard_decoder_xor(model, family="gemma")


def audit_mistral_xor(model: nn.Module) -> dict[str, Any]:
    return audit_standard_decoder_xor(model, family="mistral")


def audit_standard_decoder_xor(model: nn.Module, *, family: str) -> dict[str, Any]:
    """Audit a conventional gated-decoder checkpoint loader and in-place XOR."""
    model_dtype = validate_dense_delta_model(model)
    config = model.config
    model_type = str(getattr(config, "model_type", ""))
    expected_model_types = _STANDARD_MODEL_TYPES[family]
    if model_type not in expected_model_types:
        raise AssertionError(
            f"{family} XOR audit expected model_type in {sorted(expected_model_types)}, got {model_type!r}"
        )

    layer_path = "model.layers.0"
    layer = model.get_submodule(layer_path)
    text_config = _text_config(config)
    source_shapes = _standard_layer_source_shapes(layer, text_config, prefix=f"{layer_path}.")
    layer_result = _audit_source_group(
        model,
        source_shapes,
        model_dtype=model_dtype,
        destination_root=layer,
        context=f"{family} layer 0",
    )

    model_parameters = dict(model.named_parameters(remove_duplicate=False))
    hidden_size = int(text_config.hidden_size)
    vocab_size = int(text_config.vocab_size)
    candidate_non_layer_shapes = {
        "model.embed_tokens.weight": (vocab_size, hidden_size),
        "model.norm.weight": (hidden_size,),
    }
    if bool(getattr(text_config, "tie_word_embeddings", False)):
        embedding = model_parameters.get("model.embed_tokens.weight")
        lm_head = model_parameters.get("lm_head.weight")
        if embedding is not None and lm_head is not None and embedding is not lm_head:
            raise AssertionError(f"{family} declares tied word embeddings but vLLM destinations are not aliases")
    else:
        candidate_non_layer_shapes["lm_head.weight"] = (vocab_size, hidden_size)

    non_layer_shapes = {
        name: shape
        for name, shape in candidate_non_layer_shapes.items()
        if name in model_parameters
    }
    if not non_layer_shapes:
        raise AssertionError(f"{family} XOR audit found no conventional non-layer parameters")

    non_layer_destinations = 0
    for name, shape in non_layer_shapes.items():
        result = _audit_source_group(
            model,
            {name: shape},
            model_dtype=model_dtype,
            destination_root=None,
            context=f"{family} {name}",
        )
        non_layer_destinations += result["destination_parameters_checked"]

    return {
        "family": family,
        "layer": 0,
        "destination_parameters_checked": layer_result["destination_parameters_checked"],
        "source_tensors_checked": layer_result["source_tensors_checked"],
        "non_layer_parameters_checked": non_layer_destinations,
    }


def _standard_layer_source_shapes(
    layer: nn.Module,
    config,
    *,
    prefix: str,
) -> dict[str, tuple[int, ...]]:
    live = dict(layer.named_parameters(remove_duplicate=False))
    hidden_size = int(config.hidden_size)
    num_attention_heads = int(config.num_attention_heads)
    num_key_value_heads = int(getattr(config, "num_key_value_heads", None) or num_attention_heads)
    head_dim = int(getattr(config, "head_dim", None) or hidden_size // num_attention_heads)
    intermediate_size = int(config.intermediate_size)

    shapes: dict[str, tuple[int, ...]] = {
        f"{prefix}self_attn.q_proj.weight": (num_attention_heads * head_dim, hidden_size),
        f"{prefix}self_attn.k_proj.weight": (num_key_value_heads * head_dim, hidden_size),
        f"{prefix}self_attn.v_proj.weight": (num_key_value_heads * head_dim, hidden_size),
        f"{prefix}self_attn.o_proj.weight": (hidden_size, num_attention_heads * head_dim),
        f"{prefix}mlp.gate_proj.weight": (intermediate_size, hidden_size),
        f"{prefix}mlp.up_proj.weight": (intermediate_size, hidden_size),
        f"{prefix}mlp.down_proj.weight": (hidden_size, intermediate_size),
    }
    packed_destinations = {"self_attn.qkv_proj.weight", "mlp.gate_up_proj.weight"}
    standard_destinations = {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
    for name, parameter in live.items():
        if name in packed_destinations or name in standard_destinations:
            continue
        if parameter.ndim != 1:
            raise AssertionError(
                f"standard dense XOR audit needs an explicit source shape for {prefix}{name}: "
                f"destination shape={tuple(parameter.shape)}"
            )
        shapes[f"{prefix}{name}"] = tuple(parameter.shape)
    return shapes


def _audit_source_group(
    model: nn.Module,
    source_shapes: dict[str, tuple[int, ...]],
    *,
    model_dtype: torch.dtype,
    destination_root: nn.Module | None,
    context: str,
) -> dict[str, int]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    old_source = {
        name: _random_bits(shape, dtype=dtype, device=device, seed=index)
        for index, (name, shape) in enumerate(source_shapes.items(), start=1)
    }
    new_source = {
        name: xor_bits(value, torch.ones_like(integer_view(value)).view(dtype)) for name, value in old_source.items()
    }
    source_delta = {name: xor_bits(old_source[name], new_source[name]) for name in old_source}

    if destination_root is None:
        model_parameters = dict(model.named_parameters(remove_duplicate=False))
        selected = source_shapes.keys() & model_parameters.keys()
        if selected != source_shapes.keys():
            missing = source_shapes.keys() - selected
            raise AssertionError(f"{context} has no direct vLLM destination for {sorted(missing)}")
        destination_parameters = {name: model_parameters[name] for name in selected}
        route: Callable[[dict[str, torch.Tensor]], list[DestinationDelta]] = lambda source: _route_named_group(
            model,
            destination_parameters,
            source,
            context=context,
        )
    else:
        destination_parameters = dict(destination_root.named_parameters(remove_duplicate=False))
        route = lambda source: route_values_to_scratch(
            destination_root,
            lambda: model.load_weights(source.items()),
        )

    old_destinations = _by_name(route(old_source))
    new_destinations = _by_name(route(new_source))
    delta_destinations = _by_name(route(source_delta))
    expected_names = set(destination_parameters)
    if old_destinations.keys() != new_destinations.keys() or old_destinations.keys() != delta_destinations.keys():
        raise AssertionError(f"{context} routed inconsistent destination sets")
    if old_destinations.keys() != expected_names:
        raise AssertionError(
            f"{context} routed destinations {sorted(old_destinations)}; expected {sorted(expected_names)}"
        )

    for name in expected_names:
        if not torch.count_nonzero(integer_view(old_destinations[name])):
            raise AssertionError(f"{context} did not load destination {name}")
        expected = xor_bits(old_destinations[name], new_destinations[name])
        if not torch.equal(integer_view(delta_destinations[name]), integer_view(expected)):
            raise AssertionError(f"source XOR did not commute through {context} loader for {name}")

    live_before = {name: parameter.detach().clone() for name, parameter in destination_parameters.items()}
    pointers_before = {name: parameter.data_ptr() for name, parameter in destination_parameters.items()}
    apply_dense_source_deltas_(model, source_delta, model_dtype=model_dtype)
    try:
        for name, parameter in destination_parameters.items():
            if parameter.data_ptr() != pointers_before[name]:
                raise AssertionError(f"applying {context} delta replaced live storage for {name}")
            expected = xor_bits(live_before[name], delta_destinations[name])
            if not torch.equal(integer_view(parameter), integer_view(expected)):
                raise AssertionError(f"applying {context} delta produced incorrect bytes for {name}")
    finally:
        apply_dense_source_deltas_(model, source_delta, model_dtype=model_dtype)

    for name, parameter in destination_parameters.items():
        if parameter.data_ptr() != pointers_before[name]:
            raise AssertionError(f"restoring {context} bytes replaced live storage for {name}")
        if not torch.equal(integer_view(parameter), integer_view(live_before[name])):
            raise AssertionError(f"failed to restore original {context} bytes for {name}")

    return {
        "destination_parameters_checked": len(destination_parameters),
        "source_tensors_checked": len(source_shapes),
    }


def _route_named_group(
    model: nn.Module,
    destination_parameters: dict[str, nn.Parameter],
    source: dict[str, torch.Tensor],
    *,
    context: str,
) -> list[DestinationDelta]:
    return route_values_to_named_parameters(
        destination_parameters.items(),
        lambda: model.load_weights(source.items()),
        context=context,
    )


def _random_bits(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    nbytes = prod(shape) * torch.empty((), dtype=dtype).element_size()
    return (
        torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=device, generator=generator).view(dtype).view(shape)
    )


def _by_name(routed: list[DestinationDelta]) -> dict[str, torch.Tensor]:
    return {name: item.delta for item in routed for name in item.names}


def _text_config(config):
    get_text_config = getattr(config, "get_text_config", None)
    return get_text_config() if callable(get_text_config) else config


__all__ = [
    "audit_gemma_xor",
    "audit_llama3_xor",
    "audit_mistral_xor",
    "audit_qwen3_xor",
    "audit_standard_decoder_xor",
]
