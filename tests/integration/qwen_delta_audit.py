"""Importable worker callback for the Qwen3 BF16 delta integration test."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from prime_rl.inference.vllm.worker.bf16_delta import (
    DestinationDelta,
    route_bf16_values_to_named_parameters,
    route_bf16_values_to_scratch,
    xor_bf16,
)
from prime_rl.inference.vllm.worker.qwen3_bf16_delta import apply_qwen3_bf16_source_deltas_


def audit_qwen3_bf16_delta(model: nn.Module) -> dict[str, Any]:
    """Audit TP-aware Qwen routing and in-place XOR inside a vLLM worker."""
    layer_index = 0
    layer = model.model.layers[layer_index]
    config = model.config
    live_parameters = dict(layer.named_parameters(remove_duplicate=False))
    head_dim = config.head_dim
    q_rows = config.num_attention_heads * head_dim
    kv_rows = config.num_key_value_heads * head_dim
    intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size
    prefix = f"model.layers.{layer_index}."
    source_shapes = {
        f"{prefix}self_attn.q_proj.weight": (q_rows, hidden_size),
        f"{prefix}self_attn.k_proj.weight": (kv_rows, hidden_size),
        f"{prefix}self_attn.v_proj.weight": (kv_rows, hidden_size),
        f"{prefix}self_attn.o_proj.weight": (hidden_size, q_rows),
        f"{prefix}self_attn.q_norm.weight": (head_dim,),
        f"{prefix}self_attn.k_norm.weight": (head_dim,),
        f"{prefix}mlp.gate_proj.weight": (intermediate_size, hidden_size),
        f"{prefix}mlp.up_proj.weight": (intermediate_size, hidden_size),
        f"{prefix}mlp.down_proj.weight": (hidden_size, intermediate_size),
        f"{prefix}input_layernorm.weight": (hidden_size,),
        f"{prefix}post_attention_layernorm.weight": (hidden_size,),
    }
    device = next(model.parameters()).device
    generators = {
        name: torch.Generator(device=device).manual_seed(index) for index, name in enumerate(source_shapes, start=1)
    }
    old_source = {
        name: torch.randint(
            -(2**15),
            2**15,
            shape,
            dtype=torch.int16,
            device=device,
            generator=generators[name],
        ).view(torch.bfloat16)
        for name, shape in source_shapes.items()
    }
    new_source = {
        name: xor_bf16(value, torch.ones_like(value.view(torch.int16)).view(torch.bfloat16))
        for name, value in old_source.items()
    }
    source_delta = {name: xor_bf16(old_source[name], new_source[name]) for name in old_source}

    def route(source: dict[str, torch.Tensor]) -> list[DestinationDelta]:
        return route_bf16_values_to_scratch(layer, lambda: model.load_weights(source.items()))

    def by_name(routed: list[DestinationDelta]) -> dict[str, torch.Tensor]:
        return {name: item.delta for item in routed for name in item.names}

    old_destinations = by_name(route(old_source))
    new_destinations = by_name(route(new_source))
    delta_destinations = by_name(route(source_delta))

    selected_destinations = {
        "self_attn.qkv_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    }
    for name in selected_destinations:
        expected = xor_bf16(old_destinations[name], new_destinations[name])
        if not torch.equal(delta_destinations[name].view(torch.int16), expected.view(torch.int16)):
            raise AssertionError(f"source XOR did not commute through Qwen's loader for {name}")

    live_before = {name: live_parameters[name].detach().clone() for name in selected_destinations}
    pointers_before = {name: live_parameters[name].data_ptr() for name in selected_destinations}
    apply_qwen3_bf16_source_deltas_(model, source_delta, layer_index=layer_index)
    try:
        for name in selected_destinations:
            parameter = live_parameters[name]
            if parameter.data_ptr() != pointers_before[name]:
                raise AssertionError(f"applying the delta replaced live storage for {name}")
            expected = xor_bf16(live_before[name], delta_destinations[name])
            if not torch.equal(parameter.view(torch.int16), expected.view(torch.int16)):
                raise AssertionError(f"applying the routed delta produced incorrect bytes for {name}")
    finally:
        apply_qwen3_bf16_source_deltas_(model, source_delta, layer_index=layer_index)

    for name in selected_destinations:
        parameter = live_parameters[name]
        if parameter.data_ptr() != pointers_before[name]:
            raise AssertionError(f"restoring the original bytes replaced live storage for {name}")
        if not torch.equal(parameter.view(torch.int16), live_before[name].view(torch.int16)):
            raise AssertionError(f"failed to restore the original bytes for {name}")

    model_parameters = dict(model.named_parameters(remove_duplicate=False))
    non_layer_shapes = {
        "model.embed_tokens.weight": (config.vocab_size, hidden_size),
        "model.norm.weight": (hidden_size,),
        "lm_head.weight": (config.vocab_size, hidden_size),
    }
    missing = non_layer_shapes.keys() - model_parameters.keys()
    if missing:
        raise AssertionError(f"Qwen non-layer audit cannot find vLLM parameters: {sorted(missing)}")
    non_layer_delta = {
        name: torch.ones(shape, dtype=torch.int16, device=device).view(torch.bfloat16)
        for name, shape in non_layer_shapes.items()
    }
    non_layer_routed = route_bf16_values_to_named_parameters(
        [(name, model_parameters[name]) for name in non_layer_delta],
        lambda: model.load_weights(non_layer_delta.items()),
        context="Qwen3 TP non-layer audit",
    )
    non_layer_by_name = {name: item.delta for item in non_layer_routed for name in item.names}
    non_layer_before = {name: model_parameters[name].detach().clone() for name in non_layer_delta}
    non_layer_pointers = {name: model_parameters[name].data_ptr() for name in non_layer_delta}
    apply_qwen3_bf16_source_deltas_(model, non_layer_delta, layer_index=-1)
    try:
        for name in non_layer_delta:
            parameter = model_parameters[name]
            if parameter.data_ptr() != non_layer_pointers[name]:
                raise AssertionError(f"applying the delta replaced live storage for {name}")
            expected = xor_bf16(non_layer_before[name], non_layer_by_name[name])
            if not torch.equal(parameter.view(torch.int16), expected.view(torch.int16)):
                raise AssertionError(f"applying the routed delta produced incorrect bytes for {name}")
    finally:
        apply_qwen3_bf16_source_deltas_(model, non_layer_delta, layer_index=-1)

    for name in non_layer_delta:
        parameter = model_parameters[name]
        if parameter.data_ptr() != non_layer_pointers[name]:
            raise AssertionError(f"restoring the original bytes replaced live storage for {name}")
        if not torch.equal(parameter.view(torch.int16), non_layer_before[name].view(torch.int16)):
            raise AssertionError(f"failed to restore the original bytes for {name}")

    return {
        "layer": layer_index,
        "destination_parameters_checked": len(selected_destinations),
        "source_tensors_checked": len(old_source),
        "non_layer_parameters_checked": len(non_layer_delta),
    }


__all__ = ["audit_qwen3_bf16_delta"]
