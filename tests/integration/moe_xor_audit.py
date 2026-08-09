"""Importable vLLM worker callback for the GLM-4 MoE XOR loader audit."""

from __future__ import annotations

from math import prod
from typing import Any

import torch
from torch import nn

from prime_rl.inference.vllm.worker.xor_delta import apply_deltas_, route_values_to_scratch, xor_bits
from prime_rl.weight_sync.xor_delta import integer_view


def audit_glm4_moe_xor(model: nn.Module) -> dict[str, Any]:
    config = model.config
    if str(getattr(config, "model_type", "")) != "glm4_moe":
        raise AssertionError(f"GLM-4 MoE XOR audit got model_type={config.model_type!r}")

    layer_index = int(config.first_k_dense_replace)
    layer_path = f"model.layers.{layer_index}"
    layer = model.get_submodule(layer_path)
    source_specs = _glm4_moe_layer_source_specs(config, prefix=f"{layer_path}.")
    device = next(model.parameters()).device

    old_source = {
        name: _random_bits(shape, dtype=dtype, device=device, seed=index)
        for index, (name, (shape, dtype)) in enumerate(source_specs.items(), start=1)
    }
    new_source = {
        name: xor_bits(value, torch.ones_like(integer_view(value)).view(value.dtype))
        for name, value in old_source.items()
    }
    source_delta = {name: xor_bits(old_source[name], new_source[name]) for name in old_source}

    def route(source: dict[str, torch.Tensor]):
        return route_values_to_scratch(layer, lambda: model.load_weights(source.items()))

    old_destinations = _by_name(route(old_source))
    new_destinations = _by_name(route(new_source))
    delta_items = route(source_delta)
    delta_destinations = _by_name(delta_items)
    expected_names = set(dict(layer.named_parameters(remove_duplicate=False)))
    if old_destinations.keys() != new_destinations.keys() or old_destinations.keys() != delta_destinations.keys():
        raise AssertionError("GLM-4 MoE loader routed inconsistent destination sets")
    if old_destinations.keys() != expected_names:
        missing = expected_names - old_destinations.keys()
        unexpected = old_destinations.keys() - expected_names
        raise AssertionError(
            f"GLM-4 MoE loader coverage mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    for name in expected_names:
        expected = xor_bits(old_destinations[name], new_destinations[name])
        if not torch.equal(integer_view(delta_destinations[name]), integer_view(expected)):
            raise AssertionError(f"source XOR did not commute through the GLM-4 MoE loader for {name}")

    live_parameters = dict(layer.named_parameters(remove_duplicate=False))
    live_before = {name: parameter.detach().clone() for name, parameter in live_parameters.items()}
    pointers_before = {name: parameter.data_ptr() for name, parameter in live_parameters.items()}
    apply_deltas_(delta_items)
    try:
        for name, parameter in live_parameters.items():
            if parameter.data_ptr() != pointers_before[name]:
                raise AssertionError(f"applying the GLM-4 MoE delta replaced live storage for {name}")
            expected = xor_bits(live_before[name], delta_destinations[name])
            if not torch.equal(integer_view(parameter), integer_view(expected)):
                raise AssertionError(f"applying the GLM-4 MoE delta produced incorrect bytes for {name}")
    finally:
        apply_deltas_(delta_items)

    for name, parameter in live_parameters.items():
        if parameter.data_ptr() != pointers_before[name]:
            raise AssertionError(f"restoring GLM-4 MoE bytes replaced live storage for {name}")
        if not torch.equal(integer_view(parameter), integer_view(live_before[name])):
            raise AssertionError(f"failed to restore original GLM-4 MoE bytes for {name}")

    local_expert_parameters = sum("experts" in name for name in expected_names)
    if local_expert_parameters == 0:
        raise AssertionError("GLM-4 MoE XOR audit found no local expert parameters")
    return {
        "family": "glm4_moe",
        "layer": layer_index,
        "destination_parameters_checked": len(expected_names),
        "source_tensors_checked": len(source_specs),
        "local_expert_parameters_checked": local_expert_parameters,
    }


def _glm4_moe_layer_source_specs(
    config,
    *,
    prefix: str,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    hidden_size = int(config.hidden_size)
    head_dim = int(config.head_dim)
    query_size = int(config.num_attention_heads) * head_dim
    kv_size = int(config.num_key_value_heads) * head_dim
    expert_size = int(config.moe_intermediate_size)
    shared_expert_size = expert_size * int(config.n_shared_experts)
    model_dtype = torch.bfloat16

    specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        f"{prefix}self_attn.q_proj.weight": ((query_size, hidden_size), model_dtype),
        f"{prefix}self_attn.k_proj.weight": ((kv_size, hidden_size), model_dtype),
        f"{prefix}self_attn.v_proj.weight": ((kv_size, hidden_size), model_dtype),
        f"{prefix}self_attn.o_proj.weight": ((hidden_size, query_size), model_dtype),
        f"{prefix}input_layernorm.weight": ((hidden_size,), model_dtype),
        f"{prefix}post_attention_layernorm.weight": ((hidden_size,), model_dtype),
        f"{prefix}mlp.gate.weight": ((int(config.n_routed_experts), hidden_size), torch.float32),
        f"{prefix}mlp.gate.e_score_correction_bias": ((int(config.n_routed_experts),), torch.float32),
        f"{prefix}mlp.shared_experts.gate_proj.weight": ((shared_expert_size, hidden_size), model_dtype),
        f"{prefix}mlp.shared_experts.up_proj.weight": ((shared_expert_size, hidden_size), model_dtype),
        f"{prefix}mlp.shared_experts.down_proj.weight": ((hidden_size, shared_expert_size), model_dtype),
    }
    if bool(config.attention_bias):
        specs.update(
            {
                f"{prefix}self_attn.q_proj.bias": ((query_size,), model_dtype),
                f"{prefix}self_attn.k_proj.bias": ((kv_size,), model_dtype),
                f"{prefix}self_attn.v_proj.bias": ((kv_size,), model_dtype),
            }
        )
    for expert_index in range(int(config.n_routed_experts)):
        expert_prefix = f"{prefix}mlp.experts.{expert_index}"
        specs.update(
            {
                f"{expert_prefix}.gate_proj.weight": ((expert_size, hidden_size), model_dtype),
                f"{expert_prefix}.up_proj.weight": ((expert_size, hidden_size), model_dtype),
                f"{expert_prefix}.down_proj.weight": ((hidden_size, expert_size), model_dtype),
            }
        )
    return specs


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
        torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=device, generator=generator)
        .view(dtype)
        .view(shape)
    )


def _by_name(routed) -> dict[str, torch.Tensor]:
    return {name: item.delta for item in routed for name in item.names}


__all__ = ["audit_glm4_moe_xor"]
