from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

_MODEL = "Qwen/Qwen3-0.6B-Base"


def _audit_qwen_delta(model: nn.Module) -> dict[str, Any]:
    """Run the BF16 XOR commutation test inside the vLLM worker."""
    from prime_rl.inference.vllm.worker.bf16_delta import (
        DestinationDelta,
        route_bf16_values_to_scratch,
        xor_bf16,
    )
    from prime_rl.inference.vllm.worker.qwen3_bf16_delta import apply_qwen3_bf16_source_deltas_

    layer_index = 0
    layer = model.model.layers[layer_index]
    config = model.config
    live_parameters = dict(layer.named_parameters(remove_duplicate=False))

    qkv = live_parameters["self_attn.qkv_proj.weight"].detach()
    head_dim = config.head_dim
    q_rows = config.num_attention_heads * head_dim
    kv_rows = config.num_key_value_heads * head_dim
    if qkv.shape[0] != q_rows + 2 * kv_rows:
        raise AssertionError(
            f"unexpected QKV shape {tuple(qkv.shape)} for q_rows={q_rows}, kv_rows={kv_rows}; "
            "this test requires tensor parallel size 1"
        )

    gate_up = live_parameters["mlp.gate_up_proj.weight"].detach()
    intermediate_size = config.intermediate_size
    if gate_up.shape[0] != 2 * intermediate_size:
        raise AssertionError(
            f"unexpected gate/up shape {tuple(gate_up.shape)} for intermediate_size={intermediate_size}; "
            "this test requires tensor parallel size 1"
        )

    prefix = f"model.layers.{layer_index}."
    old_source = {
        f"{prefix}self_attn.q_proj.weight": qkv[:q_rows].clone(),
        f"{prefix}self_attn.k_proj.weight": qkv[q_rows : q_rows + kv_rows].clone(),
        f"{prefix}self_attn.v_proj.weight": qkv[q_rows + kv_rows :].clone(),
        f"{prefix}self_attn.o_proj.weight": live_parameters["self_attn.o_proj.weight"].detach().clone(),
        f"{prefix}self_attn.q_norm.weight": live_parameters["self_attn.q_norm.weight"].detach().clone(),
        f"{prefix}self_attn.k_norm.weight": live_parameters["self_attn.k_norm.weight"].detach().clone(),
        f"{prefix}mlp.gate_proj.weight": gate_up[:intermediate_size].clone(),
        f"{prefix}mlp.up_proj.weight": gate_up[intermediate_size:].clone(),
        f"{prefix}mlp.down_proj.weight": live_parameters["mlp.down_proj.weight"].detach().clone(),
        f"{prefix}input_layernorm.weight": live_parameters["input_layernorm.weight"].detach().clone(),
        f"{prefix}post_attention_layernorm.weight": live_parameters["post_attention_layernorm.weight"].detach().clone(),
    }

    # Toggle one mantissa bit in every source value. The resulting source delta
    # has the BF16 bit pattern 0x0001 everywhere and must only be copied/routed.
    new_source = {}
    for name, value in old_source.items():
        bit_mask = torch.ones_like(value.view(torch.int16)).view(torch.bfloat16)
        new_source[name] = xor_bf16(value, bit_mask)
    source_delta = {name: xor_bf16(old_source[name], new_source[name]) for name in old_source}

    def route(source: dict[str, torch.Tensor]) -> list[DestinationDelta]:
        return route_bf16_values_to_scratch(layer, lambda: model.load_weights(source.items()))

    def by_name(routed: list[DestinationDelta]) -> dict[str, torch.Tensor]:
        return {name: item.delta for item in routed for name in item.names}

    old_destinations = by_name(route(old_source))
    new_destinations = by_name(route(new_source))
    routed_deltas = route(source_delta)
    delta_destinations = by_name(routed_deltas)

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

    pointers_before = {name: live_parameters[name].data_ptr() for name in selected_destinations}
    apply_qwen3_bf16_source_deltas_(model, source_delta, layer_index=layer_index)
    try:
        for name in selected_destinations:
            parameter = live_parameters[name]
            if parameter.data_ptr() != pointers_before[name]:
                raise AssertionError(f"applying the delta replaced live storage for {name}")
            if not torch.equal(parameter.view(torch.int16), new_destinations[name].view(torch.int16)):
                raise AssertionError(f"applying the routed delta produced incorrect bytes for {name}")
    finally:
        # XOR is its own inverse; do not leave the worker's live model modified.
        apply_qwen3_bf16_source_deltas_(model, source_delta, layer_index=layer_index)

    for name in selected_destinations:
        parameter = live_parameters[name]
        if parameter.data_ptr() != pointers_before[name]:
            raise AssertionError(f"restoring the original bytes replaced live storage for {name}")
        if not torch.equal(parameter.view(torch.int16), old_destinations[name].view(torch.int16)):
            raise AssertionError(f"failed to restore the original bytes for {name}")

    return {
        "layer": layer_index,
        "destination_parameters_checked": len(selected_destinations),
        "source_tensors_checked": len(old_source),
    }


def test_qwen3_bf16_source_xor_routes_and_applies_byte_exactly(monkeypatch: pytest.MonkeyPatch):
    # vLLM's apply_model API transports a Python callable to the worker. The
    # secure serializer in vLLM 0.26 requires explicit pickle opt-in for that.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from vllm import LLM

    llm = LLM(
        model=_MODEL,
        dtype="bfloat16",
        quantization=None,
        tensor_parallel_size=1,
        enforce_eager=True,
        skip_tokenizer_init=True,
        max_model_len=64,
        gpu_memory_utilization=0.5,
        disable_log_stats=True,
    )

    assert llm.apply_model(_audit_qwen_delta) == [
        {
            "layer": 0,
            "destination_parameters_checked": 8,
            "source_tensors_checked": 11,
        }
    ]
