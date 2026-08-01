from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch
from torch import nn

from prime_rl.inference.vllm.worker.bf16_delta import (
    BF16DeltaError,
    DestinationDelta,
    apply_bf16_deltas_,
    route_bf16_values_to_named_parameters,
    route_bf16_values_to_scratch,
    xor_bf16,
)


class _PackedParameter(nn.Parameter):
    """Stand-in for vLLM Parameter subclasses with specialized load methods."""

    def load_shard(
        self,
        loaded_weight: torch.Tensor,
        shard_id: str | int,
        output_sizes: dict[str | int, int],
    ) -> None:
        shard_offset = 0
        for candidate, shard_size in output_sizes.items():
            if candidate == shard_id:
                destination = self.data.narrow(0, shard_offset, shard_size)
                assert destination.shape == loaded_weight.shape
                destination.copy_(loaded_weight)
                return
            shard_offset += shard_size
        raise ValueError(f"unknown shard {shard_id!r}")


class _PackedLinear(nn.Module):
    """Small stand-in for vLLM's QKV and merged-column weight loaders."""

    def __init__(self, input_size: int, output_sizes: dict[str | int, int]):
        super().__init__()
        self.output_sizes = output_sizes
        self.weight = _PackedParameter(
            torch.zeros((sum(output_sizes.values()), input_size), dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str | int) -> None:
        assert isinstance(param, _PackedParameter)
        param.load_shard(loaded_weight, shard_id, self.output_sizes)


class _ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = _PackedLinear(4, {"q": 4, "k": 2, "v": 2})
        self.o_proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)


class _ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = _PackedLinear(4, {0: 5, 1: 5})
        self.down_proj = nn.Linear(5, 4, bias=False, dtype=torch.bfloat16)


class _ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _ToyAttention()
        self.mlp = _ToyMLP()
        self.input_layernorm = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16), requires_grad=False)
        self.post_attention_layernorm = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16), requires_grad=False)


class _ToyVllmModel(nn.Module):
    """Uses Qwen-like checkpoint names and fused vLLM-like destinations."""

    _STACKED_MAPPING = (
        ("self_attn.qkv_proj.weight", "self_attn.q_proj.weight", "q"),
        ("self_attn.qkv_proj.weight", "self_attn.k_proj.weight", "k"),
        ("self_attn.qkv_proj.weight", "self_attn.v_proj.weight", "v"),
        ("mlp.gate_up_proj.weight", "mlp.gate_proj.weight", 0),
        ("mlp.gate_up_proj.weight", "mlp.up_proj.weight", 1),
    )

    def __init__(self):
        super().__init__()
        self.layer = _ToyLayer()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.layer.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        for source_name, value in weights:
            for destination_name, mapped_source_name, shard_id in self._STACKED_MAPPING:
                if source_name != mapped_source_name:
                    continue
                parameter = params[destination_name]
                parameter.weight_loader(parameter, value, shard_id)
                loaded.add(destination_name)
                break
            else:
                parameter = params[source_name]
                parameter.data.copy_(value)
                loaded.add(source_name)
        return loaded


def _random_bf16_bits(shape: tuple[int, ...], *, generator: torch.Generator) -> torch.Tensor:
    # Generate arbitrary bit patterns rather than finite floating-point values.
    # This catches loaders that accidentally perform arithmetic or canonicalize NaNs.
    bits = torch.randint(-(2**15), 2**15, shape, dtype=torch.int32, generator=generator).to(torch.int16)
    return bits.contiguous().view(torch.bfloat16)


def _make_source_state(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shapes = {
        "self_attn.q_proj.weight": (4, 4),
        "self_attn.k_proj.weight": (2, 4),
        "self_attn.v_proj.weight": (2, 4),
        "self_attn.o_proj.weight": (4, 4),
        "mlp.gate_proj.weight": (5, 4),
        "mlp.up_proj.weight": (5, 4),
        "mlp.down_proj.weight": (4, 5),
        "input_layernorm": (4,),
        "post_attention_layernorm": (4,),
    }
    return {name: _random_bf16_bits(shape, generator=generator) for name, shape in shapes.items()}


def _route(model: _ToyVllmModel, state: dict[str, torch.Tensor]) -> list[DestinationDelta]:
    return route_bf16_values_to_scratch(model.layer, lambda: model.load_weights(state.items()))


def _by_name(deltas: list[DestinationDelta]) -> dict[str, torch.Tensor]:
    return {item.names[0]: item.delta for item in deltas}


def _live_by_name(model: _ToyVllmModel) -> dict[str, nn.Parameter]:
    return dict(model.layer.named_parameters(remove_duplicate=False))


def test_source_xor_routes_to_exact_destination_xor_and_updates_in_place():
    model = _ToyVllmModel()
    old_source = _make_source_state(seed=1)
    new_source = _make_source_state(seed=2)
    source_delta = {name: xor_bf16(old_source[name], new_source[name]) for name in old_source}

    old_destinations = _by_name(_route(model, old_source))
    new_destinations = _by_name(_route(model, new_source))
    routed_deltas = _route(model, source_delta)
    delta_destinations = _by_name(routed_deltas)

    assert old_destinations.keys() == new_destinations.keys() == delta_destinations.keys()
    for name in old_destinations:
        expected = xor_bf16(old_destinations[name], new_destinations[name])
        assert torch.equal(delta_destinations[name].view(torch.int16), expected.view(torch.int16)), name

    model.load_weights(old_source.items())
    live_parameters = _live_by_name(model)
    pointers_before = {name: parameter.data_ptr() for name, parameter in live_parameters.items()}

    apply_bf16_deltas_(routed_deltas)

    for name, parameter in live_parameters.items():
        assert parameter.data_ptr() == pointers_before[name]
        assert torch.equal(parameter.view(torch.int16), new_destinations[name].view(torch.int16)), name


def test_routing_restores_live_parameters_when_loader_raises():
    model = _ToyVllmModel()
    live_before = _live_by_name(model)
    pointers_before = {name: parameter.data_ptr() for name, parameter in live_before.items()}

    def failing_loader() -> None:
        model.load_weights(_make_source_state(seed=3).items())
        raise RuntimeError("loader failed")

    with pytest.raises(RuntimeError, match="loader failed"):
        route_bf16_values_to_scratch(model.layer, failing_loader)

    live_after = _live_by_name(model)
    assert live_after.keys() == live_before.keys()
    for name in live_before:
        assert live_after[name] is live_before[name]
        assert live_after[name].data_ptr() == pointers_before[name]


def test_routing_rejects_non_bf16_destination_before_calling_loader():
    layer = nn.Linear(4, 4, bias=False, dtype=torch.float32)
    called = False

    def loader() -> None:
        nonlocal called
        called = True

    with pytest.raises(BF16DeltaError, match="only torch.bfloat16 is supported"):
        route_bf16_values_to_scratch(layer, loader)
    assert not called


def test_xor_bf16_rejects_shape_and_dtype_mismatches():
    value = torch.zeros((2, 2), dtype=torch.bfloat16)

    with pytest.raises(BF16DeltaError, match="shape mismatch"):
        xor_bf16(value, torch.zeros(3, dtype=torch.bfloat16))

    with pytest.raises(BF16DeltaError, match="requires torch.bfloat16"):
        xor_bf16(value, torch.zeros((2, 2), dtype=torch.float32))


def test_explicit_parameter_selection_routes_without_swapping_other_parameters():
    model = _ToyVllmModel()
    parameter = model.layer.input_layernorm
    other_parameter = model.layer.post_attention_layernorm
    other_pointer = other_parameter.data_ptr()
    value = _make_source_state(seed=4)["input_layernorm"]

    routed = route_bf16_values_to_named_parameters(
        [("input_layernorm", parameter)],
        lambda: model.load_weights([("input_layernorm", value)]),
    )

    assert len(routed) == 1
    assert routed[0].parameter is parameter
    assert torch.equal(routed[0].delta.view(torch.int16), value.view(torch.int16))
    assert other_parameter.data_ptr() == other_pointer
