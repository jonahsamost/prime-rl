from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prime_rl.inference.vllm.worker.dense_xor import (
    apply_dense_source_deltas_,
    source_layer_module_path,
    validate_dense_delta_model,
)
from prime_rl.inference.vllm.worker.xor_delta import (
    DeltaError,
    DestinationDelta,
    apply_deltas_,
    route_values_to_named_parameters,
    route_values_to_scratch,
    xor_bits,
)
from prime_rl.weight_sync.xor_delta import delta_dtype_nbytes, integer_view


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

    def __init__(self, input_size: int, output_sizes: dict[str | int, int], dtype: torch.dtype):
        super().__init__()
        self.output_sizes = output_sizes
        self.weight = _PackedParameter(
            torch.zeros((sum(output_sizes.values()), input_size), dtype=dtype),
            requires_grad=False,
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str | int) -> None:
        assert isinstance(param, _PackedParameter)
        param.load_shard(loaded_weight, shard_id, self.output_sizes)


class _ToyAttention(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.qkv_proj = _PackedLinear(4, {"q": 4, "k": 2, "v": 2}, dtype)
        self.o_proj = nn.Linear(4, 4, bias=False, dtype=dtype)


class _ToyMLP(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.gate_up_proj = _PackedLinear(4, {0: 5, 1: 5}, dtype)
        self.down_proj = nn.Linear(5, 4, bias=False, dtype=dtype)


class _ToyLayer(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.self_attn = _ToyAttention(dtype)
        self.mlp = _ToyMLP(dtype)
        self.input_layernorm = nn.Parameter(torch.zeros(4, dtype=dtype), requires_grad=False)
        self.post_attention_layernorm = nn.Parameter(torch.zeros(4, dtype=dtype), requires_grad=False)


class _ToyVllmModel(nn.Module):
    """Uses Qwen-like checkpoint names and fused vLLM-like destinations."""

    _STACKED_MAPPING = (
        ("self_attn.qkv_proj.weight", "self_attn.q_proj.weight", "q"),
        ("self_attn.qkv_proj.weight", "self_attn.k_proj.weight", "k"),
        ("self_attn.qkv_proj.weight", "self_attn.v_proj.weight", "v"),
        ("mlp.gate_up_proj.weight", "mlp.gate_proj.weight", 0),
        ("mlp.gate_up_proj.weight", "mlp.up_proj.weight", 1),
    )

    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.layer = _ToyLayer(dtype)

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


class _ToyLayerStack(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer(dtype)])


class _ToyDenseVllmModel(_ToyVllmModel):
    def __init__(self, dtype: torch.dtype):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(model_type="llama")
        self.model = _ToyLayerStack(dtype)

    @property
    def layer(self) -> _ToyLayer:
        return self.model.layers[0]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        prefix = "model.layers.0."
        normalized = []
        for name, value in weights:
            if not name.startswith(prefix):
                raise ValueError(f"unexpected source name {name!r}")
            normalized.append((name.removeprefix(prefix), value))
        return super().load_weights(normalized)


def _random_bits(shape: tuple[int, ...], dtype: torch.dtype, *, generator: torch.Generator) -> torch.Tensor:
    # Generate arbitrary bit patterns rather than finite floating-point values.
    # This catches loaders that accidentally perform arithmetic or canonicalize NaNs.
    nbytes = int(torch.tensor(shape).prod().item()) * delta_dtype_nbytes(dtype)
    return torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=generator).view(dtype).view(shape)


def _make_source_state(seed: int, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
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
    return {name: _random_bits(shape, dtype, generator=generator) for name, shape in shapes.items()}


def _route(model: _ToyVllmModel, state: dict[str, torch.Tensor]) -> list[DestinationDelta]:
    return route_values_to_scratch(model.layer, lambda: model.load_weights(state.items()))


def _by_name(deltas: list[DestinationDelta]) -> dict[str, torch.Tensor]:
    return {item.names[0]: item.delta for item in deltas}


def _live_by_name(model: _ToyVllmModel) -> dict[str, nn.Parameter]:
    return dict(model.layer.named_parameters(remove_duplicate=False))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_source_xor_routes_to_exact_destination_xor_and_updates_in_place(dtype):
    model = _ToyVllmModel(dtype)
    old_source = _make_source_state(seed=1, dtype=dtype)
    new_source = _make_source_state(seed=2, dtype=dtype)
    source_delta = {name: xor_bits(old_source[name], new_source[name]) for name in old_source}

    old_destinations = _by_name(_route(model, old_source))
    new_destinations = _by_name(_route(model, new_source))
    routed_deltas = _route(model, source_delta)
    delta_destinations = _by_name(routed_deltas)

    assert old_destinations.keys() == new_destinations.keys() == delta_destinations.keys()
    for name in old_destinations:
        expected = xor_bits(old_destinations[name], new_destinations[name])
        assert torch.equal(integer_view(delta_destinations[name]), integer_view(expected)), name

    model.load_weights(old_source.items())
    live_parameters = _live_by_name(model)
    pointers_before = {name: parameter.data_ptr() for name, parameter in live_parameters.items()}

    apply_deltas_(routed_deltas)

    for name, parameter in live_parameters.items():
        assert parameter.data_ptr() == pointers_before[name]
        assert torch.equal(integer_view(parameter), integer_view(new_destinations[name])), name


def test_routing_restores_live_parameters_when_loader_raises():
    model = _ToyVllmModel()
    live_before = _live_by_name(model)
    pointers_before = {name: parameter.data_ptr() for name, parameter in live_before.items()}

    def failing_loader() -> None:
        model.load_weights(_make_source_state(seed=3).items())
        raise RuntimeError("loader failed")

    with pytest.raises(RuntimeError, match="loader failed"):
        route_values_to_scratch(model.layer, failing_loader)

    live_after = _live_by_name(model)
    assert live_after.keys() == live_before.keys()
    for name in live_before:
        assert live_after[name] is live_before[name]
        assert live_after[name].data_ptr() == pointers_before[name]


def test_routing_rejects_unsupported_destination_before_calling_loader():
    layer = nn.Linear(4, 4, bias=False, dtype=torch.float64)
    called = False

    def loader() -> None:
        nonlocal called
        called = True

    with pytest.raises(DeltaError, match="unsupported dtype"):
        route_values_to_scratch(layer, loader)
    assert not called


def test_xor_bits_rejects_shape_and_dtype_mismatches():
    value = torch.zeros((2, 2), dtype=torch.bfloat16)

    with pytest.raises(DeltaError, match="shape mismatch"):
        xor_bits(value, torch.zeros(3, dtype=torch.bfloat16))

    with pytest.raises(DeltaError, match="matching supported dtypes"):
        xor_bits(value, torch.zeros((2, 2), dtype=torch.float32))


def test_explicit_parameter_selection_routes_without_swapping_other_parameters():
    model = _ToyVllmModel()
    parameter = model.layer.input_layernorm
    other_parameter = model.layer.post_attention_layernorm
    other_pointer = other_parameter.data_ptr()
    value = _make_source_state(seed=4)["input_layernorm"]

    routed = route_values_to_named_parameters(
        [("input_layernorm", parameter)],
        lambda: model.load_weights([("input_layernorm", value)]),
    )

    assert len(routed) == 1
    assert routed[0].parameter is parameter
    assert torch.equal(integer_view(routed[0].delta), integer_view(value))
    assert other_parameter.data_ptr() == other_pointer


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_dense_adapter_applies_conventional_layer_deltas(dtype):
    model = _ToyDenseVllmModel(dtype)
    prefix = "model.layers.0."
    old_source = {prefix + name: value for name, value in _make_source_state(10, dtype).items()}
    new_source = {prefix + name: value for name, value in _make_source_state(11, dtype).items()}
    source_delta = {name: xor_bits(old_source[name], new_source[name]) for name in old_source}
    model.load_weights(old_source.items())

    apply_dense_source_deltas_(model, source_delta)

    expected = _by_name(route_values_to_scratch(model.layer, lambda: model.load_weights(new_source.items())))
    for name, parameter in _live_by_name(model).items():
        assert torch.equal(integer_view(parameter), integer_view(expected[name])), name


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.layers.12.self_attn.q_proj.weight", "model.layers.12"),
        ("transformer.h.3.attn.c_attn.weight", "transformer.h.3"),
        ("decoder.blocks.7.mlp.weight", "decoder.blocks.7"),
        ("model.embed_tokens.weight", None),
    ],
)
def test_source_layer_module_path(name, expected):
    assert source_layer_module_path(name) == expected


def test_dense_adapter_rejects_moe_models():
    model = _ToyDenseVllmModel(torch.bfloat16)
    model.config.num_local_experts = 8

    with pytest.raises(DeltaError, match="does not support MoE"):
        validate_dense_delta_model(model)
