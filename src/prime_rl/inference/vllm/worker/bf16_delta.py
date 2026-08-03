"""Experimental BF16 XOR-delta routing for vLLM weight loaders.

This module contains the receiver-side primitive for proving that a model's
checkpoint loader is bit preserving.  It intentionally does not implement a
transport, compression, optimizer integration, or quantized-weight support.

The caller selects one bounded layer, temporarily routes source-layout values
through the model's existing ``load_weights`` implementation into zeroed
scratch storage, and then restores the live storage.  If the loader only
performs same-dtype byte movement (slicing, concatenation into fused weights,
TP selection, and copies), routing an XOR value produces the XOR value in the
resident vLLM layout.

This is not safe for loaders that cast, quantize, perform floating-point
arithmetic, or otherwise transform values. Support for a real model must be
established by a byte-exact routing test before enabling it here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


class BF16DeltaError(RuntimeError):
    """Raised when a layer cannot safely participate in BF16 delta routing."""


@dataclass(frozen=True)
class DestinationDelta:
    """A destination-layout XOR tensor and the live parameter it updates."""

    names: tuple[str, ...]
    parameter: nn.Parameter
    delta: torch.Tensor
    data_ptr: int


@dataclass(frozen=True)
class _ParameterSlot:
    owner: nn.Module | None
    local_name: str
    qualified_name: str
    parameter: nn.Parameter


def xor_bf16(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return the exact bitwise XOR of two BF16 tensors as a BF16 tensor."""
    _validate_bf16_pair(left, right, context="source XOR")
    return torch.bitwise_xor(left.view(torch.int16), right.view(torch.int16)).view(torch.bfloat16)


@torch.no_grad()
def route_bf16_values_to_scratch(
    layer: nn.Module,
    load_values: Callable[[], Any],
) -> list[DestinationDelta]:
    """Route source-layout BF16 values into zeroed destination-layout scratch.

    ``load_values`` must invoke the owning model's normal name-routing and
    parameter weight-loader logic using values for only ``layer``.  While the
    callback runs, every parameter below ``layer`` keeps its exact object and
    specialized vLLM Parameter subclass, but its ``.data`` points at zeroed
    scratch storage.  Keeping the object is important because vLLM loaders can
    call subclass methods such as ``load_qkv_weight``.

    Live storage is restored even if the callback raises.  The returned scratch
    tensors remain valid after restoration and can be passed to
    :func:`apply_bf16_deltas_`.

    This function deliberately supports only contiguous BF16 parameters.  It
    also rejects distinct parameters sharing storage; exact aliases of the same
    ``Parameter`` object are preserved and represented by one destination.
    """
    return _route_parameter_slots(
        _collect_parameter_slots(layer),
        load_values,
        context=type(layer).__name__,
    )


@torch.no_grad()
def route_bf16_values_to_named_parameters(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    load_values: Callable[[], Any],
    *,
    context: str = "parameter selection",
) -> list[DestinationDelta]:
    """Route values into scratch for an explicit bounded parameter selection."""
    slots = [
        _ParameterSlot(owner=None, local_name=name.rsplit(".", 1)[-1], qualified_name=name, parameter=parameter)
        for name, parameter in named_parameters
    ]
    return _route_parameter_slots(slots, load_values, context=context)


def _route_parameter_slots(
    slots: list[_ParameterSlot],
    load_values: Callable[[], Any],
    *,
    context: str,
) -> list[DestinationDelta]:
    if not slots:
        raise BF16DeltaError(f"{context} has no parameters to route")

    scratch_by_parameter: dict[int, torch.Tensor] = {}
    original_data_by_parameter: dict[int, torch.Tensor] = {}
    names_by_parameter: dict[int, list[str]] = {}
    live_by_parameter: dict[int, nn.Parameter] = {}
    pointers_by_parameter: dict[int, int] = {}
    storage_owners: dict[tuple[str, int | None, int], nn.Parameter] = {}

    for slot in slots:
        parameter = slot.parameter
        parameter_id = id(parameter)
        _validate_destination_parameter(slot.qualified_name, parameter)
        _reject_distinct_shared_storage(slot.qualified_name, parameter, storage_owners)

        names_by_parameter.setdefault(parameter_id, []).append(slot.qualified_name)
        live_by_parameter[parameter_id] = parameter
        pointers_by_parameter[parameter_id] = parameter.data_ptr()
        if parameter_id not in scratch_by_parameter:
            original_data_by_parameter[parameter_id] = parameter.data
            scratch_by_parameter[parameter_id] = torch.zeros_like(parameter, memory_format=torch.preserve_format)

    try:
        for parameter_id, parameter in live_by_parameter.items():
            parameter.data = scratch_by_parameter[parameter_id]
        load_values()
        _validate_scratch_storage(live_by_parameter, scratch_by_parameter, names_by_parameter)
    finally:
        for parameter_id, parameter in live_by_parameter.items():
            parameter.data = original_data_by_parameter[parameter_id]

    deltas: list[DestinationDelta] = []
    for parameter_id, parameter in live_by_parameter.items():
        expected_ptr = pointers_by_parameter[parameter_id]
        if parameter.data_ptr() != expected_ptr:
            raise BF16DeltaError(
                f"live storage pointer changed for {names_by_parameter[parameter_id]}: "
                f"expected {expected_ptr}, got {parameter.data_ptr()}"
            )
        deltas.append(
            DestinationDelta(
                names=tuple(names_by_parameter[parameter_id]),
                parameter=parameter,
                delta=scratch_by_parameter[parameter_id],
                data_ptr=expected_ptr,
            )
        )
    return deltas


@torch.no_grad()
def apply_bf16_deltas_(deltas: Iterable[DestinationDelta]) -> None:
    """XOR destination-layout deltas into live BF16 parameter storage in place."""
    materialized = list(deltas)
    seen_parameters: set[int] = set()

    # Validate the complete bounded update before mutating any live bytes.
    for item in materialized:
        parameter_id = id(item.parameter)
        if parameter_id in seen_parameters:
            raise BF16DeltaError(f"duplicate destination parameter in delta update: {item.names}")
        seen_parameters.add(parameter_id)

        if item.parameter.data_ptr() != item.data_ptr:
            raise BF16DeltaError(
                f"live storage pointer changed before applying {item.names}: "
                f"expected {item.data_ptr}, got {item.parameter.data_ptr()}"
            )
        _validate_bf16_pair(item.parameter, item.delta, context=f"destination {item.names}")

    for item in materialized:
        item.parameter.view(torch.int16).bitwise_xor_(item.delta.view(torch.int16))


def _collect_parameter_slots(root: nn.Module) -> list[_ParameterSlot]:
    slots: list[_ParameterSlot] = []

    def visit(module: nn.Module, prefix: str) -> None:
        for local_name, parameter in module._parameters.items():
            if parameter is None:
                continue
            qualified_name = f"{prefix}.{local_name}" if prefix else local_name
            slots.append(
                _ParameterSlot(
                    owner=module,
                    local_name=local_name,
                    qualified_name=qualified_name,
                    parameter=parameter,
                )
            )
        for child_name, child in module._modules.items():
            if child is None:
                continue
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            visit(child, child_prefix)

    visit(root, "")
    return slots


def _validate_destination_parameter(name: str, parameter: nn.Parameter) -> None:
    if parameter.dtype != torch.bfloat16:
        raise BF16DeltaError(f"{name} has unsupported dtype {parameter.dtype}; only torch.bfloat16 is supported")
    if parameter.device.type == "meta":
        raise BF16DeltaError(f"{name} is on the meta device")
    if not parameter.is_contiguous():
        raise BF16DeltaError(f"{name} is non-contiguous with stride {parameter.stride()}")


def _validate_bf16_pair(left: torch.Tensor, right: torch.Tensor, *, context: str) -> None:
    if left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
        raise BF16DeltaError(f"{context} requires torch.bfloat16 tensors, got left={left.dtype}, right={right.dtype}")
    if left.shape != right.shape:
        raise BF16DeltaError(f"{context} shape mismatch: left={tuple(left.shape)}, right={tuple(right.shape)}")
    if left.device != right.device:
        raise BF16DeltaError(f"{context} device mismatch: left={left.device}, right={right.device}")
    if not left.is_contiguous() or not right.is_contiguous():
        raise BF16DeltaError(f"{context} requires contiguous tensors")


def _reject_distinct_shared_storage(
    name: str,
    parameter: nn.Parameter,
    storage_owners: dict[tuple[str, int | None, int], nn.Parameter],
) -> None:
    storage = parameter.untyped_storage()
    storage_key = (parameter.device.type, parameter.device.index, storage.data_ptr())
    previous = storage_owners.setdefault(storage_key, parameter)
    if previous is not parameter:
        raise BF16DeltaError(
            f"{name} shares storage with a distinct Parameter; shared-storage delta routing is not supported"
        )


def _validate_scratch_storage(
    live_by_parameter: dict[int, nn.Parameter],
    scratch_by_parameter: dict[int, torch.Tensor],
    names_by_parameter: dict[int, list[str]],
) -> None:
    for parameter_id, parameter in live_by_parameter.items():
        expected = scratch_by_parameter[parameter_id]
        actual = parameter.data
        if actual.data_ptr() != expected.data_ptr():
            raise BF16DeltaError(
                f"weight loader replaced scratch storage for {names_by_parameter[parameter_id]}; "
                "only in-place loading is supported"
            )
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise BF16DeltaError(
                f"weight loader changed scratch metadata for {names_by_parameter[parameter_id]}: "
                f"expected dtype={expected.dtype}, shape={tuple(expected.shape)}, "
                f"got dtype={actual.dtype}, shape={tuple(actual.shape)}"
            )


__all__ = [
    "BF16DeltaError",
    "DestinationDelta",
    "apply_bf16_deltas_",
    "route_bf16_values_to_named_parameters",
    "route_bf16_values_to_scratch",
    "xor_bf16",
]
