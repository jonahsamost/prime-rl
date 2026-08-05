"""Experimental XOR-delta routing for vLLM weight loaders.

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

from prime_rl.weight_sync.xor_delta import SUPPORTED_DELTA_DTYPES, integer_view


class DeltaError(RuntimeError):
    """Raised when a layer cannot safely participate in XOR delta routing."""


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


@dataclass(frozen=True)
class _ParameterRoutingEntry:
    names: tuple[str, ...]
    parameter: nn.Parameter
    data_ptr: int


@dataclass(frozen=True)
class ParameterRoutingPlan:
    """Validated, stable destination parameters for repeated loader routing."""

    entries: tuple[_ParameterRoutingEntry, ...]


def xor_bits(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return the exact bitwise XOR of two supported floating-point tensors."""
    _validate_pair(left, right, context="source XOR")
    return torch.bitwise_xor(integer_view(left), integer_view(right)).view(left.dtype)


@torch.no_grad()
def route_values_to_scratch(
    layer: nn.Module,
    load_values: Callable[[], Any],
) -> list[DestinationDelta]:
    """Route source-layout XOR values into zeroed destination-layout scratch.

    ``load_values`` must invoke the owning model's normal name-routing and
    parameter weight-loader logic using values for only ``layer``.  While the
    callback runs, every parameter below ``layer`` keeps its exact object and
    specialized vLLM Parameter subclass, but its ``.data`` points at zeroed
    scratch storage.  Keeping the object is important because vLLM loaders can
    call subclass methods such as ``load_qkv_weight``.

    Live storage is restored even if the callback raises.  The returned scratch
    tensors remain valid after restoration and can be passed to
    :func:`apply_deltas_`.

    This function deliberately supports only contiguous parameters with an
    explicitly supported dtype. It
    also rejects distinct parameters sharing storage; exact aliases of the same
    ``Parameter`` object are preserved and represented by one destination.
    """
    return route_values_with_plan(parameter_routing_plan(layer), load_values)


@torch.no_grad()
def route_values_to_named_parameters(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    load_values: Callable[[], Any],
    *,
    context: str = "parameter selection",
) -> list[DestinationDelta]:
    """Route values into scratch for an explicit bounded parameter selection."""
    return route_values_with_plan(named_parameter_routing_plan(named_parameters, context=context), load_values)


def parameter_routing_plan(root: nn.Module) -> ParameterRoutingPlan:
    return _build_parameter_routing_plan(_collect_parameter_slots(root), context=type(root).__name__)


def named_parameter_routing_plan(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    context: str = "parameter selection",
) -> ParameterRoutingPlan:
    slots = [
        _ParameterSlot(owner=None, local_name=name.rsplit(".", 1)[-1], qualified_name=name, parameter=parameter)
        for name, parameter in named_parameters
    ]
    return _build_parameter_routing_plan(slots, context=context)


def _build_parameter_routing_plan(slots: list[_ParameterSlot], *, context: str) -> ParameterRoutingPlan:
    if not slots:
        raise DeltaError(f"{context} has no parameters to route")

    parameters: dict[int, nn.Parameter] = {}
    names_by_parameter: dict[int, list[str]] = {}
    storage_owners: dict[tuple[str, int | None, int], nn.Parameter] = {}
    for slot in slots:
        parameter = slot.parameter
        parameter_id = id(parameter)
        _validate_destination_parameter(slot.qualified_name, parameter)
        _reject_distinct_shared_storage(slot.qualified_name, parameter, storage_owners)
        parameters.setdefault(parameter_id, parameter)
        names_by_parameter.setdefault(parameter_id, []).append(slot.qualified_name)

    return ParameterRoutingPlan(
        entries=tuple(
            _ParameterRoutingEntry(
                names=tuple(names_by_parameter[parameter_id]),
                parameter=parameter,
                data_ptr=parameter.data_ptr(),
            )
            for parameter_id, parameter in parameters.items()
        ),
    )


def route_values_with_plan(
    plan: ParameterRoutingPlan,
    load_values: Callable[[], Any],
) -> list[DestinationDelta]:
    original_data: list[torch.Tensor] = []
    scratch: list[torch.Tensor] = []
    for entry in plan.entries:
        if entry.parameter.data_ptr() != entry.data_ptr:
            raise DeltaError(
                f"live storage pointer changed for {entry.names}: expected {entry.data_ptr}, "
                f"got {entry.parameter.data_ptr()}"
            )
        original_data.append(entry.parameter.data)
        scratch.append(torch.zeros_like(entry.parameter, memory_format=torch.preserve_format))

    try:
        for entry, value in zip(plan.entries, scratch, strict=True):
            entry.parameter.data = value
        load_values()
        for entry, expected in zip(plan.entries, scratch, strict=True):
            _validate_scratch_storage(entry.parameter, expected, entry.names)
    finally:
        for entry, value in zip(plan.entries, original_data, strict=True):
            entry.parameter.data = value

    for entry in plan.entries:
        if entry.parameter.data_ptr() != entry.data_ptr:
            raise DeltaError(
                f"live storage pointer changed for {entry.names}: expected {entry.data_ptr}, "
                f"got {entry.parameter.data_ptr()}"
            )
    return [
        DestinationDelta(
            names=entry.names,
            parameter=entry.parameter,
            delta=value,
            data_ptr=entry.data_ptr,
        )
        for entry, value in zip(plan.entries, scratch, strict=True)
    ]


@torch.no_grad()
def apply_deltas_(deltas: Iterable[DestinationDelta]) -> None:
    """XOR destination-layout deltas into live parameter storage in place."""
    materialized = list(deltas)
    seen_parameters: set[int] = set()

    # Validate the complete bounded update before mutating any live bytes.
    for item in materialized:
        parameter_id = id(item.parameter)
        if parameter_id in seen_parameters:
            raise DeltaError(f"duplicate destination parameter in delta update: {item.names}")
        seen_parameters.add(parameter_id)

        if item.parameter.data_ptr() != item.data_ptr:
            raise DeltaError(
                f"live storage pointer changed before applying {item.names}: "
                f"expected {item.data_ptr}, got {item.parameter.data_ptr()}"
            )
        _validate_pair(item.parameter, item.delta, context=f"destination {item.names}")

    for item in materialized:
        integer_view(item.parameter).bitwise_xor_(integer_view(item.delta))


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
    if parameter.dtype not in SUPPORTED_DELTA_DTYPES:
        raise DeltaError(f"{name} has unsupported dtype {parameter.dtype}; expected one of {SUPPORTED_DELTA_DTYPES}")
    if parameter.device.type == "meta":
        raise DeltaError(f"{name} is on the meta device")
    if not parameter.is_contiguous():
        raise DeltaError(f"{name} is non-contiguous with stride {parameter.stride()}")


def _validate_pair(left: torch.Tensor, right: torch.Tensor, *, context: str) -> None:
    if left.dtype not in SUPPORTED_DELTA_DTYPES or right.dtype != left.dtype:
        raise DeltaError(f"{context} requires matching supported dtypes, got left={left.dtype}, right={right.dtype}")
    if left.shape != right.shape:
        raise DeltaError(f"{context} shape mismatch: left={tuple(left.shape)}, right={tuple(right.shape)}")
    if left.device != right.device:
        raise DeltaError(f"{context} device mismatch: left={left.device}, right={right.device}")
    if not left.is_contiguous() or not right.is_contiguous():
        raise DeltaError(f"{context} requires contiguous tensors")


def _reject_distinct_shared_storage(
    name: str,
    parameter: nn.Parameter,
    storage_owners: dict[tuple[str, int | None, int], nn.Parameter],
) -> None:
    storage = parameter.untyped_storage()
    storage_key = (parameter.device.type, parameter.device.index, storage.data_ptr())
    previous = storage_owners.setdefault(storage_key, parameter)
    if previous is not parameter:
        raise DeltaError(
            f"{name} shares storage with a distinct Parameter; shared-storage delta routing is not supported"
        )


def _validate_scratch_storage(
    parameter: nn.Parameter,
    expected: torch.Tensor,
    names: tuple[str, ...],
) -> None:
    actual = parameter.data
    if actual.data_ptr() != expected.data_ptr():
        raise DeltaError(
            f"weight loader replaced scratch storage for {names}; only in-place loading is supported"
        )
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise DeltaError(
            f"weight loader changed scratch metadata for {names}: "
            f"expected dtype={expected.dtype}, shape={tuple(expected.shape)}, "
            f"got dtype={actual.dtype}, shape={tuple(actual.shape)}"
        )


__all__ = [
    "DeltaError",
    "DestinationDelta",
    "ParameterRoutingPlan",
    "apply_deltas_",
    "named_parameter_routing_plan",
    "parameter_routing_plan",
    "route_values_to_named_parameters",
    "route_values_to_scratch",
    "route_values_with_plan",
    "xor_bits",
]
