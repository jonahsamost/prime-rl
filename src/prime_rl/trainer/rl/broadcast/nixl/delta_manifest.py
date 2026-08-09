"""Serializable metadata for direct NIXL XOR-delta pulls."""

from __future__ import annotations

from math import prod

import msgspec

from prime_rl.weight_sync.grouping import weight_transfer_group_name
from prime_rl.weight_sync.xor_delta import (
    NVCOMP_FRAME_ALIGNMENT,
    DeltaTensorMetadata,
    DeltaUpdate,
    delta_dtype_from_name,
    delta_dtype_nbytes,
)

NIXL_DELTA_PROTOCOL_VERSION = 1


class NIXLDeltaAgent(msgspec.Struct, frozen=True):
    name: str
    metadata: bytes
    device_id: int


class NIXLDeltaTensor(msgspec.Struct, frozen=True):
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    global_shape: tuple[int, ...]
    shard_dim: int | None
    shard_index: int
    shard_count: int

    @classmethod
    def from_metadata(cls, value: DeltaTensorMetadata) -> NIXLDeltaTensor:
        return cls(
            name=value.name,
            shape=value.shape,
            dtype=value.dtype,
            nbytes=value.nbytes,
            global_shape=value.resolved_global_shape,
            shard_dim=value.shard_dim,
            shard_index=value.shard_index,
            shard_count=value.shard_count,
        )

    def to_metadata(self) -> DeltaTensorMetadata:
        return DeltaTensorMetadata(
            name=self.name,
            shape=self.shape,
            dtype=self.dtype,
            nbytes=self.nbytes,
            global_shape=self.global_shape,
            shard_dim=self.shard_dim,
            shard_index=self.shard_index,
            shard_count=self.shard_count,
        )


class NIXLDeltaFrame(msgspec.Struct, frozen=True):
    agent: int
    addr: int
    compressed_nbytes: int
    uncompressed_nbytes: int
    tensors: tuple[NIXLDeltaTensor, ...]


class NIXLDeltaGroup(msgspec.Struct, frozen=True):
    name: str
    frames: tuple[NIXLDeltaFrame, ...]


class NIXLDeltaManifest(msgspec.Struct, frozen=True):
    protocol_version: int
    base_step: int
    step: int
    agents: tuple[NIXLDeltaAgent, ...]
    groups: tuple[NIXLDeltaGroup, ...]

    def encode(self) -> bytes:
        validate_nixl_delta_manifest(self)
        return msgspec.msgpack.encode(self)

    @classmethod
    def decode(cls, data: bytes) -> NIXLDeltaManifest:
        manifest = msgspec.msgpack.decode(data, type=cls)
        validate_nixl_delta_manifest(manifest)
        return manifest


class NIXLPolicyMetadata(msgspec.Struct, frozen=True):
    kind: str
    payload: bytes
    base_step: int = -1
    step: int = 0

    def encode(self) -> bytes:
        if self.kind not in ("full", "xor"):
            raise ValueError(f"unsupported NIXL policy metadata kind {self.kind!r}")
        if self.kind == "full" and self.base_step != -1:
            raise ValueError("full NIXL policy metadata must use base_step=-1")
        if self.kind == "xor" and self.step != self.base_step + 1:
            raise ValueError("XOR NIXL policy metadata must name consecutive versions")
        return msgspec.msgpack.encode(self)

    @classmethod
    def decode(cls, data: bytes) -> NIXLPolicyMetadata:
        value = msgspec.msgpack.decode(data, type=cls)
        if value.kind not in ("full", "xor"):
            raise ValueError(f"unsupported NIXL policy metadata kind {value.kind!r}")
        value.encode()
        return value


def build_local_delta_groups(update: DeltaUpdate, group_names: list[str]) -> list[list[NIXLDeltaFrame]]:
    groups_by_name = {name: index for index, name in enumerate(group_names)}
    groups: list[list[NIXLDeltaFrame]] = [[] for _ in group_names]
    for frame, payload in zip(update.frames, update.frame_payloads(), strict=True):
        tensors = update.tensors[frame.first_tensor_index : frame.first_tensor_index + frame.tensor_count]
        frame_groups = {weight_transfer_group_name(tensor.name) for tensor in tensors}
        if len(frame_groups) != 1:
            raise ValueError(f"NIXL XOR frame spans transfer groups {sorted(frame_groups)!r}")
        group_name = frame_groups.pop()
        try:
            group_index = groups_by_name[group_name]
        except KeyError as error:
            raise ValueError(f"NIXL XOR tensor group {group_name!r} is absent from the trainer table") from error
        groups[group_index].append(
            NIXLDeltaFrame(
                agent=0,
                addr=payload.data_ptr(),
                compressed_nbytes=frame.compressed_nbytes,
                uncompressed_nbytes=frame.uncompressed_nbytes,
                tensors=tuple(NIXLDeltaTensor.from_metadata(tensor) for tensor in tensors),
            )
        )
    return groups


def merge_delta_manifest_fragments(fragments: list[NIXLDeltaManifest]) -> NIXLDeltaManifest:
    if not fragments:
        raise ValueError("cannot merge an empty set of NIXL delta manifest fragments")
    reference = fragments[0]
    agents: list[NIXLDeltaAgent] = []
    groups: list[list[NIXLDeltaFrame]] = [[] for _ in reference.groups]
    for fragment in fragments:
        if fragment.base_step != reference.base_step or fragment.step != reference.step:
            raise ValueError("NIXL delta fragments name different policy transitions")
        if tuple(group.name for group in fragment.groups) != tuple(group.name for group in reference.groups):
            raise ValueError("NIXL delta fragments have different transfer groups")
        if len(fragment.agents) != 1:
            raise ValueError("a local NIXL delta fragment must describe exactly one agent")
        agent_index = len(agents)
        agents.append(fragment.agents[0])
        for group_index, group in enumerate(fragment.groups):
            groups[group_index].extend(
                NIXLDeltaFrame(
                    agent=agent_index,
                    addr=frame.addr,
                    compressed_nbytes=frame.compressed_nbytes,
                    uncompressed_nbytes=frame.uncompressed_nbytes,
                    tensors=frame.tensors,
                )
                for frame in group.frames
            )
    manifest = NIXLDeltaManifest(
        protocol_version=NIXL_DELTA_PROTOCOL_VERSION,
        base_step=reference.base_step,
        step=reference.step,
        agents=tuple(agents),
        groups=tuple(
            NIXLDeltaGroup(name=reference.groups[index].name, frames=tuple(frames))
            for index, frames in enumerate(groups)
        ),
    )
    validate_nixl_delta_manifest(manifest)
    return manifest


def validate_nixl_delta_manifest(manifest: NIXLDeltaManifest) -> None:
    if manifest.protocol_version != NIXL_DELTA_PROTOCOL_VERSION:
        raise ValueError(f"unsupported NIXL delta protocol version {manifest.protocol_version}")
    if manifest.step != manifest.base_step + 1:
        raise ValueError(f"NIXL XOR update must name consecutive versions: {manifest.base_step}->{manifest.step}")
    if not manifest.agents:
        raise ValueError("NIXL delta manifest has no trainer agents")
    agent_names = [agent.name for agent in manifest.agents]
    if any(not agent.name or not agent.metadata or agent.device_id < 0 for agent in manifest.agents):
        raise ValueError("NIXL delta manifest has invalid trainer agent metadata")
    if len(agent_names) != len(set(agent_names)):
        raise ValueError("NIXL delta manifest has duplicate trainer agents")
    if not manifest.groups:
        raise ValueError("NIXL delta manifest has no transfer groups")
    seen_groups: set[str] = set()
    seen_tensors: set[tuple[int, str]] = set()
    for group in manifest.groups:
        if not group.name or group.name in seen_groups:
            raise ValueError(f"invalid or duplicate NIXL delta group {group.name!r}")
        seen_groups.add(group.name)
        ranges_by_agent: dict[int, list[tuple[int, int]]] = {}
        for frame in group.frames:
            if not 0 <= frame.agent < len(manifest.agents):
                raise ValueError(f"NIXL delta frame references invalid agent {frame.agent}")
            if frame.addr <= 0 or frame.compressed_nbytes <= 0 or frame.uncompressed_nbytes <= 0:
                raise ValueError("NIXL delta frame has an invalid address or size")
            if frame.addr % NVCOMP_FRAME_ALIGNMENT:
                raise ValueError(f"NIXL delta frame address {frame.addr:#x} is not aligned")
            if not frame.tensors:
                raise ValueError("NIXL delta frame has no tensors")
            if sum(tensor.nbytes for tensor in frame.tensors) != frame.uncompressed_nbytes:
                raise ValueError("NIXL delta frame tensor bytes do not match its uncompressed size")
            if {weight_transfer_group_name(tensor.name) for tensor in frame.tensors} != {group.name}:
                raise ValueError(f"NIXL delta frame tensors do not belong to group {group.name!r}")
            frame_range = (frame.addr, frame.addr + frame.compressed_nbytes)
            agent_ranges = ranges_by_agent.setdefault(frame.agent, [])
            if any(frame_range[0] < end and start < frame_range[1] for start, end in agent_ranges):
                raise ValueError("NIXL delta frame address ranges overlap within a transfer group")
            agent_ranges.append(frame_range)
            for tensor in frame.tensors:
                key = (frame.agent, tensor.name)
                if not tensor.name or key in seen_tensors:
                    raise ValueError(f"invalid or duplicate NIXL delta tensor {key}")
                seen_tensors.add(key)
                dtype = delta_dtype_from_name(tensor.dtype)
                expected_nbytes = prod(tensor.shape) * delta_dtype_nbytes(dtype)
                if tensor.nbytes != expected_nbytes:
                    raise ValueError(
                        f"NIXL delta tensor {tensor.name!r} has {tensor.nbytes} bytes; "
                        f"its shape requires {expected_nbytes}"
                    )
                if len(tensor.shape) != len(tensor.global_shape):
                    raise ValueError(f"NIXL delta tensor {tensor.name!r} has incompatible local/global ranks")
                if tensor.shard_count <= 0 or not 0 <= tensor.shard_index < tensor.shard_count:
                    raise ValueError(f"NIXL delta tensor {tensor.name!r} has invalid shard metadata")


__all__ = [
    "NIXL_DELTA_PROTOCOL_VERSION",
    "NIXLDeltaAgent",
    "NIXLDeltaFrame",
    "NIXLDeltaGroup",
    "NIXLDeltaManifest",
    "NIXLDeltaTensor",
    "NIXLPolicyMetadata",
    "build_local_delta_groups",
    "merge_delta_manifest_fragments",
    "validate_nixl_delta_manifest",
]
