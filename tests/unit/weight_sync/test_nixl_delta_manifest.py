from __future__ import annotations

import torch

from prime_rl.trainer.rl.broadcast.nixl.delta_manifest import (
    NIXL_DELTA_PROTOCOL_VERSION,
    NIXLDeltaAgent,
    NIXLDeltaGroup,
    NIXLDeltaManifest,
    build_local_delta_groups,
    merge_delta_manifest_fragments,
)
from prime_rl.trainer.rl.broadcast.nixl.graph import TensorOperation, chain_preserves_dtype
from prime_rl.weight_sync.xor_delta import CompressedDeltaFrame, DeltaTensorMetadata, DeltaUpdate


def _local_fragment(agent_name: str, address_offset: int = 0) -> NIXLDeltaManifest:
    payload = torch.zeros(256, dtype=torch.uint8)
    update = DeltaUpdate(
        base_step=4,
        step=5,
        tensors=(
            DeltaTensorMetadata(
                name="model.layers.0.mlp.weight",
                shape=(2, 2),
                dtype="bfloat16",
                nbytes=8,
            ),
        ),
        frames=(
            CompressedDeltaFrame(
                first_tensor_index=0,
                tensor_count=1,
                uncompressed_nbytes=8,
                compressed_nbytes=16,
            ),
        ),
        payload=payload,
    )
    groups = build_local_delta_groups(update, ["non_layer", "layer.0"])
    frame = groups[1][0]
    groups[1][0] = type(frame)(
        agent=frame.agent,
        addr=4096 + address_offset,
        compressed_nbytes=frame.compressed_nbytes,
        uncompressed_nbytes=frame.uncompressed_nbytes,
        tensors=frame.tensors,
    )
    return NIXLDeltaManifest(
        protocol_version=NIXL_DELTA_PROTOCOL_VERSION,
        base_step=4,
        step=5,
        agents=(NIXLDeltaAgent(agent_name, b"metadata", 0),),
        groups=tuple(
            NIXLDeltaGroup(name=name, frames=tuple(frames))
            for name, frames in zip(("non_layer", "layer.0"), groups, strict=True)
        ),
    )


def test_nixl_delta_manifest_round_trip():
    fragment = _local_fragment("trainer-0")

    assert NIXLDeltaManifest.decode(fragment.encode()) == fragment


def test_nixl_delta_manifest_merge_reindexes_agents():
    merged = merge_delta_manifest_fragments([_local_fragment("trainer-0"), _local_fragment("trainer-1", 256)])

    assert [agent.name for agent in merged.agents] == ["trainer-0", "trainer-1"]
    assert [frame.agent for frame in merged.groups[1].frames] == [0, 1]


def test_nixl_delta_graph_rejects_intermediate_dtype_conversion():
    assert chain_preserves_dtype(
        (4, 4),
        torch.float8_e4m3fn,
        (TensorOperation("transpose", (0, 1)), TensorOperation("contiguous")),
    )
    assert chain_preserves_dtype(
        (4, 4),
        torch.bfloat16,
        (TensorOperation("transpose", (0, 1)), TensorOperation("contiguous")),
    )
    assert not chain_preserves_dtype(
        (4, 4),
        torch.bfloat16,
        (TensorOperation("float"), TensorOperation("bfloat16")),
    )
