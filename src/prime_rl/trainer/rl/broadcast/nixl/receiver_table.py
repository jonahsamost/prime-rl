"""Inference staging destinations for trainer-initiated NIXL writes."""

from __future__ import annotations

import msgspec

from prime_rl.trainer.rl.broadcast.nixl.agent import MemDesc


class ReceiverAgent(msgspec.Struct, frozen=True):
    name: str
    metadata: bytes
    device_id: int
    rank: int


class ReceiverRoute(msgspec.Struct, frozen=True):
    trainer_agent_name: str
    source: MemDesc
    destination: MemDesc


class ReceiverGroup(msgspec.Struct, frozen=True):
    name: str
    routes: list[ReceiverRoute]


class ReceiverTable(msgspec.Struct, frozen=True):
    agent: ReceiverAgent
    groups: list[ReceiverGroup]

    def encode(self) -> bytes:
        return msgspec.msgpack.encode(self)

    @classmethod
    def decode(cls, data: bytes) -> ReceiverTable:
        return msgspec.msgpack.decode(data, type=cls)
