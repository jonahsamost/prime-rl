from __future__ import annotations

from prime_rl.configs.rl import RLConfig
from prime_rl.trainer.rl.broadcast.nixl.agent import NixlAgent
from prime_rl.trainer.rl.broadcast.nixl.receiver_table import (
    ReceiverAgent,
    ReceiverGroup,
    ReceiverRoute,
    ReceiverTable,
)


class FakeNixlBinding:
    def __init__(self, transfer_state: str = "DONE") -> None:
        self.transfer_state = transfer_state
        self.operation: str | None = None

    def make_prepped_xfer(self, *, operation: str, **kwargs):
        del kwargs
        self.operation = operation
        return "handle"

    def transfer(self, handle: str) -> str:
        assert handle == "handle"
        return self.transfer_state

    def add_remote_agent(self, metadata: bytes) -> bytes:
        assert metadata == b"metadata"
        return b"remote-agent"


def test_post_read_and_write_use_requested_nixl_operations() -> None:
    agent = object.__new__(NixlAgent)
    agent._agent = FakeNixlBinding()

    assert agent.post_read("local", [0], "remote") == "handle"
    assert agent._agent.operation == "READ"
    assert agent.post_write("local", [0], "remote") == "handle"
    assert agent._agent.operation == "WRITE"


def test_add_remote_agent_normalizes_bytes_to_text() -> None:
    agent = object.__new__(NixlAgent)
    agent._agent = FakeNixlBinding()

    assert agent.add_remote_agent(b"metadata") == "remote-agent"


def test_receiver_table_round_trip() -> None:
    table = ReceiverTable(
        agent=ReceiverAgent(
            name="inference-r0",
            metadata=b"metadata",
            device_id=1,
            rank=3,
        ),
        groups=[
            ReceiverGroup(
                name="layer.0",
                routes=[
                    ReceiverRoute(
                        trainer_agent_name="trainer-r0",
                        source=(100, 32, 0),
                        destination=(200, 32, 1),
                    )
                ],
            )
        ],
    )

    assert ReceiverTable.decode(table.encode()) == table


def test_push_protocol_propagates_to_all_rl_components() -> None:
    config = RLConfig.model_validate(
        {
            "weight_broadcast": {"type": "nixl", "protocol": "push"},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
            "inference": {},
        }
    )

    assert config.trainer.weight_broadcast.protocol == "push"
    assert config.orchestrator.weight_broadcast.protocol == "push"
    assert config.inference is not None
    assert config.inference.weight_broadcast.protocol == "push"
