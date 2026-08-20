from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from prime_rl.configs.rl import RLConfig
from prime_rl.inference.vllm.worker import nixl_push as worker_push_module
from prime_rl.inference.vllm.worker.nixl_push import NIXLPushWeightUpdateWorker
from prime_rl.trainer.rl.broadcast.nixl import push as push_module
from prime_rl.trainer.rl.broadcast.nixl.agent import NixlAgent, group_notification
from prime_rl.trainer.rl.broadcast.nixl.push import NIXLPushWeightBroadcast
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
        self.notification = b""
        self.notifications: dict[str, set[bytes]] = {}
        self.sent_notification: tuple[str, bytes] | None = None

    def make_prepped_xfer(self, *, operation: str, **kwargs):
        del kwargs
        self.operation = operation
        return "handle"

    def transfer(self, handle: str, notif_msg: bytes = b"") -> str:
        assert handle == "handle"
        self.notification = notif_msg
        return self.transfer_state

    def add_remote_agent(self, metadata: bytes) -> bytes:
        assert metadata == b"metadata"
        return b"remote-agent"

    def check_remote_xfer_done(
        self,
        peer_name: str,
        message: bytes,
        *,
        backends: list[str],
        tag_is_prefix: bool,
    ) -> bool:
        assert backends == ["UCX"]
        assert not tag_is_prefix
        if message not in self.notifications.get(peer_name, set()):
            return False
        self.notifications[peer_name].remove(message)
        return True

    def send_notif(self, peer_name: str, message: bytes) -> None:
        self.sent_notification = (peer_name, message)


def test_post_read_and_write_use_requested_nixl_operations() -> None:
    agent = object.__new__(NixlAgent)
    agent._agent = FakeNixlBinding()

    assert agent.post_read("local", [0], "remote") == "handle"
    assert agent._agent.operation == "READ"
    assert agent.post_write("local", [0], "remote") == "handle"
    assert agent._agent.operation == "WRITE"
    notification = group_notification(3, 7)
    assert agent.post_write("local", [0], "remote", notification) == "handle"
    assert agent._agent.notification == notification


def test_add_remote_agent_normalizes_bytes_to_text() -> None:
    agent = object.__new__(NixlAgent)
    agent._agent = FakeNixlBinding()

    assert agent.add_remote_agent(b"metadata") == "remote-agent"


def test_nixl_notifications_preserve_messages_received_early() -> None:
    agent = object.__new__(NixlAgent)
    agent._agent = FakeNixlBinding()
    current = group_notification(3, 7)
    future = group_notification(4, 7)
    agent._agent.notifications = {"peer": {future, current}}

    agent.wait_for_notifications({"peer": current}, timeout=1)
    agent.wait_for_notifications({"peer": future}, timeout=1)
    agent.send_notification("peer", current)

    assert agent._agent.sent_notification == ("peer", current)


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
            "weight_broadcast": {
                "type": "nixl",
                "protocol": "push",
                "push_buffer_count": 4,
            },
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
            "inference": {},
        }
    )

    assert config.trainer.weight_broadcast.protocol == "push"
    assert config.trainer.weight_broadcast.push_buffer_count == 4
    assert config.orchestrator.weight_broadcast.protocol == "push"
    assert config.inference is not None
    assert config.inference.weight_broadcast.protocol == "push"


def test_push_pipeline_waits_for_previous_replay_before_reusing_slot(monkeypatch) -> None:
    broadcast = object.__new__(NIXLPushWeightBroadcast)
    broadcast.transfer_group_names = ["group.0", "group.1", "group.2"]
    broadcast.staging_buffer_count = 2
    broadcast.world = SimpleNamespace(is_master=False)
    broadcast.logger = SimpleNamespace(info=lambda message: None)

    events: list[str] = []
    broadcast.initialize_transfer = lambda model: None
    broadcast.finish_staging_buffer_transfer = lambda group: events.append(f"reuse:{group}")
    broadcast.stage_group = lambda group: events.append(f"stage:{group}")
    broadcast.post_group_writes = lambda group: events.append(f"post:{group}") or []
    broadcast.finish_group_writes = lambda group, pending: events.append(f"finish:{group}")
    monkeypatch.setattr(push_module.dist, "barrier", lambda: None)

    broadcast.broadcast_weights(SimpleNamespace(), step=7)

    assert events[:5] == [
        "stage:0",
        "post:0",
        "stage:1",
        "post:1",
        "finish:0",
    ]
    assert events == [
        "stage:0",
        "post:0",
        "stage:1",
        "post:1",
        "finish:0",
        "reuse:0",
        "stage:2",
        "post:2",
        "finish:1",
        "reuse:1",
        "finish:2",
        "reuse:2",
    ]


def test_push_pipeline_finishes_write_before_reusing_single_slot(monkeypatch) -> None:
    broadcast = object.__new__(NIXLPushWeightBroadcast)
    broadcast.transfer_group_names = ["group.0", "group.1"]
    broadcast.staging_buffer_count = 1
    broadcast.world = SimpleNamespace(is_master=False)
    broadcast.logger = SimpleNamespace(info=lambda message: None)

    events: list[str] = []
    broadcast.initialize_transfer = lambda model: None
    broadcast.finish_staging_buffer_transfer = lambda slot: events.append(f"reuse:{slot}")
    broadcast.stage_group = lambda group: events.append(f"stage:{group}")
    broadcast.post_group_writes = lambda group: events.append(f"post:{group}") or []
    broadcast.finish_group_writes = lambda group, pending: events.append(f"finish:{group}")
    monkeypatch.setattr(push_module.dist, "barrier", lambda: None)

    broadcast.broadcast_weights(SimpleNamespace(), step=7)

    assert events == [
        "stage:0",
        "post:0",
        "finish:0",
        "reuse:0",
        "stage:1",
        "post:1",
        "finish:1",
        "reuse:1",
    ]


def test_push_pipeline_wraps_configured_buffer_ring(monkeypatch) -> None:
    broadcast = object.__new__(NIXLPushWeightBroadcast)
    broadcast.transfer_group_names = [f"group.{index}" for index in range(7)]
    broadcast.staging_buffer_count = 4
    broadcast.world = SimpleNamespace(is_master=False)
    broadcast.logger = SimpleNamespace(info=lambda message: None)

    events: list[str] = []
    broadcast.initialize_transfer = lambda model: None
    broadcast.finish_staging_buffer_transfer = lambda group: events.append(f"reuse:{group}")
    broadcast.stage_group = lambda group: events.append(f"stage:{group}")
    broadcast.post_group_writes = lambda group: []
    broadcast.finish_group_writes = lambda group, pending: None
    monkeypatch.setattr(push_module.dist, "barrier", lambda: None)

    broadcast.broadcast_weights(SimpleNamespace(), step=7)

    assert events == [
        "stage:0",
        "stage:1",
        "stage:2",
        "stage:3",
        "reuse:0",
        "stage:4",
        "reuse:1",
        "stage:5",
        "reuse:2",
        "stage:6",
        "reuse:3",
        "reuse:4",
        "reuse:5",
        "reuse:6",
    ]


def test_push_receiver_requires_matching_buffer_ring(monkeypatch) -> None:
    worker = object.__new__(NIXLPushWeightUpdateWorker)
    worker.device = torch.device("cuda", 0)
    elements = {torch.bfloat16: 1024}

    assert worker.choose_receive_buffer_count({}, staging_buffer_count=4) == 4

    monkeypatch.setattr(worker_push_module, "size_cuda_buffers", lambda *args, **kwargs: 1)
    with pytest.raises(RuntimeError, match="matching trainer and inference staging counts"):
        worker.choose_receive_buffer_count(elements, staging_buffer_count=2)

    monkeypatch.setattr(worker_push_module, "size_cuda_buffers", lambda *args, **kwargs: 4)
    assert worker.choose_receive_buffer_count(elements, staging_buffer_count=4) == 4
    assert not worker.ack_before_replay
