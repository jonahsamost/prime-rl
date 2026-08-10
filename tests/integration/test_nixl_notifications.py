from __future__ import annotations

import time
from uuid import uuid4

import pytest

from prime_rl.trainer.rl.broadcast.nixl.agent import NixlAgent
from prime_rl.trainer.rl.broadcast.nixl.notifications import group_notification

pytestmark = pytest.mark.gpu


def test_nixl_group_notifications_round_trip_over_ucx() -> None:
    suffix = uuid4().hex
    trainer = NixlAgent(f"notification-test-trainer-{suffix}")
    inference = NixlAgent(f"notification-test-inference-{suffix}")
    trainer_peer = inference.add_remote_agent(trainer.get_metadata())
    inference_peer = trainer.add_remote_agent(inference.get_metadata())
    inference.make_connection(trainer_peer)
    trainer.make_connection(inference_peer)

    ready = group_notification(
        session_id=suffix,
        kind="ready",
        step=3,
        group_index=9,
        buffer_index=1,
        inference_rank=0,
    )
    trainer.send_notification(inference_peer, ready.encode())
    assert wait_for_message(inference, trainer.name, ready.encode())

    acknowledgement = group_notification(
        session_id=suffix,
        kind="ack",
        step=3,
        group_index=9,
        buffer_index=1,
        inference_rank=0,
    )
    inference.send_notification(trainer_peer, acknowledgement.encode())
    assert wait_for_message(trainer, inference.name, acknowledgement.encode())


def wait_for_message(agent: NixlAgent, sender: str, message: bytes, timeout: float = 10) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if message in agent.get_notifications().get(sender, []):
            return True
        time.sleep(0.001)
    return False
