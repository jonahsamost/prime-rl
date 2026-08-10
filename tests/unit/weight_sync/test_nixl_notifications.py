from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import pytest

from prime_rl.trainer.rl.broadcast.nixl.notifications import (
    NIXLGroupNotification,
    NIXLNotificationInbox,
    group_notification,
    wait_for_notifications,
)


class FakeNotificationAgent:
    def __init__(self, batches: list[dict[str, list[bytes]]]) -> None:
        self.batches = batches

    def get_notifications(self) -> dict[str, list[bytes]]:
        return self.batches.pop(0) if self.batches else {}


def make_notification(
    *,
    kind: Literal["ready", "ack"] = "ack",
    step: int = 4,
    group_index: int = 7,
    buffer_index: int = 1,
    inference_rank: int = 0,
) -> NIXLGroupNotification:
    return group_notification(
        session_id="test-session",
        kind=kind,
        step=step,
        group_index=group_index,
        buffer_index=buffer_index,
        inference_rank=inference_rank,
    )


def encode_by_sender(values: Mapping[str, NIXLGroupNotification]) -> dict[str, list[bytes]]:
    return {sender: [notification.encode()] for sender, notification in values.items()}


def test_group_notification_round_trip() -> None:
    notification = make_notification()

    assert NIXLGroupNotification.decode(notification.encode()) == notification


def test_notification_inbox_deduplicates_and_preserves_future_generations() -> None:
    current = make_notification()
    future = make_notification(group_index=8, buffer_index=0)
    inbox = NIXLNotificationInbox()
    inbox.add({"inference-0": [current.encode(), current.encode(), future.encode()]})

    assert inbox.consume({"inference-0": current})
    assert not inbox.consume({"inference-0": current})
    assert inbox.consume({"inference-0": future})


def test_wait_for_notifications_requires_every_expected_sender() -> None:
    expected = {
        "inference-0": make_notification(inference_rank=0),
        "inference-1": make_notification(inference_rank=1),
    }
    agent = FakeNotificationAgent(
        [
            encode_by_sender({"inference-1": expected["inference-1"]}),
            encode_by_sender({"inference-0": expected["inference-0"]}),
        ]
    )

    wait_for_notifications(
        agent,
        NIXLNotificationInbox(),
        expected,
        timeout=1,
        context="test",
        poll_interval=0,
    )


def test_wait_for_notifications_discards_stale_generation() -> None:
    stale = make_notification(step=3, group_index=20)
    current = make_notification()
    inbox = NIXLNotificationInbox()
    inbox.add({"inference-0": [stale.encode()]})

    wait_for_notifications(
        FakeNotificationAgent([{"inference-0": [current.encode()]}]),
        inbox,
        {"inference-0": current},
        timeout=1,
        context="test",
        poll_interval=0,
    )

    assert not inbox.consume({"inference-0": stale})


def test_wait_for_notifications_honors_cancellation() -> None:
    with pytest.raises(RuntimeError, match="cancelled"):
        wait_for_notifications(
            FakeNotificationAgent([]),
            NIXLNotificationInbox(),
            {"inference-0": make_notification()},
            timeout=1,
            context="test",
            cancelled=lambda: True,
            poll_interval=0,
        )
