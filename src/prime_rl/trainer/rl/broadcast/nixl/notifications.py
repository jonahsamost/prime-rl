"""Generation-aware control messages for staged NIXL weight groups."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

import msgspec

NIXL_GROUP_NOTIFICATION_VERSION = 1
GroupNotificationKind = Literal["ready", "ack"]
type NotificationKey = tuple[str, str, int, int, int, int]


class NotificationAgent(Protocol):
    def get_notifications(self) -> dict[str, list[bytes]]: ...


class NIXLGroupNotification(msgspec.Struct, frozen=True):
    protocol_version: int
    session_id: str
    kind: GroupNotificationKind
    step: int
    group_index: int
    buffer_index: int
    inference_rank: int

    def encode(self) -> bytes:
        validate_group_notification(self)
        return msgspec.msgpack.encode(self)

    @classmethod
    def decode(cls, data: bytes) -> NIXLGroupNotification:
        notification = msgspec.msgpack.decode(data, type=cls)
        validate_group_notification(notification)
        return notification


def validate_group_notification(notification: NIXLGroupNotification) -> None:
    if notification.protocol_version != NIXL_GROUP_NOTIFICATION_VERSION:
        raise ValueError(
            f"unsupported NIXL group notification version {notification.protocol_version}"
        )
    if not notification.session_id:
        raise ValueError("NIXL group notification requires a session ID")
    if notification.kind not in ("ready", "ack"):
        raise ValueError(f"unsupported NIXL group notification kind {notification.kind!r}")
    if min(
        notification.step,
        notification.group_index,
        notification.buffer_index,
        notification.inference_rank,
    ) < 0:
        raise ValueError("NIXL group notification indices must be non-negative")


@dataclass
class NIXLNotificationInbox:
    _messages: set[tuple[str, NotificationKey]] = field(default_factory=set)

    def add(self, notifications: Mapping[str, list[bytes]]) -> None:
        for sender, messages in notifications.items():
            self._messages.update(
                (sender, notification_key(NIXLGroupNotification.decode(message)))
                for message in messages
            )

    def discard_before(self, step: int, group_index: int) -> None:
        generation = (step, group_index)
        self._messages = {
            item
            for item in self._messages
            if (item[1][2], item[1][3]) >= generation
        }

    def consume(self, expected: Mapping[str, NIXLGroupNotification]) -> bool:
        messages = {
            (sender, notification_key(notification))
            for sender, notification in expected.items()
        }
        if not messages.issubset(self._messages):
            return False
        self._messages.difference_update(messages)
        return True

    def contains(self, sender: str, notification: NIXLGroupNotification) -> bool:
        return (sender, notification_key(notification)) in self._messages


def notification_key(notification: NIXLGroupNotification) -> NotificationKey:
    return (
        notification.session_id,
        notification.kind,
        notification.step,
        notification.group_index,
        notification.buffer_index,
        notification.inference_rank,
    )


def wait_for_notifications(
    agent: NotificationAgent,
    inbox: NIXLNotificationInbox,
    expected: Mapping[str, NIXLGroupNotification],
    *,
    timeout: float,
    context: str,
    cancelled: Callable[[], bool] | None = None,
    poll_interval: float = 0.0005,
) -> None:
    if not expected:
        raise ValueError(f"cannot wait for an empty NIXL notification set, context={context!r}")
    first = next(iter(expected.values()))
    inbox.discard_before(first.step, first.group_index)
    deadline = time.monotonic() + timeout
    while True:
        if cancelled is not None and cancelled():
            raise RuntimeError(f"NIXL notification wait cancelled, context={context!r}")
        inbox.add(agent.get_notifications())
        if inbox.consume(expected):
            return
        if time.monotonic() >= deadline:
            missing = [
                sender
                for sender, notification in expected.items()
                if not inbox.contains(sender, notification)
            ]
            raise TimeoutError(
                f"NIXL notification wait timed out after {timeout}s, context={context!r}, "
                f"missing={missing}"
            )
        time.sleep(poll_interval)


def group_notification(
    *,
    session_id: str,
    kind: GroupNotificationKind,
    step: int,
    group_index: int,
    buffer_index: int,
    inference_rank: int,
) -> NIXLGroupNotification:
    return NIXLGroupNotification(
        protocol_version=NIXL_GROUP_NOTIFICATION_VERSION,
        session_id=session_id,
        kind=kind,
        step=step,
        group_index=group_index,
        buffer_index=buffer_index,
        inference_rank=inference_rank,
    )


__all__ = [
    "NIXL_GROUP_NOTIFICATION_VERSION",
    "NIXLGroupNotification",
    "NIXLNotificationInbox",
    "group_notification",
    "validate_group_notification",
    "wait_for_notifications",
]
