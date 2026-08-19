"""Small NIXL adapter used by the trainer and vLLM workers."""

from __future__ import annotations

import os
import socket
import struct
import time
from typing import Any, Callable, Literal, Mapping, Sequence

from torch import Tensor

MemDesc = tuple[int, int, int]
TransferOperation = Literal["READ", "WRITE"]


def group_notification(group_index: int, generation: int) -> bytes:
    return struct.pack(">QQ", group_index, generation)


class NixlAgent:
    def __init__(self, name: str) -> None:
        # The pinned nixl-cu12 wheel currently publishes this module namespace.
        from nixl_cu13._api import nixl_agent, nixl_agent_config  # type: ignore[import-not-found]

        self.name = name
        self._agent = nixl_agent(name, nixl_agent_config(backends=["UCX"]))

    def register_tensor(self, tensor: Tensor) -> None:
        self._agent.register_memory(tensor, backends=["UCX"])

    def get_metadata(self) -> bytes:
        return self._agent.get_agent_metadata()

    def add_remote_agent(self, metadata: bytes) -> str:
        peer_name = self._agent.add_remote_agent(metadata)
        return peer_name.decode("utf-8") if isinstance(peer_name, bytes) else peer_name

    def make_connection(self, peer_name: str) -> None:
        self._agent.make_connection(peer_name)

    def prepare_xfer_dlist(self, descs: Sequence[MemDesc], agent_name: str | None = None) -> Any:
        return self._agent.prep_xfer_dlist(
            agent_name=agent_name or "",
            xfer_list=list(descs),
            mem_type="cuda",
            backends=["UCX"],
        )

    def post_read(
        self,
        local: Any,
        indices: Sequence[int],
        remote: Any,
        notification: bytes = b"",
    ) -> Any:
        return self._post_transfer("READ", local, indices, remote, notification)

    def post_write(
        self,
        local: Any,
        indices: Sequence[int],
        remote: Any,
        notification: bytes = b"",
    ) -> Any:
        return self._post_transfer("WRITE", local, indices, remote, notification)

    def _post_transfer(
        self,
        operation: TransferOperation,
        local: Any,
        indices: Sequence[int],
        remote: Any,
        notification: bytes,
    ) -> Any:
        handle = self._agent.make_prepped_xfer(
            operation=operation,
            local_xfer_side=local,
            local_indices=list(indices),
            remote_xfer_side=remote,
            remote_indices=list(indices),
            backends=["UCX"],
        )
        state = (
            self._agent.transfer(handle, notif_msg=notification)
            if notification
            else self._agent.transfer(handle)
        )
        if state in ("ERR", "ERROR", "FAIL"):
            raise RuntimeError(f"NIXL {operation} post failed with state {state}")
        return handle

    def send_notification(self, peer_name: str, notification: bytes) -> None:
        self._agent.send_notif(peer_name, notification)

    def wait_for_notifications(
        self,
        expected: Mapping[str, bytes],
        *,
        timeout: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        pending = dict(expected)
        deadline = time.monotonic() + timeout
        while pending:
            if cancelled is not None and cancelled():
                raise RuntimeError("NIXL notification wait cancelled")
            for sender, message in list(pending.items()):
                if self._agent.check_remote_xfer_done(
                    sender,
                    message,
                    backends=["UCX"],
                    tag_is_prefix=False,
                ):
                    del pending[sender]
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"NIXL notification wait timed out after {timeout}s, missing={list(pending)}"
                )
            time.sleep(0.0005)

    def wait(
        self,
        handle: Any,
        context: str = "",
        timeout: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if cancelled is not None and cancelled():
                self._agent.release_xfer_handle(handle)
                raise RuntimeError(f"NIXL transfer cancelled, context={context!r}")
            state = self._agent.check_xfer_state(handle)
            if state in ("DONE", "SUCCESS"):
                self._agent.release_xfer_handle(handle)
                return
            if state in ("ERR", "ERROR", "FAIL"):
                self._agent.release_xfer_handle(handle)
                raise RuntimeError(f"NIXL transfer failed with state={state}, context={context!r}")
            if deadline is not None and time.monotonic() >= deadline:
                self._agent.release_xfer_handle(handle)
                raise TimeoutError(f"NIXL transfer timed out after {timeout}s, context={context!r}")
            time.sleep(0.0005)


def make_agent_name(role: str, global_rank: int) -> str:
    return f"{role}-{socket.gethostname()}-r{global_rank}"


def set_ucx_env_defaults() -> None:
    os.environ.setdefault("UCX_TLS", "rc_x,rc,dc_x,dc,cuda_copy")
    os.environ.setdefault("UCX_IB_GPU_DIRECT_RDMA", "y")
    os.environ.setdefault("UCX_RNDV_SCHEME", "get_zcopy")
    os.environ.setdefault("UCX_RNDV_THRESH", "0")
    os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")
    os.environ.setdefault("UCX_WARN_UNUSED_ENV_VARS", "n")
