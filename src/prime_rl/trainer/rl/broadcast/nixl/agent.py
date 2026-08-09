"""Small NIXL adapter used by the trainer and vLLM workers."""

from __future__ import annotations

import os
import socket
import time
from importlib import import_module
from typing import Any, Callable, Sequence

from torch import Tensor, version

MemDesc = tuple[int, int, int]


def _nixl_api_modules() -> tuple[str, ...]:
    cuda_major = version.cuda.split(".", 1)[0] if version.cuda else None
    cuda_modules = ("nixl_cu13._api", "nixl_cu12._api")
    if cuda_major == "12":
        cuda_modules = tuple(reversed(cuda_modules))
    return (*cuda_modules, "nixl._api")


def _create_ucx_agent(name: str) -> Any:
    unavailable: list[str] = []
    for module_name in _nixl_api_modules():
        try:
            api = import_module(module_name)
        except ImportError:
            unavailable.append(f"{module_name} not installed")
            continue
        agent = api.nixl_agent(name, api.nixl_agent_config(backends=["UCX"]))
        if "UCX" in agent.get_plugin_list():
            return agent
        unavailable.append(f"{module_name} has no UCX plugin")
        del agent
    raise RuntimeError(f"NIXL UCX backend is unavailable ({'; '.join(unavailable)})")


class NixlAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        set_ucx_env_defaults()
        self._agent = _create_ucx_agent(name)

    def register_tensor(self, tensor: Tensor) -> Any:
        return self._agent.register_memory(tensor, backends=["UCX"])

    def deregister_tensor(self, registration: Any) -> None:
        self._agent.deregister_memory(registration, backends=["UCX"])

    def get_metadata(self) -> bytes:
        return self._agent.get_agent_metadata()

    def add_remote_agent(self, metadata: bytes) -> str:
        return self._agent.add_remote_agent(metadata)

    def make_connection(self, peer_name: str) -> None:
        self._agent.make_connection(peer_name)

    def prepare_xfer_dlist(self, descs: Sequence[MemDesc], agent_name: str | None = None) -> Any:
        return self._agent.prep_xfer_dlist(
            agent_name=agent_name or "",
            xfer_list=list(descs),
            mem_type="cuda",
            backends=["UCX"],
        )

    def post_read(self, local: Any, indices: Sequence[int], remote: Any) -> Any:
        handle = self._agent.make_prepped_xfer(
            operation="READ",
            local_xfer_side=local,
            local_indices=list(indices),
            remote_xfer_side=remote,
            remote_indices=list(indices),
            backends=["UCX"],
        )
        state = self._agent.transfer(handle)
        if state in ("ERR", "ERROR", "FAIL"):
            raise RuntimeError(f"NIXL READ post failed with state {state}")
        return handle

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
    os.environ.setdefault("UCX_TLS", "all")
    os.environ.setdefault("UCX_IB_GPU_DIRECT_RDMA", "y")
    os.environ.setdefault("UCX_RNDV_SCHEME", "get_zcopy")
    os.environ.setdefault("UCX_RNDV_THRESH", "0")
    os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")
    os.environ.setdefault("UCX_WARN_UNUSED_ENV_VARS", "n")
