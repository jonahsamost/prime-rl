from __future__ import annotations

import multiprocessing as mp
from typing import Any, TypeAlias
from uuid import uuid4

import pytest
import torch

from prime_rl.trainer.rl.broadcast.nixl.agent import NixlAgent, set_ucx_env_defaults

pytestmark = pytest.mark.gpu

ReceiverInfo: TypeAlias = tuple[bytes, int, int, int]


def run_receiver(ready: Any, control: Any, result: Any, suffix: str) -> None:
    torch.cuda.set_device(0)
    set_ucx_env_defaults()
    receiver = NixlAgent(f"write-test-receiver-{suffix}")
    destination = torch.zeros(4096, dtype=torch.int32, device="cuda")
    receiver.register_tensor(destination)
    ready.put(
        (
            receiver.get_metadata(),
            destination.data_ptr(),
            destination.nbytes,
            torch.cuda.current_device(),
        )
    )

    control.get(timeout=30)
    torch.cuda.synchronize()
    expected = torch.arange(4096, dtype=torch.int32, device="cuda")
    result.put(torch.equal(destination, expected))


def run_sender(receiver_info: ReceiverInfo, suffix: str) -> None:
    torch.cuda.set_device(0)
    set_ucx_env_defaults()
    sender = NixlAgent(f"write-test-sender-{suffix}")
    source = torch.arange(4096, dtype=torch.int32, device="cuda")
    sender.register_tensor(source)

    metadata, destination_addr, destination_nbytes, destination_device = receiver_info
    receiver_peer = sender.add_remote_agent(metadata)
    sender.make_connection(receiver_peer)
    local = sender.prepare_xfer_dlist([(source.data_ptr(), source.nbytes, torch.cuda.current_device())])
    remote = sender.prepare_xfer_dlist(
        [(destination_addr, destination_nbytes, destination_device)],
        agent_name=receiver_peer,
    )
    handle = sender.post_write(local, [0], remote)
    sender.wait(handle, context="NIXL WRITE integration test", timeout=10)


def test_nixl_write_updates_registered_cuda_memory() -> None:
    context = mp.get_context("spawn")
    ready = context.Queue()
    control = context.Queue()
    result = context.Queue()
    suffix = uuid4().hex

    receiver = context.Process(target=run_receiver, args=(ready, control, result, suffix))
    receiver.start()
    receiver_info = ready.get(timeout=30)

    sender = context.Process(target=run_sender, args=(receiver_info, suffix))
    sender.start()
    sender.join(timeout=30)
    control.put("verify")
    transferred = result.get(timeout=30)
    receiver.join(timeout=30)

    assert sender.exitcode == 0
    assert receiver.exitcode == 0
    assert transferred
