"""In-memory ModelExpress metadata service for launcher-managed local runs."""

from __future__ import annotations

import argparse
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
from modelexpress import p2p_pb2, p2p_pb2_grpc


class InMemoryP2pService(p2p_pb2_grpc.P2pServiceServicer):
    """Minimal ModelExpress rendezvous service used by a single RL run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._identities: dict[str, bytes] = {}
        self._workers: dict[tuple[str, str], p2p_pb2.WorkerMetadata] = {}

    @staticmethod
    def _source_id(identity: p2p_pb2.SourceIdentity) -> str:
        encoded = identity.SerializeToString(deterministic=True)
        return hashlib.sha256(encoded).hexdigest()[:16]

    def PublishMetadata(self, request, context):
        source_id = self._source_id(request.identity)
        worker = p2p_pb2.WorkerMetadata()
        worker.CopyFrom(request.worker)
        worker.updated_at = time.time_ns()
        with self._lock:
            self._identities[source_id] = request.identity.SerializeToString(deterministic=True)
            self._workers[(source_id, request.worker_id)] = worker
        return p2p_pb2.PublishMetadataResponse(
            success=True,
            mx_source_id=source_id,
            worker_id=request.worker_id,
        )

    def ListSources(self, request, context):
        identity = request.identity.SerializeToString(deterministic=True) if request.HasField("identity") else None
        status = request.status_filter if request.HasField("status_filter") else None
        with self._lock:
            instances = [
                p2p_pb2.SourceInstanceRef(
                    mx_source_id=source_id,
                    worker_id=worker_id,
                    model_name=request.identity.model_name if identity is not None else "",
                    worker_rank=worker.worker_rank,
                )
                for (source_id, worker_id), worker in self._workers.items()
                if (identity is None or self._identities[source_id] == identity)
                and (status is None or worker.status == status)
            ]
        return p2p_pb2.ListSourcesResponse(instances=instances)

    def GetMetadata(self, request, context):
        with self._lock:
            worker = self._workers.get((request.mx_source_id, request.worker_id))
            if worker is None:
                return p2p_pb2.GetMetadataResponse(found=False)
            response_worker = p2p_pb2.WorkerMetadata()
            response_worker.CopyFrom(worker)
        return p2p_pb2.GetMetadataResponse(
            found=True,
            worker=response_worker,
            mx_source_id=request.mx_source_id,
            worker_id=request.worker_id,
        )

    def UpdateStatus(self, request, context):
        key = (request.mx_source_id, request.worker_id)
        with self._lock:
            worker = self._workers.get(key)
            if worker is None or worker.worker_rank != request.worker_rank:
                return p2p_pb2.UpdateStatusResponse(success=False, message="worker not found")
            worker.status = request.status
            worker.updated_at = time.time_ns()
        return p2p_pb2.UpdateStatusResponse(success=True)


def serve(host: str, port: int) -> None:
    server = grpc.server(
        ThreadPoolExecutor(max_workers=16),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )
    p2p_pb2_grpc.add_P2pServiceServicer_to_server(InMemoryP2pService(), server)
    address = f"{host}:{port}"
    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"failed to bind ModelExpress metadata server to {address}")
    server.start()
    print(f"ModelExpress metadata server listening on {address}", flush=True)
    server.wait_for_termination()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
