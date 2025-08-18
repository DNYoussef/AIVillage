import json
import socket
import threading
import time

import pytest

from .peer_discovery import PeerDiscovery


class DummyNode:
    def __init__(self) -> None:
        self.node_id = "node"
        self.listen_port = 9000
        self.peer_registry = {}
        self.connections = {}
        self.local_capabilities = None
        self.use_tls = False
        self.ssl_context = None

    async def send_to_peer(
        self, peer_id, message
    ) -> None:  # pragma: no cover - placeholder
        pass


def test_add_and_remove_peer() -> None:
    node = DummyNode()
    discovery = PeerDiscovery(node)
    discovery.add_known_peer("1.1.1.1", 9000)
    assert ("1.1.1.1", 9000) in discovery.discovered_peers
    discovery.remove_peer("1.1.1.1", 9000)
    assert ("1.1.1.1", 9000) not in discovery.discovered_peers


def test_discovery_stats_counts() -> None:
    node = DummyNode()
    discovery = PeerDiscovery(node)
    discovery.add_known_peer("2.2.2.2", 9000)
    stats = discovery.get_discovery_stats()
    assert stats["discovered_peers_count"] == 1


@pytest.mark.asyncio
async def test_retry_targets() -> None:
    node = DummyNode()
    discovery = PeerDiscovery(node)
    discovery.failed_peers[("3.3.3.3", 9000)] = time.time() - 601
    targets = discovery._get_retry_targets()
    assert ("3.3.3.3", 9000) in targets
    assert ("3.3.3.3", 9000) not in discovery.failed_peers


def _start_discovery_server(host: str, port: int, stop_event: threading.Event) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen()
    sock.settimeout(1)
    while not stop_event.is_set():
        try:
            conn, _ = sock.accept()
        except TimeoutError:
            continue
        with conn:
            length_data = conn.recv(4)
            if len(length_data) != 4:
                continue
            length = int.from_bytes(length_data, "big")
            _ = conn.recv(length)
            response = {
                "sender_id": "peer",
                "peer_info": {},
                "capabilities": {},
            }
            resp_bytes = json.dumps(response).encode("utf-8")
            conn.send(len(resp_bytes).to_bytes(4, "big") + resp_bytes)
    sock.close()


def test_discovery_delivery_rate_and_latency() -> None:
    stop_event = threading.Event()
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    host, port = server_sock.getsockname()
    server_sock.close()

    server_thread = threading.Thread(
        target=_start_discovery_server, args=(host, port, stop_event), daemon=True
    )
    server_thread.start()
    time.sleep(0.1)

    node = DummyNode()
    discovery = PeerDiscovery(node)

    attempts = 20
    successes = 0
    latencies: list[float] = []

    for _ in range(attempts):
        start = time.perf_counter()
        discovery._discover_peer_sync(host, port)
        elapsed = time.perf_counter() - start
        if discovery.stats["peers_discovered"] > successes:
            successes += 1
            latencies.append(elapsed)

    stop_event.set()
    server_thread.join(timeout=1)

    success_rate = successes / attempts
    avg_latency = sum(latencies) / len(latencies)

    assert success_rate >= 0.99
    assert avg_latency < 0.1
