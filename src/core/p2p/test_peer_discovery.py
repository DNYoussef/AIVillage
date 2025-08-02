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

    async def send_to_peer(self, peer_id, message):  # pragma: no cover - placeholder
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
