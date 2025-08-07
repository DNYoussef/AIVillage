from pathlib import Path
import sys

# Ensure src package is importable
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.p2p.p2p_node import P2PNode, PeerCapabilities


def create_peer(idx: int) -> PeerCapabilities:
    return PeerCapabilities(
        device_id=f"peer-{idx}",
        cpu_cores=4,
        ram_mb=8192,
        battery_percent=80,
        network_type="wifi",
    )


def test_get_suitable_evolution_peers_respects_network_size():
    node = P2PNode(node_id="test-node")

    # Add more than five peers to the registry
    for i in range(7):
        peer = create_peer(i)
        node.peer_registry[peer.device_id] = peer
        # Populate connections to simulate an active network
        node.connections[peer.device_id] = None

    peers = node.get_suitable_evolution_peers()

    assert len(peers) == 7
