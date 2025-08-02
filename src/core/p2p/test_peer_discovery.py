import argparse
from .peer_discovery import PeerDiscovery

# CLI option for tests
parser = argparse.ArgumentParser(description="Peer discovery test options")
parser.add_argument("--min-peers", type=int, default=1, help="minimum peers to discover")
# parse_known_args to ignore pytest's own flags
ARGS, _ = parser.parse_known_args()


def test_min_peers_cli_parsing():
    """Ensure the custom CLI flag is parsed correctly."""
    assert isinstance(ARGS.min_peers, int)


def test_peer_discovery_initial_state():
    """PeerDiscovery should start with no discovered peers."""
    class DummyNode:
        pass

    discovery = PeerDiscovery(DummyNode())
    assert discovery.discovered_peers == set()
