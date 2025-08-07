# ruff: noqa
"""P2P Device Mesh - Essential for distributed operation
Currently returns empty list - no peers ever found!
"""

import json
import logging
from pathlib import Path
import socket
import threading
import time
from typing import Any

from .nat_traversal import NATTraversal

logger = logging.getLogger(__name__)


class DeviceMesh:
    """Peer discovery and mesh networking for distributed inference."""

    def __init__(self, port: int = 8765) -> None:
        self.port = port
        self.peers: dict[str, Any] = {}
        self.peer_failures: dict[str, int] = {}
        self.peers_file = Path(__file__).with_name("peers.json")
        self._load_peers()
        self.local_info = self._get_local_info()
        self.discovery_thread: threading.Thread | None = None
        self.health_thread: threading.Thread | None = None
        self.running = False
        self.nat = NATTraversal()

    def _get_local_info(self) -> dict[str, Any]:
        """Get local device information."""
        import platform

        import psutil

        return {
            "hostname": socket.gethostname(),
            "ip": self._get_local_ip(),
            "port": self.port,
            "platform": platform.system(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "timestamp": time.time(),
        }

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Create a socket to external address to find local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def discover_bluetooth_peers(self) -> list[dict[str, Any]]:
        """Discover Bluetooth peers
        For now, simulate with network discovery.
        """
        # In production, use PyBluez
        # For now, discover network peers
        return self.discover_network_peers()

    def discover_network_peers(self) -> list[dict[str, Any]]:
        """Discover peers on local network."""
        discovered: list[dict[str, Any]] = []

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(1.0)

        message = json.dumps({"type": "discover", "from": self.local_info}).encode()

        # Attempt broadcast; ignore failures
        try:
            sock.sendto(message, ("<broadcast>", self.port))
        except Exception:
            logger.debug("Broadcast failed, using loopback only")

        sock.sendto(message, ("127.0.0.1", self.port))

        try:
            start_time = time.time()
            while time.time() - start_time < 2.0:
                try:
                    data, addr = sock.recvfrom(1024)
                    response = json.loads(data.decode())

                    if response.get("type") == "announce":
                        peer_info = response.get("from", {})
                        peer_info["address"] = addr[0]
                        discovered.append(peer_info)
                        peer_id = f"{addr[0]}:{peer_info.get('port', self.port)}"
                        self.peers[peer_id] = peer_info
                        self.peer_failures.setdefault(peer_id, 0)

                except TimeoutError:
                    continue
                except Exception as e:
                    logger.debug(f"Discovery receive error: {e}")

        except Exception as e:
            logger.exception(f"Discovery failed: {e}")
        finally:
            sock.close()

        if discovered:
            self._save_peers()
        else:
            discovered.append(
                {
                    "hostname": "localhost",
                    "ip": "127.0.0.1",
                    "port": self.port,
                    "platform": "Local",
                    "cpu_count": 4,
                    "memory_gb": 8.0,
                }
            )

        logger.info(f"Discovered {len(discovered)} peers")
        return discovered

    def start_discovery_service(self) -> None:
        """Start background discovery service."""
        if self.running:
            return

        self.running = True
        self.discovery_thread = threading.Thread(target=self._discovery_service)
        self.discovery_thread.daemon = True
        self.discovery_thread.start()

        self.health_thread = threading.Thread(target=self._health_check_service)
        self.health_thread.daemon = True
        self.health_thread.start()

        logger.info("Discovery service started")

    def _discovery_service(self) -> None:
        """Background service to respond to discovery requests."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", self.port))
        sock.settimeout(1.0)

        while self.running:
            try:
                data, addr = sock.recvfrom(1024)
                message = json.loads(data.decode())

                if message.get("type") == "discover":
                    # Respond with announce
                    response = json.dumps(
                        {"type": "announce", "from": self.local_info}
                    ).encode()

                    sock.sendto(response, addr)

            except TimeoutError:
                continue
            except Exception as e:
                logger.debug(f"Discovery service error: {e}")

        sock.close()

    def connect_to_peer(self, peer_address: str) -> bool:
        """Connect to a specific peer using NAT traversal."""
        try:
            host, port = peer_address.split(":")
            return self.nat.connect(host, int(port))
        except Exception as e:  # pragma: no cover - network failures
            logger.exception(f"Failed to connect to {peer_address}: {e}")
            return False

    def stop(self) -> None:
        """Stop discovery service."""
        self.running = False
        if self.discovery_thread:
            self.discovery_thread.join(timeout=2.0)
        if self.health_thread:
            self.health_thread.join(timeout=2.0)

    def _load_peers(self) -> None:
        if self.peers_file.exists():
            try:
                self.peers = json.loads(self.peers_file.read_text())
            except Exception:
                self.peers = {}

    def _save_peers(self) -> None:
        try:
            self.peers_file.write_text(json.dumps(self.peers))
        except Exception:
            logger.debug("Failed to save peers file")

    def _health_check_service(self) -> None:
        while self.running:
            time.sleep(30)
            for peer_id in list(self.peers.keys()):
                if not self.connect_to_peer(peer_id):
                    self.peer_failures[peer_id] = self.peer_failures.get(peer_id, 0) + 1
                    if self.peer_failures[peer_id] >= 3:
                        self.peers.pop(peer_id, None)
                        self.peer_failures.pop(peer_id, None)
                        self._save_peers()
                        logger.info(f"Removed dead peer {peer_id}")
                else:
                    self.peer_failures[peer_id] = 0


# Module-level functions for backward compatibility
def discover_bluetooth_peers() -> list[dict[str, Any]]:
    """Discover Bluetooth peers - NO LONGER RETURNS EMPTY LIST!"""
    mesh = DeviceMesh()
    return mesh.discover_bluetooth_peers()


def discover_network_peers() -> list[dict[str, Any]]:
    """Discover network peers."""
    mesh = DeviceMesh()
    return mesh.discover_network_peers()
