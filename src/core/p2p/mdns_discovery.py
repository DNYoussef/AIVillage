"""mDNS Peer Discovery for LibP2P Mesh Network.

Implements multicast DNS-based peer discovery for the AIVillage mesh network.
This allows nodes to automatically discover each other on the local network
without requiring centralized coordination.

Features:
- Service advertisement and discovery
- Automatic peer registration
- Network change detection
- IPv4/IPv6 support
- Service filtering and validation
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
import socket
import time
from typing import Any

# mDNS/Zeroconf imports
try:
    from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
    from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf

    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False
    logging.warning("Zeroconf not available for mDNS discovery")

logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """Information about a discovered peer."""

    peer_id: str
    addresses: list[str]
    port: int
    capabilities: dict[str, Any]
    discovered_at: float
    last_seen: float
    service_name: str

    def is_expired(self, timeout: float = 300.0) -> bool:
        """Check if peer info has expired."""
        return time.time() - self.last_seen > timeout

    def to_multiaddr(self) -> list[str]:
        """Convert to libp2p multiaddr format."""
        multiaddrs = []
        for addr in self.addresses:
            # TCP multiaddr
            multiaddrs.append(f"/ip4/{addr}/tcp/{self.port}")
            # WebSocket multiaddr
            multiaddrs.append(f"/ip4/{addr}/tcp/{self.port + 1}/ws")
        return multiaddrs


class mDNSDiscovery:
    """mDNS-based peer discovery service."""

    SERVICE_TYPE = "_aivillage._tcp.local."
    DISCOVERY_INTERVAL = 30.0
    PEER_TIMEOUT = 300.0  # 5 minutes

    def __init__(
        self,
        node_id: str,
        listen_port: int,
        capabilities: dict[str, Any] | None = None,
        service_name_prefix: str = "aivillage",
    ) -> None:
        self.node_id = node_id
        self.listen_port = listen_port
        self.capabilities = capabilities or {}
        self.service_name = f"{service_name_prefix}-{node_id[:8]}"

        # mDNS components
        self.zeroconf: AsyncZeroconf | None = None
        self.service_browser: AsyncServiceBrowser | None = None
        self.service_info: ServiceInfo | None = None

        # Discovery state
        self.discovered_peers: dict[str, PeerInfo] = {}
        self.peer_callbacks: list[Callable[[PeerInfo, str], None]] = (
            []
        )  # (peer_info, event_type)
        self.running = False

        # Network monitoring
        self.local_addresses: set[str] = set()
        self.network_change_callbacks: list[Callable[[], None]] = []

    async def start(self) -> None:
        """Start mDNS discovery service."""
        if not ZEROCONF_AVAILABLE:
            logger.warning("Zeroconf not available, mDNS discovery disabled")
            return

        logger.info(f"Starting mDNS discovery for node {self.node_id}")

        try:
            self.running = True

            # Initialize AsyncZeroconf
            self.zeroconf = AsyncZeroconf()

            # Register our service
            await self._register_service()

            # Start browsing for peers
            await self._start_browsing()

            # Start background tasks
            asyncio.create_task(self._cleanup_loop())
            asyncio.create_task(self._network_monitor_loop())

            logger.info(f"mDNS discovery started, advertising as {self.service_name}")

        except Exception as e:
            logger.exception(f"Failed to start mDNS discovery: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop mDNS discovery service."""
        logger.info("Stopping mDNS discovery")
        self.running = False

        if self.service_browser:
            await self.service_browser.async_cancel()

        if self.zeroconf:
            if self.service_info:
                await self.zeroconf.async_unregister_service(self.service_info)
            await self.zeroconf.async_close()

        self.discovered_peers.clear()

    def add_peer_callback(self, callback: Callable[[PeerInfo, str], None]) -> None:
        """Add callback for peer discovery events.

        Args:
            callback: Function called with (peer_info, event_type)
                     event_type can be 'discovered', 'updated', 'removed'
        """
        self.peer_callbacks.append(callback)

    def remove_peer_callback(self, callback: Callable[[PeerInfo, str], None]) -> None:
        """Remove peer discovery callback."""
        if callback in self.peer_callbacks:
            self.peer_callbacks.remove(callback)

    def add_network_change_callback(self, callback: Callable[[], None]) -> None:
        """Add callback for network changes."""
        self.network_change_callbacks.append(callback)

    async def _register_service(self) -> None:
        """Register our service with mDNS."""
        # Get local IP addresses
        local_addresses = await self._get_local_addresses()

        # Create service info
        properties = {
            "peer_id": self.node_id,
            "capabilities": json.dumps(self.capabilities),
            "version": "1.0",
            "protocol": "libp2p",
            "mesh_enabled": "true",
        }

        # Convert to bytes (zeroconf requirement)
        properties_bytes = {k.encode(): v.encode() for k, v in properties.items()}

        self.service_info = ServiceInfo(
            self.SERVICE_TYPE,
            f"{self.service_name}.{self.SERVICE_TYPE}",
            addresses=[socket.inet_aton(addr) for addr in local_addresses],
            port=self.listen_port,
            properties=properties_bytes,
            server=f"{self.service_name}.local.",
        )

        await self.zeroconf.async_register_service(self.service_info)
        logger.debug(
            f"Registered service: {self.service_name} on port {self.listen_port}"
        )

    async def _start_browsing(self) -> None:
        """Start browsing for peer services."""

        class ServiceListener:
            def __init__(self, discovery_instance) -> None:
                self.discovery = discovery_instance

            def remove_service(self, zeroconf, service_type, name) -> None:
                asyncio.create_task(self.discovery._on_service_removed(name))

            def add_service(self, zeroconf, service_type, name) -> None:
                asyncio.create_task(
                    self.discovery._on_service_discovered(zeroconf, name)
                )

            def update_service(self, zeroconf, service_type, name) -> None:
                asyncio.create_task(self.discovery._on_service_updated(zeroconf, name))

        listener = ServiceListener(self)
        self.service_browser = AsyncServiceBrowser(
            self.zeroconf.zeroconf, self.SERVICE_TYPE, listener
        )

    async def _on_service_discovered(self, zeroconf, service_name: str) -> None:
        """Handle newly discovered service."""
        try:
            # Skip our own service
            if service_name.startswith(self.service_name):
                return

            info = await self._get_service_info(service_name)
            if not info:
                return

            peer_info = await self._create_peer_info(info)
            if not peer_info:
                return

            # Skip if it's our own peer ID (redundant check)
            if peer_info.peer_id == self.node_id:
                return

            # Add or update peer
            old_peer = self.discovered_peers.get(peer_info.peer_id)
            self.discovered_peers[peer_info.peer_id] = peer_info

            event_type = "discovered" if not old_peer else "updated"
            logger.info(
                f"Peer {event_type}: {peer_info.peer_id} at {peer_info.addresses}"
            )

            # Notify callbacks
            for callback in self.peer_callbacks:
                try:
                    callback(peer_info, event_type)
                except Exception as e:
                    logger.exception(f"Error in peer callback: {e}")

        except Exception as e:
            logger.exception(f"Error processing discovered service {service_name}: {e}")

    async def _on_service_updated(self, zeroconf, service_name: str) -> None:
        """Handle service update."""
        await self._on_service_discovered(zeroconf, service_name)

    async def _on_service_removed(self, service_name: str) -> None:
        """Handle service removal."""
        try:
            # Find peer by service name
            removed_peer = None
            for peer_id, peer_info in self.discovered_peers.items():
                if peer_info.service_name == service_name:
                    removed_peer = (peer_id, peer_info)
                    break

            if removed_peer:
                peer_id, peer_info = removed_peer
                del self.discovered_peers[peer_id]

                logger.info(f"Peer removed: {peer_id}")

                # Notify callbacks
                for callback in self.peer_callbacks:
                    try:
                        callback(peer_info, "removed")
                    except Exception as e:
                        logger.exception(f"Error in peer removal callback: {e}")

        except Exception as e:
            logger.exception(f"Error processing service removal {service_name}: {e}")

    async def _get_service_info(self, service_name: str) -> ServiceInfo | None:
        """Get detailed info for a service."""
        try:
            return await self.zeroconf.async_get_service_info(
                self.SERVICE_TYPE, service_name
            )
        except Exception as e:
            logger.debug(f"Failed to get service info for {service_name}: {e}")
            return None

    async def _create_peer_info(self, service_info: ServiceInfo) -> PeerInfo | None:
        """Create PeerInfo from ServiceInfo."""
        try:
            # Extract properties
            properties = {}
            if service_info.properties:
                for key, value in service_info.properties.items():
                    try:
                        properties[key.decode()] = value.decode()
                    except:
                        continue

            peer_id = properties.get("peer_id")
            if not peer_id:
                logger.debug(f"Service {service_info.name} missing peer_id")
                return None

            # Parse capabilities
            capabilities = {}
            try:
                capabilities_str = properties.get("capabilities", "{}")
                capabilities = json.loads(capabilities_str)
            except json.JSONDecodeError:
                logger.debug(f"Invalid capabilities JSON for peer {peer_id}")

            # Get addresses
            addresses = []
            for addr_bytes in service_info.addresses:
                try:
                    addr = socket.inet_ntoa(addr_bytes)
                    # Skip loopback unless it's the only address
                    if addr != "127.0.0.1" or not addresses:
                        addresses.append(addr)
                except:
                    continue

            if not addresses:
                logger.debug(f"No valid addresses for peer {peer_id}")
                return None

            return PeerInfo(
                peer_id=peer_id,
                addresses=addresses,
                port=service_info.port,
                capabilities=capabilities,
                discovered_at=time.time(),
                last_seen=time.time(),
                service_name=service_info.name,
            )

        except Exception as e:
            logger.exception(f"Error creating peer info: {e}")
            return None

    async def _cleanup_loop(self) -> None:
        """Clean up expired peers."""
        while self.running:
            try:
                time.time()
                expired_peers = []

                for peer_id, peer_info in self.discovered_peers.items():
                    if peer_info.is_expired(self.PEER_TIMEOUT):
                        expired_peers.append(peer_id)

                # Remove expired peers
                for peer_id in expired_peers:
                    peer_info = self.discovered_peers.pop(peer_id)
                    logger.info(f"Peer expired: {peer_id}")

                    # Notify callbacks
                    for callback in self.peer_callbacks:
                        try:
                            callback(peer_info, "removed")
                        except Exception as e:
                            logger.exception(f"Error in peer expiry callback: {e}")

                await asyncio.sleep(60)  # Cleanup every minute

            except Exception as e:
                logger.exception(f"Cleanup loop error: {e}")
                await asyncio.sleep(10)

    async def _network_monitor_loop(self) -> None:
        """Monitor network changes."""
        while self.running:
            try:
                current_addresses = await self._get_local_addresses()

                if current_addresses != self.local_addresses:
                    logger.info(
                        "Network addresses changed, updating service registration"
                    )
                    self.local_addresses = current_addresses

                    # Re-register service with new addresses
                    if self.service_info:
                        await self.zeroconf.async_unregister_service(self.service_info)
                        await self._register_service()

                    # Notify callbacks
                    for callback in self.network_change_callbacks:
                        try:
                            callback()
                        except Exception as e:
                            logger.exception(f"Error in network change callback: {e}")

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.exception(f"Network monitor error: {e}")
                await asyncio.sleep(10)

    async def _get_local_addresses(self) -> set[str]:
        """Get local IP addresses."""
        addresses = set()

        try:
            # Get all network interfaces
            import netifaces

            for interface in netifaces.interfaces():
                try:
                    addrs = netifaces.ifaddresses(interface)

                    # IPv4 addresses
                    if netifaces.AF_INET in addrs:
                        for addr_info in addrs[netifaces.AF_INET]:
                            addr = addr_info.get("addr")
                            if addr and not addr.startswith("127."):
                                addresses.add(addr)

                    # TODO: Add IPv6 support if needed

                except:
                    continue

        except ImportError:
            # Fallback method without netifaces
            try:
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                if local_ip and not local_ip.startswith("127."):
                    addresses.add(local_ip)
            except:
                pass

        # Always include loopback as fallback
        if not addresses:
            addresses.add("127.0.0.1")

        return addresses

    def get_discovered_peers(self) -> dict[str, PeerInfo]:
        """Get all discovered peers."""
        return self.discovered_peers.copy()

    def get_peer(self, peer_id: str) -> PeerInfo | None:
        """Get specific peer info."""
        return self.discovered_peers.get(peer_id)

    def get_peer_count(self) -> int:
        """Get number of discovered peers."""
        return len(self.discovered_peers)

    def update_capabilities(self, capabilities: dict[str, Any]) -> None:
        """Update our advertised capabilities."""
        self.capabilities = capabilities

        # Re-register service with updated capabilities
        if self.service_info and self.zeroconf:
            asyncio.create_task(self._reregister_service())

    async def _reregister_service(self) -> None:
        """Re-register service with updated info."""
        try:
            if self.service_info:
                await self.zeroconf.async_unregister_service(self.service_info)
            await self._register_service()
        except Exception as e:
            logger.exception(f"Failed to re-register service: {e}")

    def is_running(self) -> bool:
        """Check if discovery is running."""
        return self.running and ZEROCONF_AVAILABLE

    def get_status(self) -> dict[str, Any]:
        """Get discovery status."""
        return {
            "running": self.running,
            "zeroconf_available": ZEROCONF_AVAILABLE,
            "node_id": self.node_id,
            "service_name": self.service_name,
            "listen_port": self.listen_port,
            "discovered_peers": len(self.discovered_peers),
            "local_addresses": list(self.local_addresses),
            "capabilities": self.capabilities,
        }
