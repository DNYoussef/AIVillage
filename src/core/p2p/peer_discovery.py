"""Peer Discovery System for Evolution-Aware P2P Network."""

import asyncio
import contextlib
import ipaddress
import json
import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryConfig:
    """Configuration for peer discovery."""

    discovery_interval: float = 30.0  # seconds
    discovery_timeout: float = 2.0  # seconds per discovery attempt
    max_discovery_ips: int = 20  # max IPs to scan per cycle
    discovery_ports: list[int] = None  # ports to scan
    enable_mdns: bool = True  # enable mDNS discovery
    enable_broadcast: bool = True  # enable broadcast discovery
    enable_dht: bool = False  # enable DHT discovery (future)

    def __post_init__(self):
        if self.discovery_ports is None:
            self.discovery_ports = [9000, 9001, 9002, 9003, 9004, 9005]


class PeerDiscovery:
    """Advanced peer discovery for evolution coordination."""

    def __init__(self, p2p_node) -> None:
        self.p2p_node = p2p_node
        self.config = DiscoveryConfig()

        # Discovery state
        self.discovery_active = False
        self.discovery_task: asyncio.Task | None = None

        # Discovered peers
        self.discovered_peers: set[tuple[str, int]] = set()
        self.failed_peers: dict[tuple[str, int], float] = {}  # peer -> last_failure_time
        self.peer_response_times: dict[tuple[str, int], float] = {}

        # Statistics
        self.stats = {
            "discovery_cycles": 0,
            "peers_discovered": 0,
            "discovery_failures": 0,
            "avg_response_time": 0.0,
            "last_discovery_time": None,
        }

        # Background discovery
        self.discovery_queue = queue.Queue()
        self.discovery_threads: list[threading.Thread] = []

    async def start_discovery(self) -> None:
        """Start peer discovery process."""
        if self.discovery_active:
            logger.warning("Peer discovery already active")
            return

        self.discovery_active = True

        # Start main discovery loop
        self.discovery_task = asyncio.create_task(self._discovery_loop())

        # Start background discovery threads
        for i in range(3):  # 3 discovery threads
            thread = threading.Thread(target=self._discovery_worker, daemon=True, name=f"PeerDiscovery-{i}")
            thread.start()
            self.discovery_threads.append(thread)

        logger.info("Peer discovery started")

    async def stop_discovery(self) -> None:
        """Stop peer discovery process."""
        self.discovery_active = False

        if self.discovery_task:
            self.discovery_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.discovery_task

        # Stop discovery threads
        for _ in self.discovery_threads:
            self.discovery_queue.put(None)  # Poison pill

        logger.info("Peer discovery stopped")

    async def _discovery_loop(self) -> None:
        """Main discovery coordination loop."""
        while self.discovery_active:
            try:
                await self._run_discovery_cycle()
                await asyncio.sleep(self.config.discovery_interval)
            except Exception as e:
                logger.exception(f"Discovery loop error: {e}")
                await asyncio.sleep(10)

    async def _run_discovery_cycle(self) -> None:
        """Run a complete discovery cycle."""
        start_time = time.time()

        # Get discovery targets
        targets = await self._get_discovery_targets()

        if not targets:
            return

        # Queue discovery tasks
        discovery_tasks = []
        for host, port in targets[: self.config.max_discovery_ips]:
            discovery_tasks.append((host, port))

        # Submit tasks to worker threads
        for task in discovery_tasks:
            try:
                self.discovery_queue.put_nowait(task)
            except queue.Full:
                break  # Queue full, skip remaining tasks

        # Update stats
        self.stats["discovery_cycles"] += 1
        self.stats["last_discovery_time"] = time.time()

        cycle_time = time.time() - start_time
        logger.debug(f"Discovery cycle completed in {cycle_time:.2f}s, queued {len(discovery_tasks)} tasks")

    def _discovery_worker(self) -> None:
        """Background worker for peer discovery."""
        while self.discovery_active:
            try:
                task = self.discovery_queue.get(timeout=1.0)
                if task is None:  # Poison pill
                    break

                host, port = task
                self._discover_peer_sync(host, port)

            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"Discovery worker error: {e}")

    def _discover_peer_sync(self, host: str, port: int) -> None:
        """Synchronously discover a single peer."""
        start_time = time.time()

        try:
            # Skip recently failed peers
            peer_addr = (host, port)
            if peer_addr in self.failed_peers:
                last_failure = self.failed_peers[peer_addr]
                if time.time() - last_failure < 300:  # 5 minute cooldown
                    return

            # Skip self
            if port == self.p2p_node.listen_port and host in self._get_local_ips():
                return

            # Attempt connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.discovery_timeout)

            try:
                sock.connect((host, port))

                # Send discovery message
                discovery_msg = {
                    "type": "PEER_DISCOVERY",
                    "sender_id": self.p2p_node.node_id,
                    "sender_port": self.p2p_node.listen_port,
                    "timestamp": time.time(),
                    "capabilities": (
                        self.p2p_node.local_capabilities.__dict__ if self.p2p_node.local_capabilities else {}
                    ),
                }

                # Simple message format for discovery
                msg_data = json.dumps(discovery_msg).encode("utf-8")
                length_data = len(msg_data).to_bytes(4, "big")

                sock.send(length_data + msg_data)

                # Read response
                resp_length_data = sock.recv(4)
                if len(resp_length_data) == 4:
                    resp_length = int.from_bytes(resp_length_data, "big")
                    resp_data = sock.recv(resp_length)

                    if len(resp_data) == resp_length:
                        response = json.loads(resp_data.decode("utf-8"))

                        # Process discovery response
                        response_time = time.time() - start_time
                        self._process_discovery_response(host, port, response, response_time)

                        # Update stats
                        self.stats["peers_discovered"] += 1
                        self.peer_response_times[peer_addr] = response_time

                        # Remove from failed peers
                        if peer_addr in self.failed_peers:
                            del self.failed_peers[peer_addr]

                        logger.debug(f"Discovered peer at {host}:{port} (response: {response_time:.3f}s)")

            finally:
                sock.close()

        except Exception as e:
            # Mark as failed
            self.failed_peers[(host, port)] = time.time()
            self.stats["discovery_failures"] += 1
            logger.debug(f"Discovery failed for {host}:{port}: {e}")

    def _process_discovery_response(self, host: str, port: int, response: dict, response_time: float) -> None:
        """Process discovery response from peer."""
        try:
            peer_id = response.get("sender_id")
            response.get("peer_info", {})
            capabilities_data = response.get("capabilities", {})

            if not peer_id:
                return

            # Create peer capabilities
            from .p2p_node import PeerCapabilities

            capabilities = PeerCapabilities(
                device_id=peer_id,
                cpu_cores=capabilities_data.get("cpu_cores", 1),
                ram_mb=capabilities_data.get("ram_mb", 1024),
                battery_percent=capabilities_data.get("battery_percent"),
                network_type=capabilities_data.get("network_type", "ethernet"),
                device_type=capabilities_data.get("device_type", "unknown"),
                performance_tier=capabilities_data.get("performance_tier", "medium"),
                evolution_capacity=capabilities_data.get("evolution_capacity", 0.5),
                available_for_evolution=capabilities_data.get("available_for_evolution", True),
                latency_ms=response_time * 1000,
                last_seen=time.time(),
            )

            # Register peer
            self.p2p_node.peer_registry[peer_id] = capabilities
            self.discovered_peers.add((host, port))

            # Initiate connection if not already connected
            if peer_id not in self.p2p_node.connections:
                asyncio.create_task(self._connect_to_discovered_peer(host, port, peer_id))

        except Exception as e:
            logger.exception(f"Error processing discovery response from {host}:{port}: {e}")

    async def _connect_to_discovered_peer(self, host: str, port: int, peer_id: str) -> None:
        """Connect to discovered peer."""
        try:
            if self.p2p_node.use_tls and self.p2p_node.ssl_context:
                reader, writer = await asyncio.open_connection(host, port, ssl=self.p2p_node.ssl_context)
            else:
                reader, writer = await asyncio.open_connection(host, port)

            # Send introduction
            intro_message = {
                "type": "PEER_INTRODUCTION",
                "sender_id": self.p2p_node.node_id,
                "capabilities": (self.p2p_node.local_capabilities.__dict__ if self.p2p_node.local_capabilities else {}),
            }

            # Use the P2P node's message sending method
            await self.p2p_node.send_to_peer(peer_id, intro_message)

            # Start handling this connection
            self.p2p_node.connections[peer_id] = writer
            asyncio.create_task(self.p2p_node._handle_connection(reader, writer))

            logger.info(f"Connected to discovered peer {peer_id} at {host}:{port}")

        except Exception as e:
            logger.exception(f"Failed to connect to discovered peer {peer_id}: {e}")

    async def _get_discovery_targets(self) -> list[tuple[str, int]]:
        """Get list of discovery targets."""
        targets = []

        # Local network discovery
        if self.config.enable_broadcast:
            targets.extend(await self._get_local_network_targets())

        # mDNS discovery
        if self.config.enable_mdns:
            targets.extend(await self._get_mdns_targets())

        # Known peer retry
        targets.extend(self._get_retry_targets())

        # Remove duplicates and shuffle
        unique_targets = list(set(targets))

        # Sort by priority (prefer low latency peers)
        unique_targets.sort(key=lambda t: self.peer_response_times.get(t, float("inf")))

        return unique_targets

    async def _get_local_network_targets(self) -> list[tuple[str, int]]:
        """Get local network discovery targets."""
        targets = []

        try:
            # Get local IP address
            local_ips = self._get_local_ips()

            for local_ip in local_ips:
                # Get network range
                network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)

                # Sample IPs from network (not all to avoid flooding)
                host_ips = list(network.hosts())

                # Take every 10th IP for discovery
                sample_ips = host_ips[::10][: self.config.max_discovery_ips // 2]

                for ip in sample_ips:
                    ip_str = str(ip)
                    if ip_str not in local_ips:  # Skip self
                        for port in self.config.discovery_ports:
                            targets.append((ip_str, port))

        except Exception as e:
            logger.debug(f"Local network discovery failed: {e}")

        return targets

    async def _get_mdns_targets(self) -> list[tuple[str, int]]:
        """Get mDNS discovery targets."""
        targets = []

        try:
            # This would implement mDNS/DNS-SD discovery
            # For now, return empty list
            # In production, this would use libraries like zeroconf
            pass

        except Exception as e:
            logger.debug(f"mDNS discovery failed: {e}")

        return targets

    def _get_retry_targets(self) -> list[tuple[str, int]]:
        """Get targets for retry (previously failed peers)."""
        targets = []
        current_time = time.time()

        # Retry failed peers after cooldown
        for peer_addr, failure_time in list(self.failed_peers.items()):
            if current_time - failure_time > 600:  # 10 minute retry
                targets.append(peer_addr)
                del self.failed_peers[peer_addr]  # Remove from failed list

        return targets

    def _get_local_ips(self) -> list[str]:
        """Get local IP addresses."""
        local_ips = []

        try:
            # Get primary IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            primary_ip = s.getsockname()[0]
            s.close()
            local_ips.append(primary_ip)

            # Get all interface IPs
            import netifaces

            for interface in netifaces.interfaces():
                addresses = netifaces.ifaddresses(interface)
                if netifaces.AF_INET in addresses:
                    for addr_info in addresses[netifaces.AF_INET]:
                        ip = addr_info.get("addr")
                        if ip and ip not in local_ips and not ip.startswith("127."):
                            local_ips.append(ip)

        except Exception:
            # Fallback if netifaces not available
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ips.append(s.getsockname()[0])
                s.close()
            except:
                local_ips.append("127.0.0.1")

        return local_ips

    def add_known_peer(self, host: str, port: int) -> None:
        """Manually add a known peer for discovery."""
        self.discovered_peers.add((host, port))
        logger.info(f"Added known peer: {host}:{port}")

    def remove_peer(self, host: str, port: int) -> None:
        """Remove peer from discovery."""
        peer_addr = (host, port)
        self.discovered_peers.discard(peer_addr)
        self.failed_peers.pop(peer_addr, None)
        self.peer_response_times.pop(peer_addr, None)

    def get_discovery_stats(self) -> dict[str, Any]:
        """Get discovery statistics."""
        # Calculate average response time
        response_times = list(self.peer_response_times.values())
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0

        return {
            **self.stats,
            "avg_response_time": avg_response_time,
            "discovered_peers_count": len(self.discovered_peers),
            "failed_peers_count": len(self.failed_peers),
            "active_discovery": self.discovery_active,
            "discovery_queue_size": self.discovery_queue.qsize(),
            "worker_threads": len(self.discovery_threads),
        }

    def force_discovery_cycle(self) -> None:
        """Force an immediate discovery cycle."""
        if self.discovery_active:
            asyncio.create_task(self._run_discovery_cycle())
