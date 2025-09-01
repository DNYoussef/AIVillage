"""Security Mixnode Service - Privacy-aware routing and anonymity management

This service handles mixnode selection, privacy routing, and anonymity circuit
management for secure communications in the Navigator system.
"""

import asyncio
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import uuid

from ..interfaces.routing_interfaces import ISecurityMixnodeService, RoutingEvent
from ..events.event_bus import get_event_bus
from ..path_policy import PathProtocol, MessageContext

logger = logging.getLogger(__name__)


class PrivacyLevel(Enum):
    """Privacy protection levels"""

    MINIMAL = "minimal"  # Basic obfuscation
    STANDARD = "standard"  # Moderate protection
    HIGH = "high"  # Strong protection
    MAXIMUM = "maximum"  # Maximum anonymity


class MixnodeType(Enum):
    """Types of mixnodes available"""

    ENTRY = "entry"  # Entry point to mix network
    MIDDLE = "middle"  # Intermediate hop
    EXIT = "exit"  # Exit point from mix network
    BRIDGE = "bridge"  # Bridge/relay node
    GUARD = "guard"  # Guard/directory node


class CircuitState(Enum):
    """States of anonymity circuits"""

    BUILDING = "building"
    ESTABLISHED = "established"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    CLOSING = "closing"


@dataclass
class MixnodeInfo:
    """Information about a mixnode"""

    node_id: str
    node_type: MixnodeType
    network: str  # tor, i2p, betanet_mixnode, etc.
    bandwidth_mbps: float = 1.0
    latency_ms: float = 100.0
    reliability_score: float = 0.8
    uptime_hours: float = 24.0
    location_country: Optional[str] = None
    exit_policy: Optional[Dict[str, Any]] = None
    last_seen: float = field(default_factory=time.time)
    trust_score: float = 0.5  # 0.0 to 1.0
    flags: Set[str] = field(default_factory=set)  # Fast, Stable, Guard, etc.


@dataclass
class AnonymityCircuit:
    """Represents an anonymity circuit through mixnodes"""

    circuit_id: str
    destination: str
    privacy_level: PrivacyLevel
    mixnode_path: List[str]  # Ordered list of mixnode IDs
    state: CircuitState = CircuitState.BUILDING
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    bytes_transferred: int = 0
    latency_ms: float = 0.0
    hops: int = 0

    def __post_init__(self):
        self.hops = len(self.mixnode_path)


class SecurityMixnodeService(ISecurityMixnodeService):
    """Privacy-aware mixnode selection and security service

    Manages:
    - Mixnode discovery and maintenance
    - Privacy circuit construction and management
    - Anonymity level enforcement
    - Traffic flow obfuscation
    - Security policy compliance
    """

    def __init__(self):
        self.event_bus = get_event_bus()

        # Mixnode management
        self.available_mixnodes: Dict[str, MixnodeInfo] = {}
        self.mixnode_networks: Dict[str, Set[str]] = defaultdict(set)  # network -> node_ids
        self.mixnode_performance: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Circuit management
        self.active_circuits: Dict[str, AnonymityCircuit] = {}
        self.circuit_pool: Dict[PrivacyLevel, List[str]] = {level: [] for level in PrivacyLevel}
        self.circuit_history: deque[AnonymityCircuit] = deque(maxlen=1000)

        # Privacy configuration
        self.default_privacy_level = PrivacyLevel.STANDARD
        self.max_circuit_age_hours = 2.0  # Refresh circuits every 2 hours
        self.min_circuit_hops = 3
        self.max_circuit_hops = 5

        # Security policies
        self.blocked_countries: Set[str] = set()
        self.required_flags: Set[str] = {"Stable"}  # Require stable nodes
        self.min_bandwidth_mbps = 0.5
        self.min_uptime_hours = 12.0

        # Network preferences and capabilities
        self.network_preferences = {
            "tor": {"weight": 0.4, "anonymity": 0.9, "speed": 0.6},
            "i2p": {"weight": 0.3, "anonymity": 0.8, "speed": 0.5},
            "betanet_mixnode": {"weight": 0.3, "anonymity": 0.7, "speed": 0.8},
        }

        # Performance tracking
        self.privacy_metrics: Dict[str, Any] = {
            "circuits_created": 0,
            "circuits_failed": 0,
            "total_hops_used": 0,
            "avg_circuit_latency_ms": 0.0,
            "bytes_through_mixnodes": 0,
        }

        # Background tasks
        self.maintenance_task: Optional[asyncio.Task] = None
        self.circuit_monitor_task: Optional[asyncio.Task] = None
        self.running = False

        # Initialize with some default mixnodes
        self._initialize_default_mixnodes()

        logger.info("SecurityMixnodeService initialized")

    def _initialize_default_mixnodes(self) -> None:
        """Initialize with some default mixnode configurations"""
        # Simulated mixnodes for different networks
        default_nodes = [
            # Tor-like mixnodes
            MixnodeInfo(
                "tor_entry_1", MixnodeType.ENTRY, "tor", 5.0, 50.0, 0.9, 720, "DE", flags={"Fast", "Stable", "Guard"}
            ),
            MixnodeInfo(
                "tor_middle_1", MixnodeType.MIDDLE, "tor", 10.0, 40.0, 0.85, 480, "NL", flags={"Fast", "Stable"}
            ),
            MixnodeInfo("tor_exit_1", MixnodeType.EXIT, "tor", 8.0, 60.0, 0.8, 360, "SE", flags={"Fast", "Exit"}),
            # I2P-like mixnodes
            MixnodeInfo("i2p_router_1", MixnodeType.MIDDLE, "i2p", 2.0, 80.0, 0.75, 240, "CH", flags={"Stable"}),
            MixnodeInfo("i2p_router_2", MixnodeType.MIDDLE, "i2p", 3.0, 70.0, 0.8, 300, "IS", flags={"Fast", "Stable"}),
            # Betanet mixnodes
            MixnodeInfo(
                "betanet_mix_1",
                MixnodeType.ENTRY,
                "betanet_mixnode",
                15.0,
                30.0,
                0.9,
                168,
                "SG",
                flags={"Fast", "Stable"},
            ),
            MixnodeInfo(
                "betanet_mix_2",
                MixnodeType.MIDDLE,
                "betanet_mixnode",
                12.0,
                35.0,
                0.85,
                144,
                "JP",
                flags={"Fast", "Stable"},
            ),
            MixnodeInfo(
                "betanet_mix_3",
                MixnodeType.EXIT,
                "betanet_mixnode",
                20.0,
                25.0,
                0.95,
                200,
                "KR",
                flags={"Fast", "Stable", "Exit"},
            ),
        ]

        for node in default_nodes:
            self.available_mixnodes[node.node_id] = node
            self.mixnode_networks[node.network].add(node.node_id)

    async def start_service(self) -> None:
        """Start security and mixnode services"""
        if self.running:
            return

        self.running = True
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        self.circuit_monitor_task = asyncio.create_task(self._circuit_monitor_loop())

        logger.info("Security mixnode service started")

    async def stop_service(self) -> None:
        """Stop security services"""
        self.running = False

        for task in [self.maintenance_task, self.circuit_monitor_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Clean up active circuits
        for circuit in self.active_circuits.values():
            circuit.state = CircuitState.CLOSING

        logger.info("Security mixnode service stopped")

    async def select_privacy_mixnodes(self, destination: str, privacy_level: float = 0.8) -> List[str]:
        """Select mixnodes for privacy routing"""
        # Convert privacy level to enum
        if privacy_level >= 0.9:
            privacy_enum = PrivacyLevel.MAXIMUM
        elif privacy_level >= 0.7:
            privacy_enum = PrivacyLevel.HIGH
        elif privacy_level >= 0.5:
            privacy_enum = PrivacyLevel.STANDARD
        else:
            privacy_enum = PrivacyLevel.MINIMAL

        # Check if we have a suitable existing circuit
        suitable_circuit = await self._find_suitable_circuit(destination, privacy_enum)
        if suitable_circuit:
            return suitable_circuit.mixnode_path

        # Create new circuit
        circuit = await self._create_anonymity_circuit(destination, privacy_enum)
        if circuit:
            return circuit.mixnode_path

        return []

    async def _find_suitable_circuit(self, destination: str, privacy_level: PrivacyLevel) -> Optional[AnonymityCircuit]:
        """Find existing suitable circuit for destination"""
        for circuit in self.active_circuits.values():
            if (
                circuit.destination == destination
                and circuit.privacy_level == privacy_level
                and circuit.state == CircuitState.ESTABLISHED
                and time.time() - circuit.created_at < self.max_circuit_age_hours * 3600
            ):

                # Update last activity
                circuit.last_activity = time.time()
                return circuit

        return None

    async def _create_anonymity_circuit(
        self, destination: str, privacy_level: PrivacyLevel
    ) -> Optional[AnonymityCircuit]:
        """Create new anonymity circuit"""
        try:
            # Determine circuit parameters based on privacy level
            required_hops = self._get_required_hops(privacy_level)

            # Select mixnodes for the path
            mixnode_path = await self._select_circuit_path(destination, privacy_level, required_hops)

            if not mixnode_path or len(mixnode_path) < self.min_circuit_hops:
                logger.warning(f"Insufficient mixnodes for circuit (need {required_hops}, got {len(mixnode_path)})")
                return None

            # Create circuit
            circuit_id = f"circuit_{uuid.uuid4().hex[:8]}"
            circuit = AnonymityCircuit(
                circuit_id=circuit_id,
                destination=destination,
                privacy_level=privacy_level,
                mixnode_path=mixnode_path,
                state=CircuitState.BUILDING,
            )

            # Attempt to establish circuit
            if await self._establish_circuit(circuit):
                self.active_circuits[circuit_id] = circuit
                self.circuit_pool[privacy_level].append(circuit_id)

                # Update metrics
                self.privacy_metrics["circuits_created"] += 1
                self.privacy_metrics["total_hops_used"] += len(mixnode_path)

                logger.info(f"Created anonymity circuit {circuit_id} with {len(mixnode_path)} hops for {destination}")

                # Emit circuit creation event
                self._emit_security_event(
                    "circuit_created",
                    {
                        "circuit_id": circuit_id,
                        "destination": destination,
                        "privacy_level": privacy_level.value,
                        "hops": len(mixnode_path),
                        "networks_used": [self.available_mixnodes[node_id].network for node_id in mixnode_path],
                    },
                )

                return circuit
            else:
                self.privacy_metrics["circuits_failed"] += 1
                return None

        except Exception as e:
            logger.error(f"Failed to create anonymity circuit for {destination}: {e}")
            self.privacy_metrics["circuits_failed"] += 1
            return None

    def _get_required_hops(self, privacy_level: PrivacyLevel) -> int:
        """Get required number of hops for privacy level"""
        hop_requirements = {
            PrivacyLevel.MINIMAL: 2,
            PrivacyLevel.STANDARD: 3,
            PrivacyLevel.HIGH: 4,
            PrivacyLevel.MAXIMUM: 5,
        }
        return hop_requirements.get(privacy_level, 3)

    async def _select_circuit_path(
        self, destination: str, privacy_level: PrivacyLevel, required_hops: int
    ) -> List[str]:
        """Select optimal path through mixnodes"""
        # Filter suitable mixnodes
        suitable_nodes = self._filter_suitable_mixnodes(privacy_level)

        if len(suitable_nodes) < required_hops:
            logger.warning(f"Not enough suitable mixnodes: need {required_hops}, have {len(suitable_nodes)}")
            return []

        # Group nodes by type and network
        nodes_by_type = defaultdict(list)
        nodes_by_network = defaultdict(list)

        for node_id in suitable_nodes:
            node = self.available_mixnodes[node_id]
            nodes_by_type[node.node_type].append(node_id)
            nodes_by_network[node.network].append(node_id)

        # Build path with diversity requirements
        path = []
        used_networks = set()
        used_countries = set()

        # Select entry node
        entry_candidates = nodes_by_type.get(MixnodeType.ENTRY, [])
        if not entry_candidates:
            entry_candidates = nodes_by_type.get(MixnodeType.MIDDLE, [])

        if entry_candidates:
            entry_node = self._select_best_node(entry_candidates, used_networks, used_countries)
            if entry_node:
                path.append(entry_node)
                node_info = self.available_mixnodes[entry_node]
                used_networks.add(node_info.network)
                if node_info.location_country:
                    used_countries.add(node_info.location_country)

        # Select middle nodes
        middle_candidates = nodes_by_type.get(MixnodeType.MIDDLE, [])
        remaining_hops = required_hops - len(path) - 1  # Reserve 1 for exit if needed

        for _ in range(remaining_hops):
            if not middle_candidates:
                break

            middle_node = self._select_best_node(middle_candidates, used_networks, used_countries)
            if middle_node and middle_node not in path:
                path.append(middle_node)
                node_info = self.available_mixnodes[middle_node]
                used_networks.add(node_info.network)
                if node_info.location_country:
                    used_countries.add(node_info.location_country)
                middle_candidates.remove(middle_node)

        # Select exit node if needed and available
        if len(path) < required_hops:
            exit_candidates = nodes_by_type.get(MixnodeType.EXIT, [])
            if exit_candidates:
                exit_node = self._select_best_node(exit_candidates, used_networks, used_countries)
                if exit_node and exit_node not in path:
                    path.append(exit_node)

        # Fill remaining slots with any suitable nodes if path is still short
        while len(path) < required_hops and suitable_nodes:
            remaining_candidates = [n for n in suitable_nodes if n not in path]
            if not remaining_candidates:
                break

            next_node = self._select_best_node(remaining_candidates, used_networks, used_countries)
            if next_node:
                path.append(next_node)
                node_info = self.available_mixnodes[next_node]
                used_networks.add(node_info.network)
                if node_info.location_country:
                    used_countries.add(node_info.location_country)

        return path

    def _filter_suitable_mixnodes(self, privacy_level: PrivacyLevel) -> List[str]:
        """Filter mixnodes suitable for privacy level"""
        suitable = []

        for node_id, node_info in self.available_mixnodes.items():
            # Check basic requirements
            if (
                node_info.bandwidth_mbps >= self.min_bandwidth_mbps
                and node_info.uptime_hours >= self.min_uptime_hours
                and node_info.reliability_score >= 0.5
            ):

                # Check required flags
                if self.required_flags.issubset(node_info.flags):
                    # Check country restrictions
                    if node_info.location_country not in self.blocked_countries:
                        # Check privacy level requirements
                        if self._meets_privacy_requirements(node_info, privacy_level):
                            suitable.append(node_id)

        return suitable

    def _meets_privacy_requirements(self, node_info: MixnodeInfo, privacy_level: PrivacyLevel) -> bool:
        """Check if node meets privacy level requirements"""
        min_trust_scores = {
            PrivacyLevel.MINIMAL: 0.3,
            PrivacyLevel.STANDARD: 0.5,
            PrivacyLevel.HIGH: 0.7,
            PrivacyLevel.MAXIMUM: 0.8,
        }

        required_trust = min_trust_scores.get(privacy_level, 0.5)

        return node_info.trust_score >= required_trust

    def _select_best_node(
        self, candidates: List[str], used_networks: Set[str], used_countries: Set[str]
    ) -> Optional[str]:
        """Select best node from candidates with diversity preferences"""
        if not candidates:
            return None

        # Score each candidate
        node_scores = {}

        for node_id in candidates:
            node_info = self.available_mixnodes[node_id]
            score = 0.0

            # Base score from reliability and performance
            score += node_info.reliability_score * 40
            score += min(node_info.bandwidth_mbps / 10.0, 1.0) * 30  # Up to 30 points for bandwidth
            score += (1.0 - min(node_info.latency_ms / 200.0, 1.0)) * 20  # Lower latency = higher score
            score += node_info.trust_score * 10

            # Diversity bonuses
            if node_info.network not in used_networks:
                score += 20  # Network diversity bonus

            if node_info.location_country and node_info.location_country not in used_countries:
                score += 15  # Geographic diversity bonus

            # Flag bonuses
            if "Fast" in node_info.flags:
                score += 5
            if "Stable" in node_info.flags:
                score += 5

            node_scores[node_id] = score

        # Select highest scoring node
        best_node = max(node_scores, key=node_scores.get)
        return best_node

    async def _establish_circuit(self, circuit: AnonymityCircuit) -> bool:
        """Establish the anonymity circuit"""
        try:
            logger.debug(f"Establishing circuit {circuit.circuit_id} through {len(circuit.mixnode_path)} hops")

            # Simulate circuit establishment time
            establishment_time = len(circuit.mixnode_path) * 0.2  # 200ms per hop
            await asyncio.sleep(establishment_time)

            # Calculate circuit latency
            total_latency = 0.0
            for node_id in circuit.mixnode_path:
                node_info = self.available_mixnodes[node_id]
                total_latency += node_info.latency_ms

            circuit.latency_ms = total_latency
            circuit.state = CircuitState.ESTABLISHED

            # Update average circuit latency metric
            current_avg = self.privacy_metrics["avg_circuit_latency_ms"]
            circuits_created = self.privacy_metrics["circuits_created"]

            if circuits_created > 0:
                new_avg = ((current_avg * circuits_created) + total_latency) / (circuits_created + 1)
                self.privacy_metrics["avg_circuit_latency_ms"] = new_avg
            else:
                self.privacy_metrics["avg_circuit_latency_ms"] = total_latency

            logger.info(f"Circuit {circuit.circuit_id} established with {total_latency:.1f}ms latency")
            return True

        except Exception as e:
            logger.error(f"Failed to establish circuit {circuit.circuit_id}: {e}")
            circuit.state = CircuitState.FAILED
            return False

    def ensure_routing_privacy(self, protocol: PathProtocol, context: MessageContext) -> Dict[str, Any]:
        """Ensure privacy requirements are met for routing"""
        privacy_config = {
            "privacy_enabled": context.privacy_required,
            "anonymity_level": "none",
            "mixnode_hops": 0,
            "traffic_obfuscation": False,
            "timing_obfuscation": False,
            "payload_padding": False,
        }

        if not context.privacy_required:
            return privacy_config

        # Determine privacy level based on protocol and context
        if protocol in [PathProtocol.BETANET, PathProtocol.SCION]:
            # These protocols support mixnode routing
            privacy_level = self._determine_privacy_level(context)

            privacy_config.update(
                {
                    "anonymity_level": privacy_level.value,
                    "mixnode_hops": self._get_required_hops(privacy_level),
                    "traffic_obfuscation": privacy_level in [PrivacyLevel.HIGH, PrivacyLevel.MAXIMUM],
                    "timing_obfuscation": privacy_level == PrivacyLevel.MAXIMUM,
                    "payload_padding": privacy_level in [PrivacyLevel.HIGH, PrivacyLevel.MAXIMUM],
                }
            )

            # Add protocol-specific privacy configurations
            if protocol == PathProtocol.BETANET:
                privacy_config["betanet_mixnodes"] = True
                privacy_config["layer_encryption"] = True
            elif protocol == PathProtocol.SCION:
                privacy_config["path_diversity"] = True
                privacy_config["multipath_mixing"] = True

        elif protocol == PathProtocol.BITCHAT:
            # BitChat has limited privacy options
            privacy_config.update(
                {"anonymity_level": "mesh_obfuscation", "mesh_routing": True, "peer_discovery_privacy": True}
            )

        logger.debug(f"Privacy configuration for {protocol.value}: {privacy_config}")

        return privacy_config

    def _determine_privacy_level(self, context: MessageContext) -> PrivacyLevel:
        """Determine required privacy level from message context"""
        # High priority messages may need maximum privacy
        if context.priority >= 9:
            return PrivacyLevel.MAXIMUM
        elif context.priority >= 7:
            return PrivacyLevel.HIGH
        elif context.priority >= 5:
            return PrivacyLevel.STANDARD
        else:
            return PrivacyLevel.MINIMAL

    async def manage_anonymity_circuits(self) -> Dict[str, Any]:
        """Manage anonymous routing circuits"""
        current_time = time.time()

        # Circuit statistics
        circuits_by_state = defaultdict(int)
        circuits_by_privacy = defaultdict(int)

        for circuit in self.active_circuits.values():
            circuits_by_state[circuit.state.value] += 1
            circuits_by_privacy[circuit.privacy_level.value] += 1

        # Find circuits that need maintenance
        circuits_needing_refresh = []
        circuits_to_close = []

        for circuit_id, circuit in self.active_circuits.items():
            # Check age
            age_hours = (current_time - circuit.created_at) / 3600.0
            if age_hours > self.max_circuit_age_hours:
                circuits_needing_refresh.append(circuit_id)

            # Check inactivity
            inactive_time = current_time - circuit.last_activity
            if inactive_time > 1800:  # 30 minutes inactive
                circuits_to_close.append(circuit_id)

            # Check circuit health
            if circuit.state == CircuitState.FAILED:
                circuits_to_close.append(circuit_id)

        # Perform maintenance actions
        refreshed_count = 0
        for circuit_id in circuits_needing_refresh:
            if await self._refresh_circuit(circuit_id):
                refreshed_count += 1

        closed_count = 0
        for circuit_id in circuits_to_close:
            await self._close_circuit(circuit_id)
            closed_count += 1

        management_result = {
            "active_circuits": len(self.active_circuits),
            "circuits_by_state": dict(circuits_by_state),
            "circuits_by_privacy_level": dict(circuits_by_privacy),
            "circuits_refreshed": refreshed_count,
            "circuits_closed": closed_count,
            "avg_circuit_age_hours": sum((current_time - c.created_at) / 3600.0 for c in self.active_circuits.values())
            / max(1, len(self.active_circuits)),
            "total_bytes_transferred": sum(c.bytes_transferred for c in self.active_circuits.values()),
        }

        # Emit circuit management event
        self._emit_security_event("circuits_managed", management_result)

        return management_result

    async def _refresh_circuit(self, circuit_id: str) -> bool:
        """Refresh an aging circuit"""
        if circuit_id not in self.active_circuits:
            return False

        old_circuit = self.active_circuits[circuit_id]

        # Create new circuit with same parameters
        new_circuit = await self._create_anonymity_circuit(old_circuit.destination, old_circuit.privacy_level)

        if new_circuit:
            # Close old circuit
            await self._close_circuit(circuit_id)
            logger.info(f"Refreshed circuit for {old_circuit.destination}")
            return True

        return False

    async def _close_circuit(self, circuit_id: str) -> None:
        """Close anonymity circuit"""
        if circuit_id not in self.active_circuits:
            return

        circuit = self.active_circuits[circuit_id]
        circuit.state = CircuitState.CLOSING

        # Move to history
        self.circuit_history.append(circuit)

        # Remove from active circuits
        del self.active_circuits[circuit_id]

        # Remove from pool
        if circuit_id in self.circuit_pool[circuit.privacy_level]:
            self.circuit_pool[circuit.privacy_level].remove(circuit_id)

        logger.debug(f"Closed circuit {circuit_id}")

    async def _maintenance_loop(self) -> None:
        """Background maintenance for mixnodes and circuits"""
        while self.running:
            try:
                # Update mixnode information
                await self._update_mixnode_status()

                # Perform circuit maintenance
                await self.manage_anonymity_circuits()

                # Clean up old data
                await self._cleanup_old_data()

                # Sleep for 5 minutes
                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"Error in security maintenance loop: {e}")
                await asyncio.sleep(60)

    async def _circuit_monitor_loop(self) -> None:
        """Monitor circuit health and performance"""
        while self.running:
            try:
                # Check circuit health
                for circuit in self.active_circuits.values():
                    if await self._check_circuit_health(circuit):
                        circuit.state = CircuitState.ACTIVE
                    else:
                        circuit.state = CircuitState.DEGRADED

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in circuit monitor loop: {e}")
                await asyncio.sleep(30)

    async def _update_mixnode_status(self) -> None:
        """Update mixnode availability and performance"""
        # In a real implementation, this would query mixnode directories
        # For now, simulate some updates
        for node_id, node_info in self.available_mixnodes.items():
            # Simulate reliability changes
            node_info.reliability_score += random.uniform(-0.05, 0.05)
            node_info.reliability_score = max(0.1, min(0.99, node_info.reliability_score))

            # Update last seen
            node_info.last_seen = time.time()

    async def _check_circuit_health(self, circuit: AnonymityCircuit) -> bool:
        """Check if circuit is healthy"""
        # Check if all nodes in path are still available
        for node_id in circuit.mixnode_path:
            if node_id not in self.available_mixnodes:
                return False

            node = self.available_mixnodes[node_id]
            if time.time() - node.last_seen > 3600:  # Node not seen for 1 hour
                return False

            if node.reliability_score < 0.3:  # Node reliability too low
                return False

        return True

    async def _cleanup_old_data(self) -> None:
        """Clean up old mixnode and circuit data"""
        current_time = time.time()

        # Remove stale mixnodes
        stale_nodes = [
            node_id
            for node_id, node_info in self.available_mixnodes.items()
            if current_time - node_info.last_seen > 7200  # 2 hours
        ]

        for node_id in stale_nodes:
            node_info = self.available_mixnodes[node_id]
            self.mixnode_networks[node_info.network].discard(node_id)
            del self.available_mixnodes[node_id]
            logger.debug(f"Removed stale mixnode {node_id}")

    def _emit_security_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit security event"""
        event = RoutingEvent(
            event_type=event_type, timestamp=time.time(), source_service="SecurityMixnodeService", data=data
        )
        self.event_bus.publish(event)

    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security and privacy statistics"""
        return {
            "privacy_metrics": self.privacy_metrics,
            "available_mixnodes": len(self.available_mixnodes),
            "mixnodes_by_network": {network: len(nodes) for network, nodes in self.mixnode_networks.items()},
            "active_circuits": len(self.active_circuits),
            "circuits_by_privacy_level": {level.value: len(circuits) for level, circuits in self.circuit_pool.items()},
            "default_privacy_level": self.default_privacy_level.value,
            "security_policies": {
                "blocked_countries": list(self.blocked_countries),
                "required_flags": list(self.required_flags),
                "min_bandwidth_mbps": self.min_bandwidth_mbps,
                "min_uptime_hours": self.min_uptime_hours,
            },
        }

    def configure_security_policies(
        self,
        blocked_countries: Optional[Set[str]] = None,
        required_flags: Optional[Set[str]] = None,
        min_bandwidth_mbps: Optional[float] = None,
    ) -> None:
        """Configure security policies"""
        if blocked_countries is not None:
            self.blocked_countries = blocked_countries

        if required_flags is not None:
            self.required_flags = required_flags

        if min_bandwidth_mbps is not None:
            self.min_bandwidth_mbps = min_bandwidth_mbps

        logger.info("Security policies updated")

        # Emit policy update event
        self._emit_security_event(
            "security_policies_updated",
            {
                "blocked_countries": list(self.blocked_countries),
                "required_flags": list(self.required_flags),
                "min_bandwidth_mbps": self.min_bandwidth_mbps,
            },
        )

    def add_mixnode(self, node_info: MixnodeInfo) -> None:
        """Add new mixnode to available pool"""
        self.available_mixnodes[node_info.node_id] = node_info
        self.mixnode_networks[node_info.network].add(node_info.node_id)

        logger.info(f"Added mixnode {node_info.node_id} ({node_info.network})")

    def update_mixnode_performance(
        self, node_id: str, latency_ms: float, bandwidth_mbps: float, success_rate: float
    ) -> None:
        """Update mixnode performance metrics"""
        if node_id in self.available_mixnodes:
            node_info = self.available_mixnodes[node_id]

            # Update with exponential moving average
            alpha = 0.1
            node_info.latency_ms = alpha * latency_ms + (1 - alpha) * node_info.latency_ms
            node_info.bandwidth_mbps = alpha * bandwidth_mbps + (1 - alpha) * node_info.bandwidth_mbps
            node_info.reliability_score = alpha * success_rate + (1 - alpha) * node_info.reliability_score
            node_info.last_seen = time.time()

            logger.debug(f"Updated performance for mixnode {node_id}")

    def track_circuit_usage(self, circuit_id: str, bytes_transferred: int) -> None:
        """Track circuit usage for analytics"""
        if circuit_id in self.active_circuits:
            circuit = self.active_circuits[circuit_id]
            circuit.bytes_transferred += bytes_transferred
            circuit.last_activity = time.time()

            # Update global metrics
            self.privacy_metrics["bytes_through_mixnodes"] += bytes_transferred
