"""
BetaNet Mixnet Privacy Layer - Consolidated v1.1 Implementation

Consolidates mixnet privacy features from deprecated files:
- Privacy modes (strict/balanced/performance) with VRF-based hop selection
- BeaconSet entropy and constant-rate padding for indistinguishability
- AS diversity routing and trust scoring
- Cover traffic generation and timing obfuscation
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import logging
import random
import secrets
import struct
import time
from typing import Any

logger = logging.getLogger(__name__)


class PrivacyMode(Enum):
    """BetaNet privacy modes from v1.1 spec"""

    STRICT = "strict"  # 5+ hops, max diversity, constant padding
    BALANCED = "balanced"  # 3+ hops, good diversity, adaptive padding
    PERFORMANCE = "performance"  # 2+ hops, basic diversity, minimal padding


@dataclass
class BeaconSetEntry:
    """BeaconSet entry for epoch-based entropy (from deprecated mixnet.py)"""

    epoch: int
    beacon_id: str
    as_group: str
    trust_score: float
    pubkey: bytes
    vrf_proof: bytes
    timestamp: float


@dataclass
class MixHop:
    """Mix network hop configuration with AS diversity tracking"""

    node_id: str
    as_group: str
    trust_score: float
    latency_ms: float
    capacity_ratio: float
    public_key: bytes
    endpoint: str
    geolocation: str = ""
    bandwidth_mbps: float = 100.0


@dataclass
class MixRoute:
    """Complete mix network route with privacy metrics"""

    hops: list[MixHop]
    total_latency: float
    min_trust: float
    as_diversity: int
    route_id: bytes
    created_at: float
    variant_count: int = 0
    privacy_score: float = 0.0


class VRFSelector:
    """VRF-based hop selection for privacy routing (consolidated)"""

    def __init__(self, seed_key: bytes | None = None):
        self.seed_key = seed_key or secrets.token_bytes(32)
        self.route_history: list[bytes] = []
        self.max_history = 8  # Avoid reusing hop sets within 8 variants

    def generate_vrf_proof(self, input_data: bytes) -> tuple[bytes, bytes]:
        """Generate VRF proof and output"""
        # Create VRF input hash
        vrf_input = hashlib.sha256(input_data + self.seed_key).digest()

        # Generate proof using HMAC (production would use proper VRF)
        proof = hmac.new(self.seed_key, vrf_input, hashlib.sha256).digest()

        # Generate output using different key derivation
        output_key = hashlib.sha256(self.seed_key + b"output").digest()
        vrf_output = hmac.new(output_key, vrf_input, hashlib.sha256).digest()

        return proof, vrf_output

    def select_hops(
        self, available_nodes: list[MixHop], privacy_mode: PrivacyMode, target_hops: int | None = None
    ) -> list[MixHop]:
        """Select optimal hops using VRF and privacy constraints"""

        # Determine hop count based on privacy mode
        if target_hops is None:
            hop_counts = {PrivacyMode.STRICT: 5, PrivacyMode.BALANCED: 3, PrivacyMode.PERFORMANCE: 2}
            target_hops = hop_counts[privacy_mode]

        if len(available_nodes) < target_hops:
            logger.warning(f"Insufficient nodes: need {target_hops}, have {len(available_nodes)}")
            target_hops = len(available_nodes)

        # Generate VRF output for deterministic selection
        route_seed = secrets.token_bytes(32)
        vrf_proof, vrf_output = self.generate_vrf_proof(route_seed)

        # Convert VRF output to node selection weights
        selected_hops = []
        remaining_nodes = available_nodes.copy()
        used_as_groups = set()

        for i in range(target_hops):
            # Calculate selection weights based on VRF, trust, and diversity
            weights = []
            for node in remaining_nodes:
                # Base weight from VRF output
                node_seed = vrf_output[i % len(vrf_output) :] + node.node_id.encode()
                weight = struct.unpack(">I", hashlib.sha256(node_seed).digest()[:4])[0]

                # Boost weight for trust score
                weight = int(weight * node.trust_score)

                # Boost weight for AS diversity (avoid same AS group)
                if privacy_mode != PrivacyMode.PERFORMANCE and node.as_group in used_as_groups:
                    weight = int(weight * 0.1)  # Heavy penalty for AS reuse

                weights.append(weight)

            # Select node based on weighted random choice
            total_weight = sum(weights)
            if total_weight == 0:
                # Fallback to random selection
                selected_node = random.choice(remaining_nodes)
            else:
                selection_point = random.randint(0, total_weight - 1)
                cumulative = 0
                selected_node = remaining_nodes[0]  # fallback

                for j, weight in enumerate(weights):
                    cumulative += weight
                    if cumulative > selection_point:
                        selected_node = remaining_nodes[j]
                        break

            selected_hops.append(selected_node)
            remaining_nodes.remove(selected_node)
            used_as_groups.add(selected_node.as_group)

        return selected_hops


class ConstantRatePadding:
    """Constant-rate padding for traffic analysis resistance"""

    def __init__(self, target_rate_bps: int = 8192):
        self.target_rate_bps = target_rate_bps
        self.padding_task: asyncio.Task | None = None
        self.stats = {"real_bytes": 0, "padding_bytes": 0, "total_bytes": 0, "sessions": 0}

    async def start_padding(self, send_callback):
        """Start constant-rate padding generation"""
        if self.padding_task:
            return

        self.padding_task = asyncio.create_task(self._padding_loop(send_callback))
        logger.info(f"Started constant-rate padding at {self.target_rate_bps} bps")

    async def stop_padding(self):
        """Stop padding generation"""
        if self.padding_task:
            self.padding_task.cancel()
            try:
                await self.padding_task
            except asyncio.CancelledError:
                pass
            self.padding_task = None
            logger.info("Stopped constant-rate padding")

    async def _padding_loop(self, send_callback):
        """Generate constant-rate padding traffic"""
        interval_ms = 1000  # 1 second intervals
        bytes_per_interval = self.target_rate_bps // 8  # bits to bytes

        while True:
            try:
                # Generate random padding
                padding_data = secrets.token_bytes(random.randint(bytes_per_interval // 2, bytes_per_interval))

                # Send as padding frame
                await send_callback(padding_data, is_padding=True)
                self.stats["padding_bytes"] += len(padding_data)

                await asyncio.sleep(interval_ms / 1000.0)

            except Exception as e:
                logger.error(f"Padding generation error: {e}")
                await asyncio.sleep(5)


class BetaNetPrivacyManager:
    """Unified privacy manager consolidating all mixnet features"""

    def __init__(self, privacy_mode: PrivacyMode = PrivacyMode.BALANCED):
        self.privacy_mode = privacy_mode
        self.vrf_selector = VRFSelector()
        self.padding_manager = ConstantRatePadding()
        self.active_circuits = {}
        self.beacon_set = {}

    async def initialize_privacy_layer(self, mixnodes: list[dict[str, Any]], enable_padding: bool = True):
        """Initialize privacy layer with mixnode discovery"""

        # Convert dict configs to MixHop objects
        mix_hops = []
        for node_config in mixnodes:
            hop = MixHop(
                node_id=node_config["id"],
                as_group=node_config.get("as_group", "unknown"),
                trust_score=node_config.get("trust_score", 0.5),
                latency_ms=node_config.get("latency_ms", 100.0),
                capacity_ratio=node_config.get("capacity_ratio", 0.8),
                public_key=node_config.get("public_key", b"placeholder"),
                endpoint=node_config["endpoint"],
                geolocation=node_config.get("geolocation", ""),
                bandwidth_mbps=node_config.get("bandwidth_mbps", 100.0),
            )
            mix_hops.append(hop)

        self.available_mixnodes = mix_hops

        # Start padding if enabled
        if enable_padding:
            padding_rates = {
                PrivacyMode.STRICT: 16384,  # 16 Kbps constant padding
                PrivacyMode.BALANCED: 8192,  # 8 Kbps adaptive padding
                PrivacyMode.PERFORMANCE: 4096,  # 4 Kbps minimal padding
            }

            self.padding_manager.target_rate_bps = padding_rates[self.privacy_mode]

        logger.info(f"Privacy layer initialized: {len(mix_hops)} mixnodes, mode={self.privacy_mode.value}")

    async def create_private_route(self, target_endpoint: str) -> str:
        """Create privacy-optimized route to target"""

        # Select hops based on privacy mode
        selected_hops = self.vrf_selector.select_hops(self.available_mixnodes, self.privacy_mode)

        # Calculate privacy metrics
        as_groups = set(hop.as_group for hop in selected_hops)
        min_trust = min(hop.trust_score for hop in selected_hops) if selected_hops else 0.0
        total_latency = sum(hop.latency_ms for hop in selected_hops)

        # Create route
        route = MixRoute(
            hops=selected_hops,
            total_latency=total_latency,
            min_trust=min_trust,
            as_diversity=len(as_groups),
            route_id=secrets.token_bytes(16),
            created_at=time.time(),
            privacy_score=self._calculate_privacy_score(selected_hops),
        )

        route_id = route.route_id.hex()
        self.active_circuits[route_id] = route

        logger.info(
            f"Created private route {route_id[:8]}: {len(selected_hops)} hops, "
            f"AS diversity={route.as_diversity}, privacy_score={route.privacy_score:.2f}"
        )

        return route_id

    def _calculate_privacy_score(self, hops: list[MixHop]) -> float:
        """Calculate overall privacy score for route"""
        if not hops:
            return 0.0

        # Factors: hop count, AS diversity, trust, latency
        hop_score = min(1.0, len(hops) / 5.0)  # Normalized to 5 hops max
        as_groups = set(hop.as_group for hop in hops)
        diversity_score = min(1.0, len(as_groups) / len(hops))  # Perfect = 1.0
        trust_score = min(hop.trust_score for hop in hops)

        # Combine factors
        privacy_score = hop_score * 0.4 + diversity_score * 0.3 + trust_score * 0.3

        return privacy_score

    async def route_message(self, route_id: str, message_data: bytes, target_endpoint: str) -> bool:
        """Route message through private mixnet circuit"""

        if route_id not in self.active_circuits:
            logger.error(f"Route {route_id[:8]} not found")
            return False

        route = self.active_circuits[route_id]

        # Apply Sphinx layered encryption (simplified)
        encrypted_data = message_data
        for hop in reversed(route.hops):
            encrypted_data = self._apply_sphinx_layer(encrypted_data, hop)

        # Send to first hop
        success = await self._send_to_mixnode(route.hops[0].endpoint, encrypted_data)

        if success:
            route.variant_count += 1
            logger.debug(f"Message routed via {route_id[:8]}, variant #{route.variant_count}")

        return success

    def _apply_sphinx_layer(self, data: bytes, hop: MixHop) -> bytes:
        """Apply Sphinx encryption layer for hop"""
        # Simplified Sphinx layer (production would use proper cryptography)
        layer_key = hashlib.sha256(hop.public_key + hop.node_id.encode()).digest()

        # Create Sphinx header with routing info
        sphinx_header = struct.pack(">I", len(data)) + hop.endpoint.encode()[:32].ljust(32, b"\x00")

        # Simple XOR encryption (production would use ChaCha20)
        encrypted_payload = bytes(a ^ b for a, b in zip(data, (layer_key * ((len(data) + 31) // 32))[: len(data)]))

        return sphinx_header + encrypted_payload

    async def _send_to_mixnode(self, endpoint: str, encrypted_data: bytes) -> bool:
        """Send encrypted packet to mixnode endpoint"""
        try:
            # Would implement actual network send to mixnode
            logger.debug(f"Sending {len(encrypted_data)} bytes to mixnode {endpoint}")

            # Simulate network delay based on realistic mixnode latency
            await asyncio.sleep(random.uniform(0.05, 0.2))

            return True
        except Exception as e:
            logger.error(f"Failed to send to mixnode {endpoint}: {e}")
            return False


class BetaNetCoverTraffic:
    """Cover traffic generation for traffic analysis resistance"""

    def __init__(self, privacy_mode: PrivacyMode):
        self.privacy_mode = privacy_mode
        self.cover_task: asyncio.Task | None = None
        self.stats = {"cover_messages": 0, "cover_bytes": 0, "real_messages": 0, "real_bytes": 0}

    async def start_cover_traffic(self, send_callback):
        """Start cover traffic generation based on privacy mode"""
        if self.cover_task:
            return

        # Configure cover traffic based on privacy mode
        intervals = {
            PrivacyMode.STRICT: 0.5,  # 2 msgs/sec constant rate
            PrivacyMode.BALANCED: 1.0,  # 1 msg/sec adaptive
            PrivacyMode.PERFORMANCE: 2.0,  # 0.5 msgs/sec minimal
        }

        interval_sec = intervals[self.privacy_mode]
        self.cover_task = asyncio.create_task(self._cover_traffic_loop(send_callback, interval_sec))

        logger.info(f"Started cover traffic: mode={self.privacy_mode.value}, " f"interval={interval_sec}s")

    async def stop_cover_traffic(self):
        """Stop cover traffic generation"""
        if self.cover_task:
            self.cover_task.cancel()
            try:
                await self.cover_task
            except asyncio.CancelledError:
                pass
            self.cover_task = None
            logger.info("Stopped cover traffic generation")

    async def _cover_traffic_loop(self, send_callback, interval_sec: float):
        """Generate cover traffic at specified intervals"""
        while True:
            try:
                # Generate realistic cover message
                cover_sizes = [512, 1024, 2048, 4096, 8192]  # Common web object sizes
                cover_size = random.choice(cover_sizes)
                cover_data = secrets.token_bytes(cover_size)

                # Send as cover traffic
                await send_callback(cover_data, is_cover=True)

                self.stats["cover_messages"] += 1
                self.stats["cover_bytes"] += len(cover_data)

                # Add jitter to prevent detectability
                jitter = random.uniform(-0.1, 0.1) * interval_sec
                await asyncio.sleep(interval_sec + jitter)

            except Exception as e:
                logger.error(f"Cover traffic generation error: {e}")
                await asyncio.sleep(5)  # Back off on errors


class ConsolidatedBetaNetMixnet:
    """
    Consolidated BetaNet mixnet implementation combining all advanced features
    from deprecated transport files
    """

    def __init__(
        self,
        privacy_mode: PrivacyMode = PrivacyMode.BALANCED,
        enable_cover_traffic: bool = True,
        enable_padding: bool = True,
    ):
        self.privacy_mode = privacy_mode
        self.privacy_manager = BetaNetPrivacyManager(privacy_mode)
        self.cover_traffic = BetaNetCoverTraffic(privacy_mode) if enable_cover_traffic else None
        self.active_routes = {}
        self.message_queue = asyncio.Queue()

    async def initialize(self, mixnode_configs: list[dict[str, Any]]):
        """Initialize mixnet with node configurations"""
        await self.privacy_manager.initialize_privacy_layer(mixnode_configs, enable_padding=True)

        # Start cover traffic if enabled
        if self.cover_traffic:
            await self.cover_traffic.start_cover_traffic(self._send_cover_message)

        logger.info(
            f"BetaNet mixnet initialized: {len(mixnode_configs)} nodes, " f"privacy_mode={self.privacy_mode.value}"
        )

    async def send_anonymous_message(self, message_data: bytes, target_endpoint: str) -> bool:
        """Send message through anonymous mixnet route"""

        # Create or reuse private route
        route_id = await self.privacy_manager.create_private_route(target_endpoint)

        # Route message through mixnet
        success = await self.privacy_manager.route_message(route_id, message_data, target_endpoint)

        if success:
            # Update stats
            if self.cover_traffic:
                self.cover_traffic.stats["real_messages"] += 1
                self.cover_traffic.stats["real_bytes"] += len(message_data)

        return success

    async def _send_cover_message(self, cover_data: bytes, is_cover: bool = True):
        """Send cover traffic message"""
        # Would implement actual cover message sending
        logger.debug(f"Generated cover message: {len(cover_data)} bytes")

    def get_privacy_stats(self) -> dict[str, Any]:
        """Get comprehensive privacy and performance statistics"""
        stats = {
            "privacy_mode": self.privacy_mode.value,
            "active_routes": len(self.active_routes),
            "total_circuits": len(self.privacy_manager.active_circuits),
        }

        if self.cover_traffic:
            stats.update(self.cover_traffic.stats)

        return stats

    async def cleanup(self):
        """Clean up all resources"""
        if self.cover_traffic:
            await self.cover_traffic.stop_cover_traffic()
        await self.privacy_manager.padding_manager.stop_padding()

        logger.info("BetaNet mixnet cleanup complete")


# Factory function for easy integration with existing AIVillage infrastructure
def create_betanet_mixnet(
    privacy_mode: str = "balanced", mixnode_endpoints: list[str] | None = None
) -> ConsolidatedBetaNetMixnet:
    """
    Create consolidated BetaNet mixnet with all advanced features

    Args:
        privacy_mode: "strict", "balanced", or "performance"
        mixnode_endpoints: List of mixnode URLs

    Returns:
        Fully configured mixnet instance
    """
    mode = PrivacyMode(privacy_mode)
    mixnet = ConsolidatedBetaNetMixnet(privacy_mode=mode)

    # Convert endpoint strings to node configs if provided
    if mixnode_endpoints:
        node_configs = []
        for i, endpoint in enumerate(mixnode_endpoints):
            config = {
                "id": f"mixnode_{i}",
                "endpoint": endpoint,
                "as_group": f"AS{hash(endpoint) % 1000}",  # Pseudo AS assignment
                "trust_score": 0.8,  # Default trust
                "latency_ms": 150.0,
                "capacity_ratio": 0.9,
                "public_key": hashlib.sha256(endpoint.encode()).digest(),
            }
            node_configs.append(config)

        # Initialize asynchronously (would need to be awaited)
        logger.info(f"Created mixnet config for {len(node_configs)} nodes")

    return mixnet
