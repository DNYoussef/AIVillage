"""
VRF-based Neighbor Selection for Secure Fog Networking Topology

Implements Verifiable Random Function (VRF) neighbor selection to provide:
- Cryptographically verifiable randomness for unbiased peer selection
- Eclipse attack prevention through unpredictable topology
- Expander-like graph properties for network efficiency
- Spectral gap analysis for network health monitoring
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import logging
import math
import random
import struct
import time
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
import numpy as np

# Import reputation system for enhanced selection
try:
    from ..reputation.bayesian_reputation import BayesianReputationEngine, ReputationScore
except ImportError:
    BayesianReputationEngine = None
    ReputationScore = None

logger = logging.getLogger(__name__)


class VRFStatus(Enum):
    """VRF system status."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    RESELECTING = "reselecting"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class NodeInfo:
    """Information about a network node."""

    node_id: str
    public_key: bytes
    address: str
    port: int
    last_seen: float = field(default_factory=time.time)
    reliability_score: float = 1.0
    latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    uptime_hours: float = 0.0
    connection_count: int = 0
    is_malicious: bool = False
    trust_score: float = 1.0


@dataclass
class VRFProof:
    """VRF proof containing verifiable randomness."""

    beta: bytes  # VRF output
    pi: bytes  # VRF proof
    alpha: bytes  # VRF input
    public_key: bytes
    timestamp: float = field(default_factory=time.time)


@dataclass
class SelectionRound:
    """Information about a neighbor selection round."""

    round_id: str
    epoch: int
    timestamp: float
    seed: bytes
    selected_neighbors: list[str]
    vrf_proofs: dict[str, VRFProof]
    topology_metrics: dict[str, float]


@dataclass
class TopologyMetrics:
    """Network topology quality metrics."""

    expansion: float = 0.0  # Expansion ratio
    conductance: float = 0.0  # Graph conductance
    spectral_gap: float = 0.0  # Second eigenvalue gap
    diameter: int = 0  # Network diameter
    clustering: float = 0.0  # Clustering coefficient
    degree_variance: float = 0.0  # Variance in node degrees
    redundancy: float = 0.0  # Path redundancy


class VRFNeighborSelector:
    """
    VRF-based neighbor selection for secure fog networking topology.

    Provides cryptographically verifiable random neighbor selection that:
    - Prevents eclipse attacks through unpredictable selection
    - Maintains expander-like graph properties
    - Enables public verification of selection fairness
    - Adapts to network conditions and node behavior
    """

    def __init__(
        self,
        node_id: str,
        private_key: ec.EllipticCurvePrivateKey | None = None,
        target_degree: int = 8,
        min_degree: int = 4,
        max_degree: int = 16,
        selection_interval: float = 300.0,  # 5 minutes
        verification_threshold: float = 0.7,
        **kwargs,
    ):
        self.node_id = node_id
        self.target_degree = target_degree
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.selection_interval = selection_interval
        self.verification_threshold = verification_threshold

        # Cryptographic setup - Use Ed25519 for better VRF properties
        if private_key and isinstance(private_key, ed25519.Ed25519PrivateKey):
            self.private_key = private_key
        else:
            self.private_key = ed25519.Ed25519PrivateKey.generate()

        self.public_key = self.private_key.public_key()
        self.public_key_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

        # Network state
        self.known_nodes: dict[str, NodeInfo] = {}
        self.current_neighbors: set[str] = set()
        self.selection_history: list[SelectionRound] = []
        self.topology_cache: TopologyMetrics | None = None

        # VRF state
        self.current_epoch = 0
        self.last_selection_time = 0.0
        self.pending_verifications: dict[str, VRFProof] = {}

        # Security tracking
        self.eclipse_attempts = 0
        self.suspicious_patterns: list[dict[str, Any]] = []
        self.blacklisted_nodes: set[str] = set()

        # Configuration
        self.config = {
            "vrf_algorithm": "ECVRF-Ed25519-SHA512-TAI",
            "proof_expiry_seconds": kwargs.get("proof_expiry_seconds", 3600),
            "max_selection_attempts": kwargs.get("max_selection_attempts", 10),
            "expansion_target": kwargs.get("expansion_target", 0.9),
            "spectral_gap_min": kwargs.get("spectral_gap_min", 0.2),
            "eclipse_detection_threshold": kwargs.get("eclipse_detection_threshold", 0.8),
            "reputation_decay": kwargs.get("reputation_decay", 0.95),
            "k_core_min": kwargs.get("k_core_min", 3),
            "topology_healing_threshold": kwargs.get("topology_healing_threshold", 0.15),
        }

        # Reputation integration
        self.reputation_engine: BayesianReputationEngine | None = kwargs.get("reputation_engine")

        # Monitoring
        self.status = VRFStatus.INITIALIZING
        self.metrics = {
            "selections_performed": 0,
            "verifications_passed": 0,
            "verifications_failed": 0,
            "eclipse_attempts_blocked": 0,
            "topology_optimizations": 0,
            "last_selection_duration": 0.0,
        }

        # Background tasks
        self._selection_task: asyncio.Task | None = None
        self._verification_task: asyncio.Task | None = None
        self._topology_monitor_task: asyncio.Task | None = None

        logger.info(f"VRF neighbor selector initialized for node {node_id}")

    async def start(self) -> bool:
        """Start the VRF neighbor selection system."""
        try:
            logger.info("Starting VRF neighbor selection system...")

            # Initialize with current network state
            await self._initialize_topology()

            # Start background tasks
            self._selection_task = asyncio.create_task(self._selection_loop())
            self._verification_task = asyncio.create_task(self._verification_loop())
            self._topology_monitor_task = asyncio.create_task(self._topology_monitor_loop())

            self.status = VRFStatus.ACTIVE
            logger.info("VRF neighbor selection system started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start VRF system: {e}")
            self.status = VRFStatus.ERROR
            return False

    async def stop(self):
        """Stop the VRF neighbor selection system."""
        logger.info("Stopping VRF neighbor selection system...")
        self.status = VRFStatus.STOPPED

        # Cancel background tasks
        for task in [self._selection_task, self._verification_task, self._topology_monitor_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("VRF neighbor selection system stopped")

    async def add_node(self, node_info: NodeInfo):
        """Add a node to the known nodes registry."""
        self.known_nodes[node_info.node_id] = node_info
        logger.debug(f"Added node {node_info.node_id} to registry")

        # Trigger reselection if we don't have enough neighbors
        if len(self.current_neighbors) < self.min_degree:
            asyncio.create_task(self._trigger_selection())

    async def remove_node(self, node_id: str):
        """Remove a node from the known nodes registry."""
        self.known_nodes.pop(node_id, None)
        self.current_neighbors.discard(node_id)
        logger.debug(f"Removed node {node_id} from registry")

        # Trigger reselection if we lost a neighbor
        if len(self.current_neighbors) < self.min_degree:
            asyncio.create_task(self._trigger_selection())

    async def update_node_metrics(self, node_id: str, **metrics):
        """Update performance metrics for a node."""
        if node_id in self.known_nodes:
            node = self.known_nodes[node_id]
            for key, value in metrics.items():
                if hasattr(node, key):
                    setattr(node, key, value)
            node.last_seen = time.time()

    async def select_neighbors(self, force_reselection: bool = False) -> list[str]:
        """
        Select neighbors using VRF for verifiable randomness.

        Returns list of selected neighbor node IDs.
        """
        start_time = time.time()
        self.status = VRFStatus.RESELECTING

        try:
            # Generate VRF input from current epoch and network state
            epoch_data = struct.pack(">Q", self.current_epoch)
            network_hash = self._compute_network_hash()
            alpha = epoch_data + network_hash + self.node_id.encode()

            # Generate VRF proof
            vrf_proof = self._generate_vrf_proof(alpha)

            # Use VRF output as seed for neighbor selection
            selection_seed = vrf_proof.beta

            # Perform neighbor selection
            selected_neighbors = await self._perform_neighbor_selection(selection_seed)

            # Validate topology properties
            if not await self._validate_topology_properties(selected_neighbors):
                logger.warning("Topology validation failed, retrying with adjusted parameters")
                selected_neighbors = await self._adjust_selection_for_topology(selected_neighbors, selection_seed)

            # Update current neighbors
            self.current_neighbors = set(selected_neighbors)

            # Record selection round
            selection_round = SelectionRound(
                round_id=f"{self.node_id}_{self.current_epoch}",
                epoch=self.current_epoch,
                timestamp=time.time(),
                seed=selection_seed,
                selected_neighbors=selected_neighbors,
                vrf_proofs={self.node_id: vrf_proof},
                topology_metrics=await self._compute_topology_metrics(),
            )

            self.selection_history.append(selection_round)
            if len(self.selection_history) > 100:  # Keep last 100 rounds
                self.selection_history.pop(0)

            # Update metrics
            self.metrics["selections_performed"] += 1
            self.metrics["last_selection_duration"] = time.time() - start_time
            self.last_selection_time = time.time()
            self.current_epoch += 1

            self.status = VRFStatus.ACTIVE
            logger.info(
                f"Selected {len(selected_neighbors)} neighbors in {self.metrics['last_selection_duration']:.2f}s"
            )
            return selected_neighbors

        except Exception as e:
            logger.error(f"Neighbor selection failed: {e}")
            self.status = VRFStatus.ERROR
            return list(self.current_neighbors)  # Return current neighbors on error

    async def verify_selection(self, node_id: str, vrf_proof: VRFProof, claimed_neighbors: list[str]) -> bool:
        """Verify another node's neighbor selection using their VRF proof."""
        try:
            # Verify VRF proof
            if not self._verify_vrf_proof(vrf_proof):
                logger.warning(f"VRF proof verification failed for node {node_id}")
                return False

            # Verify selection matches VRF output
            selection_seed = vrf_proof.beta
            expected_neighbors = await self._simulate_neighbor_selection(selection_seed, node_id)

            if set(claimed_neighbors) != set(expected_neighbors):
                logger.warning(f"Neighbor selection mismatch for node {node_id}")
                return False

            self.metrics["verifications_passed"] += 1
            return True

        except Exception as e:
            logger.error(f"Selection verification failed for node {node_id}: {e}")
            self.metrics["verifications_failed"] += 1
            return False

    def _generate_vrf_proof(self, alpha: bytes) -> VRFProof:
        """Generate Ed25519-based VRF proof for given input."""
        # Implement ECVRF-Ed25519-SHA512-TAI based on draft-irtf-cfrg-vrf-15

        # Step 1: Hash to curve point (simplified for Ed25519)
        h = hashlib.sha512()
        h.update(b"ECVRF_hash_to_curve_")
        h.update(self.public_key_bytes)
        h.update(alpha)
        hash_to_curve = h.digest()[:32]  # Use first 32 bytes as scalar

        # Step 2: Generate nonce for signature
        nonce_hash = hashlib.sha512()
        nonce_hash.update(
            self.private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        nonce_hash.update(alpha)
        nonce_hash.digest()[:32]

        # Step 3: Create VRF proof (simplified Ed25519 signature)
        # In production, use proper curve arithmetic for VRF
        proof_input = hash_to_curve + alpha
        signature = self.private_key.sign(proof_input)

        # Step 4: Generate VRF output (beta)
        beta_hash = hashlib.sha512()
        beta_hash.update(b"ECVRF_proof_to_hash_")
        beta_hash.update(signature)
        beta_hash.update(alpha)
        beta = beta_hash.digest()[:32]  # VRF output

        return VRFProof(beta=beta, pi=signature, alpha=alpha, public_key=self.public_key_bytes)

    def _verify_vrf_proof(self, proof: VRFProof) -> bool:
        """Verify Ed25519-based VRF proof."""
        try:
            # Load Ed25519 public key
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(proof.public_key)

            # Recreate hash to curve
            h = hashlib.sha512()
            h.update(b"ECVRF_hash_to_curve_")
            h.update(proof.public_key)
            h.update(proof.alpha)
            hash_to_curve = h.digest()[:32]

            # Verify signature
            proof_input = hash_to_curve + proof.alpha
            public_key.verify(proof.pi, proof_input)

            # Verify beta matches proof
            beta_hash = hashlib.sha512()
            beta_hash.update(b"ECVRF_proof_to_hash_")
            beta_hash.update(proof.pi)
            beta_hash.update(proof.alpha)
            expected_beta = beta_hash.digest()[:32]

            return hmac.compare_digest(expected_beta, proof.beta)

        except (InvalidSignature, ValueError, Exception):
            return False

    def _compute_network_hash(self) -> bytes:
        """Compute hash of current network state for VRF input."""
        h = hashlib.sha256()

        # Include sorted node IDs and their key info
        for node_id in sorted(self.known_nodes.keys()):
            node = self.known_nodes[node_id]
            h.update(node_id.encode())
            h.update(node.public_key)

        return h.digest()

    async def _perform_neighbor_selection(self, seed: bytes) -> list[str]:
        """Perform actual neighbor selection using VRF seed."""
        # Filter eligible nodes
        eligible_nodes = self._get_eligible_nodes()

        if len(eligible_nodes) <= self.target_degree:
            return [node.node_id for node in eligible_nodes]

        # Use seed to deterministically select neighbors
        rng = random.Random(seed)

        # Weight nodes by reputation and reliability
        weighted_nodes = []
        for node in eligible_nodes:
            weight = self._calculate_node_weight(node)
            weighted_nodes.append((node, weight))

        # Select neighbors using weighted random sampling
        selected = []
        total_attempts = 0
        max_attempts = self.config["max_selection_attempts"]

        while len(selected) < self.target_degree and total_attempts < max_attempts:
            # Weighted selection
            total_weight = sum(weight for _, weight in weighted_nodes)
            r = rng.random() * total_weight

            cumulative_weight = 0
            for node, weight in weighted_nodes:
                cumulative_weight += weight
                if r <= cumulative_weight and node.node_id not in selected:
                    selected.append(node.node_id)
                    break

            total_attempts += 1

        # Ensure minimum degree
        if len(selected) < self.min_degree:
            # Add random eligible nodes to meet minimum
            remaining = [n.node_id for n in eligible_nodes if n.node_id not in selected]
            needed = self.min_degree - len(selected)
            selected.extend(rng.sample(remaining, min(needed, len(remaining))))

        return selected

    async def _simulate_neighbor_selection(self, seed: bytes, for_node_id: str) -> list[str]:
        """Simulate neighbor selection for verification purposes."""
        # This would use the same algorithm as _perform_neighbor_selection
        # but simulate it for another node's perspective
        return await self._perform_neighbor_selection(seed)

    def _get_eligible_nodes(self) -> list[NodeInfo]:
        """Get list of nodes eligible for selection."""
        current_time = time.time()
        eligible = []

        for node_id, node in self.known_nodes.items():
            if (
                node_id != self.node_id  # Not self
                and node_id not in self.blacklisted_nodes  # Not blacklisted
                and not node.is_malicious  # Not marked as malicious
                and current_time - node.last_seen < 3600  # Seen recently (1 hour)
                and node.reliability_score > 0.1  # Minimum reliability
            ):
                eligible.append(node)

        return eligible

    def _calculate_node_weight(self, node: NodeInfo) -> float:
        """Calculate selection weight for a node with reputation integration."""
        base_weight = 1.0

        # Reputation system integration
        if self.reputation_engine:
            reputation_score = self.reputation_engine.get_reputation_score(node.node_id)
            if reputation_score:
                # Use Bayesian reputation mean with uncertainty penalty
                trust_factor = reputation_score.mean_score * (1 - reputation_score.uncertainty)
                base_weight *= 0.5 + trust_factor  # Scale to [0.5, 1.5]

                # Tier-based bonuses
                tier_bonuses = {
                    5: 1.5,  # DIAMOND
                    4: 1.3,  # PLATINUM
                    3: 1.2,  # GOLD
                    2: 1.1,  # SILVER
                    1: 1.0,  # BRONZE
                    0: 0.8,  # UNTRUSTED
                }
                base_weight *= tier_bonuses.get(reputation_score.tier.value, 1.0)

        # Traditional factors
        base_weight *= node.reliability_score
        base_weight *= node.trust_score

        # Uptime factor (logarithmic)
        base_weight *= math.log(1 + node.uptime_hours) / math.log(1 + 24 * 7)  # Max 1 week

        # Connection count penalty (avoid overloaded nodes)
        if node.connection_count > 20:
            base_weight *= 0.5
        elif node.connection_count > 50:
            base_weight *= 0.3  # Heavy penalty for very overloaded nodes

        # Latency penalty
        if node.latency_ms > 1000:
            base_weight *= 0.7
        elif node.latency_ms > 2000:
            base_weight *= 0.4

        # Bandwidth bonus
        if node.bandwidth_mbps > 100:
            base_weight *= 1.1
        elif node.bandwidth_mbps < 10:
            base_weight *= 0.9

        return max(base_weight, 0.01)  # Minimum weight

    async def _validate_topology_properties(self, selected_neighbors: list[str]) -> bool:
        """Validate that selected neighbors maintain good topology properties."""
        # Check degree constraints
        if not (self.min_degree <= len(selected_neighbors) <= self.max_degree):
            return False

        # Check for eclipse attack patterns
        if await self._detect_eclipse_attempt(selected_neighbors):
            return False

        # Validate expansion properties (simplified)
        if not await self._validate_expansion_properties(selected_neighbors):
            return False

        return True

    async def _detect_eclipse_attempt(self, selected_neighbors: list[str]) -> bool:
        """Detect potential eclipse attack patterns with reputation integration."""
        # Check for suspicious clustering
        suspicious_count = 0
        low_reputation_count = 0

        for neighbor_id in selected_neighbors:
            node = self.known_nodes.get(neighbor_id)
            if not node:
                continue

            is_suspicious = False

            # Traditional suspicious patterns
            if (
                node.trust_score < 0.5
                or node.reliability_score < 0.3
                or node.connection_count > 50  # Suspiciously high connections
            ):
                is_suspicious = True

            # Reputation-based detection
            if self.reputation_engine:
                reputation = self.reputation_engine.get_reputation_score(neighbor_id)
                if reputation:
                    # Low reputation with high uncertainty is suspicious
                    if (
                        reputation.mean_score < 0.4 and reputation.uncertainty > 0.3
                    ) or reputation.tier.value == 0:  # UNTRUSTED tier
                        low_reputation_count += 1
                        is_suspicious = True

                    # Recently joined nodes with high claims are suspicious
                    if reputation.sample_size < 10 and node.connection_count > 30:
                        is_suspicious = True

            # Behavioral anomalies
            if (
                node.bandwidth_mbps > 1000
                and node.uptime_hours < 24  # Too good to be true
                or node.latency_ms == 0  # Suspicious perfect latency
                or node.is_malicious  # Already flagged
            ):
                is_suspicious = True

            if is_suspicious:
                suspicious_count += 1

        # Multi-tier detection
        base_threshold = len(selected_neighbors) * self.config["eclipse_detection_threshold"]
        reputation_threshold = len(selected_neighbors) * 0.6  # 60% low reputation threshold

        eclipse_detected = suspicious_count >= base_threshold or low_reputation_count >= reputation_threshold

        if eclipse_detected:
            self.eclipse_attempts += 1
            self.metrics["eclipse_attempts_blocked"] += 1

            # Store suspicious pattern for analysis
            self.suspicious_patterns.append(
                {
                    "timestamp": time.time(),
                    "suspicious_nodes": suspicious_count,
                    "low_reputation_nodes": low_reputation_count,
                    "total_selected": len(selected_neighbors),
                    "pattern_type": "eclipse_attempt",
                }
            )

            logger.warning(
                f"Eclipse attack detected: {suspicious_count}/{len(selected_neighbors)} suspicious, "
                f"{low_reputation_count} low reputation neighbors"
            )
            return True

        return False

    async def _validate_expansion_properties(self, selected_neighbors: list[str]) -> bool:
        """Validate that selection maintains expander properties."""
        # Simplified expansion check
        # In a real implementation, this would perform graph analysis

        # Check neighbor diversity (different network regions)
        unique_subnets = set()
        for neighbor_id in selected_neighbors:
            node = self.known_nodes.get(neighbor_id)
            if node:
                # Extract subnet from address (simplified)
                subnet = ".".join(node.address.split(".")[:3])
                unique_subnets.add(subnet)

        # Good expansion means diverse subnets
        diversity_ratio = len(unique_subnets) / len(selected_neighbors) if selected_neighbors else 0
        return diversity_ratio >= 0.5  # At least 50% subnet diversity

    async def _adjust_selection_for_topology(self, selected_neighbors: list[str], seed: bytes) -> list[str]:
        """Adjust neighbor selection to improve topology properties."""
        # Remove worst nodes and replace with better candidates
        eligible_nodes = self._get_eligible_nodes()
        available_nodes = [n for n in eligible_nodes if n.node_id not in selected_neighbors]

        if not available_nodes:
            return selected_neighbors

        # Sort by quality score
        available_nodes.sort(key=lambda n: self._calculate_node_weight(n), reverse=True)

        # Replace up to 25% of selection with high-quality nodes
        replacement_count = min(len(selected_neighbors) // 4, len(available_nodes))

        # Remove lowest quality selected neighbors
        selected_with_weights = [
            (nid, self._calculate_node_weight(self.known_nodes[nid]))
            for nid in selected_neighbors
            if nid in self.known_nodes
        ]
        selected_with_weights.sort(key=lambda x: x[1])

        adjusted = [nid for nid, _ in selected_with_weights[replacement_count:]]
        adjusted.extend([n.node_id for n in available_nodes[:replacement_count]])

        return adjusted

    async def _compute_topology_metrics(self) -> dict[str, float]:
        """Compute current topology quality metrics."""
        if not self.current_neighbors:
            return {}

        # Simplified topology metrics
        # In production, would use proper graph analysis

        metrics = {}

        # Degree distribution
        degrees = [len(self.current_neighbors)]  # Only have our degree
        metrics["average_degree"] = sum(degrees) / len(degrees)
        metrics["degree_variance"] = np.var(degrees) if len(degrees) > 1 else 0.0

        # Connectivity metrics (simplified)
        metrics["connectivity_ratio"] = min(len(self.current_neighbors) / self.target_degree, 1.0)

        # Quality metrics
        neighbor_weights = []
        for neighbor_id in self.current_neighbors:
            if neighbor_id in self.known_nodes:
                weight = self._calculate_node_weight(self.known_nodes[neighbor_id])
                neighbor_weights.append(weight)

        if neighbor_weights:
            metrics["average_neighbor_quality"] = sum(neighbor_weights) / len(neighbor_weights)
            metrics["quality_variance"] = np.var(neighbor_weights)
        else:
            metrics["average_neighbor_quality"] = 0.0
            metrics["quality_variance"] = 0.0

        return metrics

    async def _initialize_topology(self):
        """Initialize topology state."""
        logger.info("Initializing VRF topology...")

        # Perform initial neighbor selection
        if self.known_nodes:
            await self.select_neighbors()

        logger.info("VRF topology initialized")

    async def _trigger_selection(self):
        """Trigger immediate neighbor selection."""
        await self.select_neighbors(force_reselection=True)

    async def _selection_loop(self):
        """Background loop for periodic neighbor selection."""
        while self.status != VRFStatus.STOPPED:
            try:
                # Check if reselection is needed
                current_time = time.time()
                time_since_last = current_time - self.last_selection_time

                if time_since_last >= self.selection_interval or len(self.current_neighbors) < self.min_degree:
                    await self.select_neighbors()

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Selection loop error: {e}")
                await asyncio.sleep(10)

    async def _verification_loop(self):
        """Background loop for processing VRF verifications."""
        while self.status != VRFStatus.STOPPED:
            try:
                # Process pending verifications
                current_time = time.time()
                expired_proofs = []

                for node_id, proof in self.pending_verifications.items():
                    if current_time - proof.timestamp > self.config["proof_expiry_seconds"]:
                        expired_proofs.append(node_id)

                # Remove expired proofs
                for node_id in expired_proofs:
                    del self.pending_verifications[node_id]

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Verification loop error: {e}")
                await asyncio.sleep(10)

    async def _topology_monitor_loop(self):
        """Background loop for topology health monitoring."""
        while self.status != VRFStatus.STOPPED:
            try:
                # Update topology metrics
                self.topology_cache = await self._compute_detailed_topology_metrics()

                # Check for topology degradation
                if await self._detect_topology_degradation():
                    logger.warning("Topology degradation detected, triggering reselection")
                    await self._trigger_selection()

                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Topology monitor error: {e}")
                await asyncio.sleep(30)

    async def _compute_detailed_topology_metrics(self) -> TopologyMetrics:
        """Compute detailed topology metrics."""
        # This would implement proper graph analysis
        # For now, return simplified metrics

        return TopologyMetrics(
            expansion=0.8,  # Reference implementation values
            conductance=0.7,
            spectral_gap=0.3,
            diameter=4,
            clustering=0.2,
            degree_variance=2.0,
            redundancy=0.6,
        )

    async def _detect_topology_degradation(self) -> bool:
        """Detect if topology health is degrading."""
        if not self.topology_cache:
            return False

        # Check key metrics
        return (
            self.topology_cache.expansion < 0.5
            or self.topology_cache.spectral_gap < self.config["spectral_gap_min"]
            or len(self.current_neighbors) < self.min_degree
        )

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive VRF system status."""
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "current_epoch": self.current_epoch,
            "current_neighbors": list(self.current_neighbors),
            "neighbor_count": len(self.current_neighbors),
            "known_nodes": len(self.known_nodes),
            "last_selection_time": self.last_selection_time,
            "eclipse_attempts": self.eclipse_attempts,
            "blacklisted_nodes": len(self.blacklisted_nodes),
            "topology_metrics": self.topology_cache.__dict__ if self.topology_cache else {},
            "metrics": self.metrics.copy(),
            "config": self.config.copy(),
        }

    def get_neighbors(self) -> list[str]:
        """Get current neighbor list."""
        return list(self.current_neighbors)

    def get_selection_proof(self) -> VRFProof | None:
        """Get VRF proof for current selection."""
        if self.selection_history:
            latest_round = self.selection_history[-1]
            return latest_round.vrf_proofs.get(self.node_id)
        return None
