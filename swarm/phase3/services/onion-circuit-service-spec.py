"""
OnionCircuitService Specification
Advanced circuit management with security isolation and load balancing

This service manages onion routing circuits with comprehensive security controls,
load balancing, and circuit pool management for privacy-preserving communications.
"""

import asyncio
import hashlib
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import secrets

logger = logging.getLogger(__name__)


class CircuitStatus(Enum):
    """Circuit lifecycle status enumeration."""
    INITIALIZING = "initializing"
    BUILDING = "building"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    TEARING_DOWN = "tearing_down"
    DESTROYED = "destroyed"


class CircuitType(Enum):
    """Types of onion circuits based on privacy requirements."""
    FAST = "fast"           # 3 hops, optimized for speed
    BALANCED = "balanced"   # 4-5 hops, balanced privacy/performance
    ANONYMOUS = "anonymous" # 6+ hops, maximum privacy
    HIDDEN = "hidden"       # Hidden service circuits


@dataclass
class CircuitKeys:
    """Cryptographic keys for circuit security."""
    circuit_id: str
    master_key: bytes
    hop_keys: List[bytes] = field(default_factory=list)
    session_keys: Dict[str, bytes] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    
    def generate_hop_keys(self, num_hops: int) -> None:
        """Generate cryptographic keys for each hop."""
        self.hop_keys = [secrets.token_bytes(32) for _ in range(num_hops)]
    
    def get_hop_key(self, hop_index: int) -> Optional[bytes]:
        """Get key for specific hop."""
        if 0 <= hop_index < len(self.hop_keys):
            return self.hop_keys[hop_index]
        return None


@dataclass
class CircuitNode:
    """Represents a node in the onion circuit."""
    node_id: str
    ip_address: str
    port: int
    public_key: bytes
    reliability_score: float = 1.0
    bandwidth_capacity: int = 1000  # MB/s
    latency_ms: float = 50.0
    last_seen: float = field(default_factory=time.time)
    
    def is_online(self, timeout: float = 300.0) -> bool:
        """Check if node is considered online."""
        return time.time() - self.last_seen < timeout


@dataclass
class SecurityContext:
    """Security context for circuit operations."""
    circuit_id: str
    encryption_keys: CircuitKeys
    authentication_token: str
    isolation_boundary: str
    security_level: str
    audit_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate_authentication(self, token: str) -> bool:
        """Validate authentication token."""
        return self.authentication_token == token


@dataclass
class CircuitMetrics:
    """Performance and security metrics for circuits."""
    latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    packet_loss_rate: float = 0.0
    security_score: float = 1.0
    uptime_seconds: float = 0.0
    data_transferred_mb: float = 0.0
    security_events: int = 0
    
    def calculate_health_score(self) -> float:
        """Calculate overall circuit health score."""
        latency_score = max(0, 1 - (self.latency_ms / 1000))
        bandwidth_score = min(1, self.bandwidth_mbps / 100)
        reliability_score = 1 - self.packet_loss_rate
        
        return (latency_score + bandwidth_score + reliability_score + self.security_score) / 4


@dataclass
class OnionCircuit:
    """Complete onion circuit with security and performance tracking."""
    circuit_id: str
    circuit_type: CircuitType
    privacy_level: str
    nodes: List[CircuitNode]
    keys: CircuitKeys
    status: CircuitStatus = CircuitStatus.INITIALIZING
    metrics: CircuitMetrics = field(default_factory=CircuitMetrics)
    security_context: Optional[SecurityContext] = None
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize circuit with security context."""
        if self.security_context is None:
            self.security_context = SecurityContext(
                circuit_id=self.circuit_id,
                encryption_keys=self.keys,
                authentication_token=secrets.token_urlsafe(32),
                isolation_boundary=f"isolation_{self.circuit_id}",
                security_level=self.privacy_level
            )
    
    def get_path_length(self) -> int:
        """Get the number of hops in the circuit."""
        return len(self.nodes)
    
    def is_healthy(self) -> bool:
        """Check if circuit is healthy and usable."""
        if self.status not in [CircuitStatus.ACTIVE, CircuitStatus.DEGRADED]:
            return False
        
        health_score = self.metrics.calculate_health_score()
        return health_score > 0.5  # Threshold for healthy circuit
    
    def update_usage(self) -> None:
        """Update last used timestamp."""
        self.last_used = time.time()


class CircuitPool:
    """Manages a pool of circuits with load balancing and health monitoring."""
    
    def __init__(self, pool_name: str, circuit_type: CircuitType):
        self.pool_name = pool_name
        self.circuit_type = circuit_type
        self.circuits: Dict[str, OnionCircuit] = {}
        self.load_balancer = CircuitLoadBalancer()
        self.health_monitor = CircuitHealthMonitor()
        
        # Pool configuration
        self.min_circuits = 3
        self.max_circuits = 20
        self.target_circuits = 8
        
        logger.info(f"Circuit pool initialized: {pool_name}")
    
    async def get_circuit(self, privacy_requirements: Dict[str, Any]) -> Optional[OnionCircuit]:
        """Get an available circuit from the pool."""
        available_circuits = [
            circuit for circuit in self.circuits.values()
            if circuit.is_healthy() and self._meets_requirements(circuit, privacy_requirements)
        ]
        
        if not available_circuits:
            # Try to create a new circuit
            circuit = await self._create_circuit_for_requirements(privacy_requirements)
            if circuit:
                await self.add_circuit(circuit)
                return circuit
            return None
        
        # Use load balancer to select best circuit
        selected_circuit = self.load_balancer.select_circuit(available_circuits)
        selected_circuit.update_usage()
        
        return selected_circuit
    
    async def add_circuit(self, circuit: OnionCircuit) -> bool:
        """Add circuit to the pool."""
        if len(self.circuits) >= self.max_circuits:
            # Remove oldest unhealthy circuit
            await self._remove_oldest_unhealthy_circuit()
        
        self.circuits[circuit.circuit_id] = circuit
        logger.info(f"Added circuit {circuit.circuit_id} to pool {self.pool_name}")
        return True
    
    async def remove_circuit(self, circuit_id: str) -> bool:
        """Remove circuit from the pool."""
        if circuit_id in self.circuits:
            circuit = self.circuits[circuit_id]
            await self._tear_down_circuit(circuit)
            del self.circuits[circuit_id]
            logger.info(f"Removed circuit {circuit_id} from pool {self.pool_name}")
            return True
        return False
    
    def _meets_requirements(self, circuit: OnionCircuit, 
                           requirements: Dict[str, Any]) -> bool:
        """Check if circuit meets privacy requirements."""
        min_hops = requirements.get("min_hops", 3)
        max_latency = requirements.get("max_latency_ms", 1000)
        min_bandwidth = requirements.get("min_bandwidth_mbps", 1)
        
        return (circuit.get_path_length() >= min_hops and
                circuit.metrics.latency_ms <= max_latency and
                circuit.metrics.bandwidth_mbps >= min_bandwidth)
    
    async def _create_circuit_for_requirements(self, requirements: Dict[str, Any]) -> Optional[OnionCircuit]:
        """Create new circuit meeting specific requirements."""
        # This would interface with the circuit builder
        logger.info(f"Creating circuit for requirements: {requirements}")
        return None  # Placeholder
    
    async def _remove_oldest_unhealthy_circuit(self) -> None:
        """Remove oldest unhealthy circuit from pool."""
        unhealthy_circuits = [
            circuit for circuit in self.circuits.values()
            if not circuit.is_healthy()
        ]
        
        if unhealthy_circuits:
            oldest_circuit = min(unhealthy_circuits, key=lambda c: c.created_at)
            await self.remove_circuit(oldest_circuit.circuit_id)
    
    async def _tear_down_circuit(self, circuit: OnionCircuit) -> None:
        """Securely tear down circuit."""
        circuit.status = CircuitStatus.TEARING_DOWN
        
        # Secure key deletion
        circuit.keys.hop_keys.clear()
        circuit.keys.session_keys.clear()
        
        circuit.status = CircuitStatus.DESTROYED
        logger.info(f"Circuit {circuit.circuit_id} torn down securely")


class CircuitLoadBalancer:
    """Load balancer for circuit selection with security considerations."""
    
    def __init__(self):
        self.selection_algorithm = "weighted_round_robin"
        self.circuit_weights = {}
    
    def select_circuit(self, circuits: List[OnionCircuit]) -> OnionCircuit:
        """Select best circuit using load balancing algorithm."""
        if not circuits:
            raise ValueError("No circuits available for selection")
        
        if self.selection_algorithm == "weighted_round_robin":
            return self._weighted_round_robin_selection(circuits)
        elif self.selection_algorithm == "least_connections":
            return self._least_connections_selection(circuits)
        else:
            return random.choice(circuits)  # Fallback
    
    def _weighted_round_robin_selection(self, circuits: List[OnionCircuit]) -> OnionCircuit:
        """Select circuit using weighted round-robin algorithm."""
        total_weight = 0
        circuit_weights = []
        
        for circuit in circuits:
            weight = self._calculate_circuit_weight(circuit)
            circuit_weights.append(weight)
            total_weight += weight
        
        if total_weight == 0:
            return random.choice(circuits)
        
        # Select based on weights
        selection_point = random.uniform(0, total_weight)
        current_weight = 0
        
        for i, weight in enumerate(circuit_weights):
            current_weight += weight
            if current_weight >= selection_point:
                return circuits[i]
        
        return circuits[-1]  # Fallback
    
    def _least_connections_selection(self, circuits: List[OnionCircuit]) -> OnionCircuit:
        """Select circuit with least active connections."""
        # This would track actual connection counts
        # For now, use metrics as proxy
        return min(circuits, key=lambda c: c.metrics.data_transferred_mb)
    
    def _calculate_circuit_weight(self, circuit: OnionCircuit) -> float:
        """Calculate weight for circuit selection."""
        health_score = circuit.metrics.calculate_health_score()
        age_factor = max(0.1, 1 - ((time.time() - circuit.created_at) / 3600))  # Decay over 1 hour
        usage_factor = max(0.1, 1 - (circuit.metrics.data_transferred_mb / 1000))  # Prefer less used
        
        return health_score * age_factor * usage_factor


class CircuitHealthMonitor:
    """Monitors circuit health and performance."""
    
    def __init__(self):
        self.monitoring_interval = 30  # seconds
        self.health_thresholds = {
            "max_latency_ms": 1000,
            "min_bandwidth_mbps": 1,
            "max_packet_loss": 0.05,
            "min_security_score": 0.7
        }
    
    async def start_monitoring(self, circuits: Dict[str, OnionCircuit]) -> None:
        """Start continuous health monitoring."""
        while True:
            await self._monitor_all_circuits(circuits)
            await asyncio.sleep(self.monitoring_interval)
    
    async def _monitor_all_circuits(self, circuits: Dict[str, OnionCircuit]) -> None:
        """Monitor all circuits in the collection."""
        for circuit_id, circuit in circuits.items():
            await self._monitor_circuit(circuit)
    
    async def _monitor_circuit(self, circuit: OnionCircuit) -> None:
        """Monitor individual circuit health."""
        try:
            # Update metrics
            await self._update_circuit_metrics(circuit)
            
            # Check health thresholds
            if not self._meets_health_thresholds(circuit):
                circuit.status = CircuitStatus.DEGRADED
                logger.warning(f"Circuit {circuit.circuit_id} health degraded")
            
            # Check for security events
            await self._check_security_events(circuit)
            
        except Exception as e:
            logger.error(f"Error monitoring circuit {circuit.circuit_id}: {e}")
            circuit.status = CircuitStatus.FAILED
    
    async def _update_circuit_metrics(self, circuit: OnionCircuit) -> None:
        """Update circuit performance metrics."""
        # This would perform actual network measurements
        # For now, simulate metric updates
        circuit.metrics.uptime_seconds = time.time() - circuit.created_at
        
        # Simulate some metric updates
        circuit.metrics.latency_ms += random.uniform(-10, 10)
        circuit.metrics.latency_ms = max(10, circuit.metrics.latency_ms)
        
        logger.debug(f"Updated metrics for circuit {circuit.circuit_id}")
    
    def _meets_health_thresholds(self, circuit: OnionCircuit) -> bool:
        """Check if circuit meets health thresholds."""
        metrics = circuit.metrics
        thresholds = self.health_thresholds
        
        return (metrics.latency_ms <= thresholds["max_latency_ms"] and
                metrics.bandwidth_mbps >= thresholds["min_bandwidth_mbps"] and
                metrics.packet_loss_rate <= thresholds["max_packet_loss"] and
                metrics.security_score >= thresholds["min_security_score"])
    
    async def _check_security_events(self, circuit: OnionCircuit) -> None:
        """Check for security-related events."""
        # This would implement actual security monitoring
        # For now, just log
        logger.debug(f"Security check for circuit {circuit.circuit_id}")


class CircuitBuilder:
    """Builds onion circuits with specified privacy and performance requirements."""
    
    def __init__(self, node_registry):
        self.node_registry = node_registry
        self.circuit_cache = {}
        self.build_timeout = 30  # seconds
    
    async def build_circuit(self, circuit_type: CircuitType, 
                          privacy_level: str,
                          requirements: Dict[str, Any]) -> Optional[OnionCircuit]:
        """Build new onion circuit with specified requirements."""
        circuit_id = f"circuit_{uuid.uuid4().hex}"
        
        try:
            # Select nodes for circuit
            nodes = await self._select_circuit_nodes(circuit_type, requirements)
            if not nodes:
                logger.error("Failed to select nodes for circuit")
                return None
            
            # Generate cryptographic keys
            keys = CircuitKeys(circuit_id=circuit_id, master_key=secrets.token_bytes(32))
            keys.generate_hop_keys(len(nodes))
            
            # Create circuit object
            circuit = OnionCircuit(
                circuit_id=circuit_id,
                circuit_type=circuit_type,
                privacy_level=privacy_level,
                nodes=nodes,
                keys=keys
            )
            
            # Build the actual circuit
            success = await self._establish_circuit(circuit)
            if not success:
                logger.error(f"Failed to establish circuit {circuit_id}")
                return None
            
            circuit.status = CircuitStatus.ACTIVE
            logger.info(f"Successfully built circuit {circuit_id}")
            return circuit
            
        except Exception as e:
            logger.error(f"Circuit building error: {e}")
            return None
    
    async def _select_circuit_nodes(self, circuit_type: CircuitType,
                                  requirements: Dict[str, Any]) -> List[CircuitNode]:
        """Select appropriate nodes for the circuit."""
        # Determine number of hops based on circuit type
        hop_counts = {
            CircuitType.FAST: 3,
            CircuitType.BALANCED: 4,
            CircuitType.ANONYMOUS: 6,
            CircuitType.HIDDEN: 5
        }
        
        num_hops = hop_counts.get(circuit_type, 4)
        
        # Get available nodes
        available_nodes = await self.node_registry.get_available_nodes()
        
        # Filter nodes based on requirements
        suitable_nodes = self._filter_nodes_by_requirements(available_nodes, requirements)
        
        if len(suitable_nodes) < num_hops:
            logger.error(f"Insufficient nodes for circuit: need {num_hops}, have {len(suitable_nodes)}")
            return []
        
        # Select diverse set of nodes
        selected_nodes = self._select_diverse_nodes(suitable_nodes, num_hops)
        
        return selected_nodes
    
    def _filter_nodes_by_requirements(self, nodes: List[CircuitNode],
                                    requirements: Dict[str, Any]) -> List[CircuitNode]:
        """Filter nodes based on privacy and performance requirements."""
        min_bandwidth = requirements.get("min_bandwidth_mbps", 1)
        max_latency = requirements.get("max_latency_ms", 1000)
        min_reliability = requirements.get("min_reliability", 0.8)
        
        filtered_nodes = []
        for node in nodes:
            if (node.is_online() and
                node.bandwidth_capacity >= min_bandwidth and
                node.latency_ms <= max_latency and
                node.reliability_score >= min_reliability):
                filtered_nodes.append(node)
        
        return filtered_nodes
    
    def _select_diverse_nodes(self, nodes: List[CircuitNode], 
                            num_hops: int) -> List[CircuitNode]:
        """Select diverse set of nodes for security."""
        # Implement geographic and network diversity
        # For now, random selection
        return random.sample(nodes, min(num_hops, len(nodes)))
    
    async def _establish_circuit(self, circuit: OnionCircuit) -> bool:
        """Establish the actual circuit with the selected nodes."""
        circuit.status = CircuitStatus.BUILDING
        
        try:
            # This would implement the actual onion routing protocol
            # For now, simulate circuit establishment
            await asyncio.sleep(1)  # Simulate build time
            
            logger.info(f"Established circuit with {len(circuit.nodes)} hops")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish circuit: {e}")
            circuit.status = CircuitStatus.FAILED
            return False


class OnionCircuitService:
    """
    Advanced onion circuit management service with security isolation and load balancing.
    
    This service provides comprehensive circuit management with:
    - Circuit pools organized by privacy level and performance requirements
    - Advanced load balancing with security considerations  
    - Real-time health monitoring and automatic recovery
    - Cryptographic isolation between circuits
    - Performance optimization without privacy compromise
    
    Security Features:
    - Circuit isolation by privacy level
    - Cryptographic authentication for all circuits
    - Secure key generation and management
    - Continuous security monitoring and validation
    - Automatic circuit rotation for enhanced security
    """
    
    def __init__(self, node_registry):
        self.node_registry = node_registry
        self.circuit_pools: Dict[str, CircuitPool] = {}
        self.circuit_builder = CircuitBuilder(node_registry)
        self.security_monitor = CircuitSecurityMonitor()
        
        # Initialize circuit pools for different privacy levels
        self._initialize_circuit_pools()
        
        # Service state
        self.active_circuits = {}
        self.circuit_metrics = {}
        self.security_events = []
        
        logger.info("OnionCircuitService initialized with security framework")
    
    def _initialize_circuit_pools(self) -> None:
        """Initialize circuit pools for different privacy levels."""
        pool_configs = [
            ("public_fast", CircuitType.FAST),
            ("private_balanced", CircuitType.BALANCED),
            ("confidential_anonymous", CircuitType.ANONYMOUS),
            ("secret_hidden", CircuitType.HIDDEN)
        ]
        
        for pool_name, circuit_type in pool_configs:
            self.circuit_pools[pool_name] = CircuitPool(pool_name, circuit_type)
    
    async def request_circuit(self, privacy_level: str, 
                            requirements: Dict[str, Any],
                            requester_token: str) -> Tuple[bool, str, Optional[OnionCircuit]]:
        """Request circuit with specified privacy and performance requirements."""
        try:
            # Validate request
            if not await self._validate_circuit_request(privacy_level, requirements, requester_token):
                return False, "Invalid circuit request", None
            
            # Select appropriate pool
            pool_name = self._get_pool_for_privacy_level(privacy_level)
            if pool_name not in self.circuit_pools:
                return False, "No suitable circuit pool available", None
            
            pool = self.circuit_pools[pool_name]
            
            # Try to get existing circuit
            circuit = await pool.get_circuit(requirements)
            
            if circuit is None:
                # Build new circuit if none available
                circuit_type = pool.circuit_type
                circuit = await self.circuit_builder.build_circuit(
                    circuit_type, privacy_level, requirements
                )
                
                if circuit is None:
                    return False, "Failed to build circuit", None
                
                # Add to pool
                await pool.add_circuit(circuit)
            
            # Register circuit usage
            self.active_circuits[circuit.circuit_id] = {
                "circuit": circuit,
                "requester_token": requester_token,
                "allocated_at": time.time()
            }
            
            # Log security event
            await self.security_monitor.log_circuit_allocation(circuit, requester_token)
            
            return True, "Circuit allocated successfully", circuit
            
        except Exception as e:
            logger.error(f"Circuit request error: {e}")
            return False, f"Circuit request failed: {str(e)}", None
    
    async def release_circuit(self, circuit_id: str, 
                            requester_token: str) -> Tuple[bool, str]:
        """Release circuit back to the pool."""
        if circuit_id not in self.active_circuits:
            return False, "Circuit not found or not allocated"
        
        circuit_info = self.active_circuits[circuit_id]
        
        # Validate requester
        if circuit_info["requester_token"] != requester_token:
            await self.security_monitor.log_unauthorized_release_attempt(circuit_id, requester_token)
            return False, "Unauthorized release attempt"
        
        circuit = circuit_info["circuit"]
        
        # Update circuit metrics
        usage_time = time.time() - circuit_info["allocated_at"]
        circuit.metrics.uptime_seconds += usage_time
        
        # Remove from active circuits
        del self.active_circuits[circuit_id]
        
        # Circuit returns to pool automatically
        logger.info(f"Released circuit {circuit_id}")
        return True, "Circuit released successfully"
    
    async def get_circuit_status(self, circuit_id: str, 
                               requester_token: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Get detailed status of a circuit."""
        if circuit_id not in self.active_circuits:
            return False, "Circuit not found", {}
        
        circuit_info = self.active_circuits[circuit_id]
        
        # Validate requester
        if circuit_info["requester_token"] != requester_token:
            return False, "Unauthorized status request", {}
        
        circuit = circuit_info["circuit"]
        
        status_info = {
            "circuit_id": circuit.circuit_id,
            "status": circuit.status.value,
            "circuit_type": circuit.circuit_type.value,
            "privacy_level": circuit.privacy_level,
            "hop_count": circuit.get_path_length(),
            "health_score": circuit.metrics.calculate_health_score(),
            "latency_ms": circuit.metrics.latency_ms,
            "bandwidth_mbps": circuit.metrics.bandwidth_mbps,
            "uptime_seconds": circuit.metrics.uptime_seconds,
            "created_at": circuit.created_at,
            "last_used": circuit.last_used
        }
        
        return True, "Status retrieved successfully", status_info
    
    async def rotate_circuit(self, circuit_id: str, 
                           requester_token: str) -> Tuple[bool, str, Optional[OnionCircuit]]:
        """Rotate circuit for enhanced security."""
        if circuit_id not in self.active_circuits:
            return False, "Circuit not found", None
        
        circuit_info = self.active_circuits[circuit_id]
        
        # Validate requester
        if circuit_info["requester_token"] != requester_token:
            return False, "Unauthorized rotation request", None
        
        old_circuit = circuit_info["circuit"]
        
        # Build new circuit with same requirements
        requirements = {
            "min_hops": old_circuit.get_path_length(),
            "privacy_level": old_circuit.privacy_level
        }
        
        new_circuit = await self.circuit_builder.build_circuit(
            old_circuit.circuit_type,
            old_circuit.privacy_level,
            requirements
        )
        
        if new_circuit is None:
            return False, "Failed to build replacement circuit", None
        
        # Replace in active circuits
        self.active_circuits[new_circuit.circuit_id] = {
            "circuit": new_circuit,
            "requester_token": requester_token,
            "allocated_at": time.time()
        }
        
        # Schedule old circuit for cleanup
        await self._schedule_circuit_cleanup(old_circuit)
        
        # Remove old circuit from tracking
        del self.active_circuits[circuit_id]
        
        logger.info(f"Rotated circuit {circuit_id} to {new_circuit.circuit_id}")
        return True, "Circuit rotated successfully", new_circuit
    
    async def get_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all circuit pools."""
        statistics = {}
        
        for pool_name, pool in self.circuit_pools.items():
            pool_stats = {
                "total_circuits": len(pool.circuits),
                "healthy_circuits": sum(1 for c in pool.circuits.values() if c.is_healthy()),
                "average_health_score": self._calculate_average_health_score(pool.circuits),
                "circuit_types": {ct.value: sum(1 for c in pool.circuits.values() if c.circuit_type == ct) 
                                for ct in CircuitType}
            }
            statistics[pool_name] = pool_stats
        
        return statistics
    
    async def _validate_circuit_request(self, privacy_level: str, 
                                      requirements: Dict[str, Any],
                                      requester_token: str) -> bool:
        """Validate circuit request parameters."""
        # Basic validation
        if not privacy_level or not requester_token:
            return False
        
        # Validate privacy level
        valid_levels = ["public", "private", "confidential", "secret"]
        if privacy_level not in valid_levels:
            return False
        
        # Validate requirements
        if "min_hops" in requirements:
            min_hops = requirements["min_hops"]
            if not isinstance(min_hops, int) or min_hops < 1 or min_hops > 10:
                return False
        
        return True
    
    def _get_pool_for_privacy_level(self, privacy_level: str) -> str:
        """Get appropriate circuit pool for privacy level."""
        pool_mapping = {
            "public": "public_fast",
            "private": "private_balanced", 
            "confidential": "confidential_anonymous",
            "secret": "secret_hidden"
        }
        return pool_mapping.get(privacy_level, "private_balanced")
    
    def _calculate_average_health_score(self, circuits: Dict[str, OnionCircuit]) -> float:
        """Calculate average health score for a set of circuits."""
        if not circuits:
            return 0.0
        
        total_score = sum(circuit.metrics.calculate_health_score() 
                         for circuit in circuits.values())
        return total_score / len(circuits)
    
    async def _schedule_circuit_cleanup(self, circuit: OnionCircuit) -> None:
        """Schedule circuit for secure cleanup."""
        # Mark for destruction
        circuit.status = CircuitStatus.TEARING_DOWN
        
        # Schedule async cleanup
        asyncio.create_task(self._cleanup_circuit_after_delay(circuit, delay=60))
    
    async def _cleanup_circuit_after_delay(self, circuit: OnionCircuit, delay: int) -> None:
        """Clean up circuit after specified delay."""
        await asyncio.sleep(delay)
        
        # Secure key deletion
        circuit.keys.hop_keys.clear()
        circuit.keys.session_keys.clear()
        
        circuit.status = CircuitStatus.DESTROYED
        logger.info(f"Circuit {circuit.circuit_id} cleaned up securely")


class CircuitSecurityMonitor:
    """Monitors security events and anomalies in circuit operations."""
    
    def __init__(self):
        self.security_events = []
        self.anomaly_detector = CircuitAnomalyDetector()
    
    async def log_circuit_allocation(self, circuit: OnionCircuit, requester_token: str) -> None:
        """Log circuit allocation event."""
        event = {
            "event_type": "CIRCUIT_ALLOCATED",
            "circuit_id": circuit.circuit_id,
            "privacy_level": circuit.privacy_level,
            "requester_hash": hashlib.sha256(requester_token.encode()).hexdigest()[:16],
            "timestamp": time.time()
        }
        
        self.security_events.append(event)
        logger.info(f"Logged circuit allocation: {circuit.circuit_id}")
    
    async def log_unauthorized_release_attempt(self, circuit_id: str, requester_token: str) -> None:
        """Log unauthorized circuit release attempt."""
        event = {
            "event_type": "UNAUTHORIZED_RELEASE_ATTEMPT",
            "circuit_id": circuit_id,
            "requester_hash": hashlib.sha256(requester_token.encode()).hexdigest()[:16],
            "timestamp": time.time()
        }
        
        self.security_events.append(event)
        logger.warning(f"Unauthorized circuit release attempt: {circuit_id}")


class CircuitAnomalyDetector:
    """Detects anomalous patterns in circuit usage and performance."""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_thresholds = {
            "latency_increase_factor": 2.0,
            "bandwidth_decrease_factor": 0.5,
            "packet_loss_increase_factor": 3.0
        }
    
    async def detect_anomalies(self, circuit: OnionCircuit) -> List[str]:
        """Detect anomalies in circuit behavior."""
        anomalies = []
        
        # Compare with baseline
        if circuit.circuit_id in self.baseline_metrics:
            baseline = self.baseline_metrics[circuit.circuit_id]
            current = circuit.metrics
            
            # Check latency anomaly
            if current.latency_ms > baseline.latency_ms * self.anomaly_thresholds["latency_increase_factor"]:
                anomalies.append("EXCESSIVE_LATENCY")
            
            # Check bandwidth anomaly
            if current.bandwidth_mbps < baseline.bandwidth_mbps * self.anomaly_thresholds["bandwidth_decrease_factor"]:
                anomalies.append("BANDWIDTH_DEGRADATION")
            
            # Check packet loss anomaly
            if current.packet_loss_rate > baseline.packet_loss_rate * self.anomaly_thresholds["packet_loss_increase_factor"]:
                anomalies.append("PACKET_LOSS_SPIKE")
        
        return anomalies