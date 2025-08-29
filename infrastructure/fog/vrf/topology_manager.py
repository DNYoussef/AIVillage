"""
VRF Topology Manager for Fog P2P Networks

Manages network topology properties including:
- K-core decomposition for high-tier node identification
- Spectral gap monitoring for expansion properties
- Network healing mechanisms
- Churn resistance and graph maintenance
"""

import asyncio
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from .vrf_neighbor_selection import NodeInfo, VRFNeighborSelector

logger = logging.getLogger(__name__)


class TopologyHealthStatus(Enum):
    """Network topology health status."""
    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    HEALING = "healing"


@dataclass
class NetworkMetrics:
    """Comprehensive network topology metrics."""
    # Basic connectivity
    node_count: int = 0
    edge_count: int = 0
    average_degree: float = 0.0
    degree_distribution: Dict[int, int] = field(default_factory=dict)
    
    # Expansion properties
    expansion_ratio: float = 0.0
    conductance: float = 0.0
    spectral_gap: float = 0.0  # λ₂ - second eigenvalue
    
    # Structural properties
    clustering_coefficient: float = 0.0
    diameter: int = 0
    radius: int = 0
    
    # K-core properties
    k_core_distribution: Dict[int, Set[str]] = field(default_factory=dict)
    max_k_core: int = 0
    k_core_nodes: Set[str] = field(default_factory=set)
    
    # Health indicators
    churn_resistance: float = 0.0
    fault_tolerance: float = 0.0
    eclipse_resistance: float = 0.0
    
    # Timing
    last_computed: float = field(default_factory=time.time)
    computation_time_ms: float = 0.0


@dataclass
class TopologyHealingAction:
    """Action to heal topology degradation."""
    action_type: str
    target_nodes: List[str]
    parameters: Dict[str, Any]
    priority: int
    expected_improvement: float
    estimated_cost: float


class GraphAnalyzer:
    """Analyzes graph properties and computes network metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compute_spectral_gap(self, adjacency_matrix: np.ndarray) -> float:
        """Compute spectral gap (λ₂) of the graph Laplacian."""
        if adjacency_matrix.shape[0] < 2:
            return 0.0
        
        try:
            # Create normalized Laplacian
            degrees = np.sum(adjacency_matrix, axis=1)
            degrees[degrees == 0] = 1  # Avoid division by zero
            
            # D^(-1/2) * A * D^(-1/2)
            d_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
            normalized_laplacian = np.eye(len(degrees)) - d_sqrt_inv @ adjacency_matrix @ d_sqrt_inv
            
            # Compute smallest eigenvalues
            n_eigs = min(5, adjacency_matrix.shape[0] - 1)
            if n_eigs <= 0:
                return 0.0
                
            eigenvals = eigsh(normalized_laplacian, k=n_eigs, which='SM', return_eigenvectors=False)
            eigenvals = np.sort(eigenvals)
            
            # Spectral gap is second smallest eigenvalue
            return float(eigenvals[1]) if len(eigenvals) > 1 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Failed to compute spectral gap: {e}")
            return 0.0
    
    def compute_k_core_decomposition(self, adjacency_dict: Dict[str, Set[str]]) -> Dict[int, Set[str]]:
        """Compute k-core decomposition of the network."""
        nodes = set(adjacency_dict.keys())
        k_cores = defaultdict(set)
        
        if not nodes:
            return dict(k_cores)
        
        # Initialize degrees
        degrees = {node: len(neighbors) for node, neighbors in adjacency_dict.items()}
        remaining_nodes = set(nodes)
        
        k = 0
        while remaining_nodes:
            k += 1
            
            # Find nodes with degree < k
            to_remove = {node for node in remaining_nodes if degrees[node] < k}
            
            if not to_remove:
                # All remaining nodes have degree >= k, they form k-core
                k_cores[k] = remaining_nodes.copy()
                break
            
            # Remove nodes and update degrees
            while to_remove:
                node = to_remove.pop()
                if node not in remaining_nodes:
                    continue
                    
                remaining_nodes.discard(node)
                k_cores[k-1].add(node)
                
                # Update neighbors' degrees
                for neighbor in adjacency_dict.get(node, set()):
                    if neighbor in remaining_nodes:
                        degrees[neighbor] -= 1
                        if degrees[neighbor] < k:
                            to_remove.add(neighbor)
        
        return dict(k_cores)
    
    def compute_expansion_ratio(self, adjacency_dict: Dict[str, Set[str]]) -> float:
        """Compute expansion ratio of the graph."""
        if not adjacency_dict:
            return 0.0
        
        nodes = list(adjacency_dict.keys())
        n = len(nodes)
        
        if n <= 1:
            return 0.0
        
        min_expansion = float('inf')
        
        # Check expansion for subsets up to n/2
        for subset_size in range(1, min(n // 2 + 1, 20)):  # Limit for performance
            # Sample random subsets
            for _ in range(min(10, math.comb(n, subset_size))):
                subset = set(np.random.choice(nodes, subset_size, replace=False))
                
                # Count edges leaving the subset
                boundary_edges = 0
                for node in subset:
                    for neighbor in adjacency_dict.get(node, set()):
                        if neighbor not in subset:
                            boundary_edges += 1
                
                # Expansion ratio = |boundary| / |subset|
                if subset_size > 0:
                    expansion = boundary_edges / subset_size
                    min_expansion = min(min_expansion, expansion)
        
        return min_expansion if min_expansion != float('inf') else 0.0
    
    def compute_clustering_coefficient(self, adjacency_dict: Dict[str, Set[str]]) -> float:
        """Compute average clustering coefficient."""
        if not adjacency_dict:
            return 0.0
        
        total_clustering = 0.0
        nodes_with_neighbors = 0
        
        for node, neighbors in adjacency_dict.items():
            if len(neighbors) < 2:
                continue
            
            # Count triangles
            triangles = 0
            neighbor_list = list(neighbors)
            
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    if neighbor_list[j] in adjacency_dict.get(neighbor_list[i], set()):
                        triangles += 1
            
            # Local clustering coefficient
            possible_edges = len(neighbors) * (len(neighbors) - 1) // 2
            if possible_edges > 0:
                local_clustering = triangles / possible_edges
                total_clustering += local_clustering
                nodes_with_neighbors += 1
        
        return total_clustering / nodes_with_neighbors if nodes_with_neighbors > 0 else 0.0
    
    def compute_diameter(self, adjacency_dict: Dict[str, Set[str]]) -> Tuple[int, int]:
        """Compute network diameter and radius using BFS."""
        if not adjacency_dict:
            return 0, 0
        
        nodes = list(adjacency_dict.keys())
        max_distance = 0
        min_eccentricity = float('inf')
        
        # Limit computation for large graphs
        sample_size = min(50, len(nodes))
        sampled_nodes = np.random.choice(nodes, sample_size, replace=False)
        
        for start_node in sampled_nodes:
            # BFS from start_node
            distances = {start_node: 0}
            queue = deque([start_node])
            
            while queue:
                current = queue.popleft()
                current_dist = distances[current]
                
                for neighbor in adjacency_dict.get(current, set()):
                    if neighbor not in distances:
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
            
            # Update diameter and radius
            if distances:
                eccentricity = max(distances.values())
                max_distance = max(max_distance, eccentricity)
                min_eccentricity = min(min_eccentricity, eccentricity)
        
        diameter = max_distance
        radius = int(min_eccentricity) if min_eccentricity != float('inf') else 0
        
        return diameter, radius


class TopologyManager:
    """
    Manages fog network topology properties and health.
    
    Responsibilities:
    - Monitor spectral gap and expansion properties
    - Track K-core membership for high-tier nodes
    - Detect topology degradation
    - Orchestrate healing actions
    - Maintain churn resistance
    """
    
    def __init__(
        self,
        vrf_selector: VRFNeighborSelector,
        target_spectral_gap: float = 0.2,
        min_k_core: int = 3,
        healing_threshold: float = 0.15,
        monitoring_interval: float = 300.0,  # 5 minutes
        **kwargs
    ):
        self.vrf_selector = vrf_selector
        self.target_spectral_gap = target_spectral_gap
        self.min_k_core = min_k_core
        self.healing_threshold = healing_threshold
        self.monitoring_interval = monitoring_interval
        
        # Graph analysis
        self.graph_analyzer = GraphAnalyzer()
        self.current_metrics: Optional[NetworkMetrics] = None
        
        # Network state
        self.adjacency_dict: Dict[str, Set[str]] = defaultdict(set)
        self.node_metadata: Dict[str, NodeInfo] = {}
        
        # Healing system
        self.healing_actions: List[TopologyHealingAction] = []
        self.healing_history: List[Dict[str, Any]] = []
        self.status = TopologyHealthStatus.GOOD
        
        # Background monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._healing_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.config = {
            "max_healing_actions": kwargs.get("max_healing_actions", 5),
            "metrics_history_size": kwargs.get("metrics_history_size", 100),
            "degree_variance_threshold": kwargs.get("degree_variance_threshold", 4.0),
            "churn_adaptation_rate": kwargs.get("churn_adaptation_rate", 0.1),
        }
        
        self.logger = logging.getLogger(__name__)
        
    async def start(self) -> bool:
        """Start topology management system."""
        try:
            self.logger.info("Starting topology management system...")
            
            # Initialize with current network state
            await self._update_network_state()
            
            # Start monitoring tasks
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            self._healing_task = asyncio.create_task(self._healing_loop())
            
            self.logger.info("Topology management system started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start topology manager: {e}")
            return False
    
    async def stop(self):
        """Stop topology management system."""
        self.logger.info("Stopping topology management system...")
        
        for task in [self._monitor_task, self._healing_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Topology management system stopped")
    
    async def update_network_topology(self, adjacency_updates: Dict[str, Set[str]]):
        """Update network topology with new adjacency information."""
        for node_id, neighbors in adjacency_updates.items():
            self.adjacency_dict[node_id] = neighbors
            
        # Update node metadata
        for node_id in adjacency_updates:
            if node_id in self.vrf_selector.known_nodes:
                self.node_metadata[node_id] = self.vrf_selector.known_nodes[node_id]
        
        # Trigger metrics recomputation
        await self._compute_network_metrics()
    
    async def get_k_core_nodes(self, min_k: int = None) -> Set[str]:
        """Get nodes in k-core with k >= min_k."""
        if not self.current_metrics:
            await self._compute_network_metrics()
        
        if not self.current_metrics:
            return set()
        
        min_k = min_k or self.min_k_core
        k_core_nodes = set()
        
        for k, nodes in self.current_metrics.k_core_distribution.items():
            if k >= min_k:
                k_core_nodes.update(nodes)
        
        return k_core_nodes
    
    async def assess_topology_health(self) -> TopologyHealthStatus:
        """Assess current topology health."""
        if not self.current_metrics:
            await self._compute_network_metrics()
        
        if not self.current_metrics:
            return TopologyHealthStatus.CRITICAL
        
        metrics = self.current_metrics
        
        # Health criteria
        spectral_gap_healthy = metrics.spectral_gap >= self.target_spectral_gap
        expansion_healthy = metrics.expansion_ratio >= 0.5
        connectivity_healthy = metrics.average_degree >= 4.0
        k_core_healthy = metrics.max_k_core >= self.min_k_core
        
        healthy_count = sum([
            spectral_gap_healthy,
            expansion_healthy,
            connectivity_healthy,
            k_core_healthy
        ])
        
        if healthy_count >= 4:
            status = TopologyHealthStatus.EXCELLENT
        elif healthy_count >= 3:
            status = TopologyHealthStatus.GOOD
        elif healthy_count >= 2:
            status = TopologyHealthStatus.DEGRADED
        else:
            status = TopologyHealthStatus.CRITICAL
        
        self.status = status
        return status
    
    async def plan_healing_actions(self) -> List[TopologyHealingAction]:
        """Plan healing actions for topology improvement."""
        if not self.current_metrics:
            await self._compute_network_metrics()
        
        healing_actions = []
        
        if not self.current_metrics:
            return healing_actions
        
        metrics = self.current_metrics
        
        # Action 1: Improve spectral gap
        if metrics.spectral_gap < self.target_spectral_gap:
            action = TopologyHealingAction(
                action_type="improve_spectral_gap",
                target_nodes=self._identify_bridge_nodes(),
                parameters={"target_gap": self.target_spectral_gap},
                priority=1,
                expected_improvement=0.3,
                estimated_cost=2.0
            )
            healing_actions.append(action)
        
        # Action 2: Strengthen k-core
        if metrics.max_k_core < self.min_k_core:
            weak_nodes = self._identify_weak_core_nodes()
            action = TopologyHealingAction(
                action_type="strengthen_k_core",
                target_nodes=weak_nodes,
                parameters={"min_k": self.min_k_core},
                priority=2,
                expected_improvement=0.4,
                estimated_cost=1.5
            )
            healing_actions.append(action)
        
        # Action 3: Improve expansion
        if metrics.expansion_ratio < 0.5:
            peripheral_nodes = self._identify_peripheral_nodes()
            action = TopologyHealingAction(
                action_type="improve_expansion",
                target_nodes=peripheral_nodes,
                parameters={"target_expansion": 0.7},
                priority=3,
                expected_improvement=0.25,
                estimated_cost=1.8
            )
            healing_actions.append(action)
        
        # Sort by priority and expected benefit
        healing_actions.sort(key=lambda x: (x.priority, -x.expected_improvement))
        
        return healing_actions[:self.config["max_healing_actions"]]
    
    async def execute_healing_action(self, action: TopologyHealingAction) -> bool:
        """Execute a topology healing action."""
        try:
            self.logger.info(f"Executing healing action: {action.action_type}")
            self.status = TopologyHealthStatus.HEALING
            
            success = False
            
            if action.action_type == "improve_spectral_gap":
                success = await self._heal_spectral_gap(action)
            elif action.action_type == "strengthen_k_core":
                success = await self._heal_k_core(action)
            elif action.action_type == "improve_expansion":
                success = await self._heal_expansion(action)
            
            # Record healing attempt
            self.healing_history.append({
                "timestamp": time.time(),
                "action": action.action_type,
                "target_nodes": action.target_nodes,
                "success": success,
                "parameters": action.parameters
            })
            
            # Keep history bounded
            if len(self.healing_history) > self.config["metrics_history_size"]:
                self.healing_history.pop(0)
            
            if success:
                self.logger.info(f"Healing action {action.action_type} completed successfully")
                # Recompute metrics after healing
                await self._compute_network_metrics()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to execute healing action {action.action_type}: {e}")
            return False
    
    def get_topology_status(self) -> Dict[str, Any]:
        """Get comprehensive topology status."""
        status = {
            "health_status": self.status.value,
            "node_count": len(self.adjacency_dict),
            "edge_count": sum(len(neighbors) for neighbors in self.adjacency_dict.values()) // 2,
            "monitoring_active": self._monitor_task and not self._monitor_task.done(),
            "healing_active": self._healing_task and not self._healing_task.done(),
            "pending_healing_actions": len(self.healing_actions),
            "recent_healings": len([h for h in self.healing_history if time.time() - h["timestamp"] < 3600])
        }
        
        if self.current_metrics:
            status.update({
                "spectral_gap": self.current_metrics.spectral_gap,
                "expansion_ratio": self.current_metrics.expansion_ratio,
                "max_k_core": self.current_metrics.max_k_core,
                "average_degree": self.current_metrics.average_degree,
                "clustering_coefficient": self.current_metrics.clustering_coefficient,
                "diameter": self.current_metrics.diameter,
                "last_metrics_update": self.current_metrics.last_computed
            })
        
        return status
    
    async def _update_network_state(self):
        """Update network state from VRF selector."""
        # Build adjacency from VRF selector's known topology
        self.adjacency_dict.clear()
        
        for node_id, node_info in self.vrf_selector.known_nodes.items():
            self.adjacency_dict[node_id] = set()
            self.node_metadata[node_id] = node_info
        
        # Add current neighbors (simplified - in practice would need full topology)
        current_neighbors = self.vrf_selector.get_neighbors()
        self.adjacency_dict[self.vrf_selector.node_id] = set(current_neighbors)
        
        # Add reverse edges for undirected graph
        for neighbor in current_neighbors:
            if neighbor in self.adjacency_dict:
                self.adjacency_dict[neighbor].add(self.vrf_selector.node_id)
    
    async def _compute_network_metrics(self) -> NetworkMetrics:
        """Compute comprehensive network metrics."""
        start_time = time.time()
        
        try:
            metrics = NetworkMetrics()
            
            if not self.adjacency_dict:
                return metrics
            
            # Basic properties
            metrics.node_count = len(self.adjacency_dict)
            metrics.edge_count = sum(len(neighbors) for neighbors in self.adjacency_dict.values()) // 2
            
            if metrics.node_count > 0:
                metrics.average_degree = 2 * metrics.edge_count / metrics.node_count
            
            # Degree distribution
            degrees = [len(neighbors) for neighbors in self.adjacency_dict.values()]
            metrics.degree_distribution = {d: degrees.count(d) for d in set(degrees)}
            
            # Spectral gap computation
            if metrics.node_count > 1:
                # Create adjacency matrix
                nodes = list(self.adjacency_dict.keys())
                n = len(nodes)
                node_to_idx = {node: i for i, node in enumerate(nodes)}
                
                adjacency_matrix = np.zeros((n, n))
                for node, neighbors in self.adjacency_dict.items():
                    i = node_to_idx[node]
                    for neighbor in neighbors:
                        if neighbor in node_to_idx:
                            j = node_to_idx[neighbor]
                            adjacency_matrix[i][j] = 1
                
                metrics.spectral_gap = self.graph_analyzer.compute_spectral_gap(adjacency_matrix)
            
            # Expansion properties
            metrics.expansion_ratio = self.graph_analyzer.compute_expansion_ratio(self.adjacency_dict)
            
            # Clustering coefficient
            metrics.clustering_coefficient = self.graph_analyzer.compute_clustering_coefficient(self.adjacency_dict)
            
            # Network diameter and radius
            metrics.diameter, metrics.radius = self.graph_analyzer.compute_diameter(self.adjacency_dict)
            
            # K-core decomposition
            metrics.k_core_distribution = self.graph_analyzer.compute_k_core_decomposition(self.adjacency_dict)
            if metrics.k_core_distribution:
                metrics.max_k_core = max(metrics.k_core_distribution.keys())
                metrics.k_core_nodes = set()
                for k, nodes in metrics.k_core_distribution.items():
                    if k >= self.min_k_core:
                        metrics.k_core_nodes.update(nodes)
            
            # Health indicators (simplified)
            metrics.churn_resistance = min(1.0, metrics.spectral_gap / self.target_spectral_gap)
            metrics.fault_tolerance = min(1.0, metrics.expansion_ratio / 0.5)
            metrics.eclipse_resistance = min(1.0, metrics.max_k_core / self.min_k_core)
            
            # Timing
            metrics.computation_time_ms = (time.time() - start_time) * 1000
            metrics.last_computed = time.time()
            
            self.current_metrics = metrics
            
            self.logger.debug(
                f"Computed network metrics: spectral_gap={metrics.spectral_gap:.3f}, "
                f"expansion={metrics.expansion_ratio:.3f}, k_core={metrics.max_k_core}"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to compute network metrics: {e}")
            return NetworkMetrics()
    
    def _identify_bridge_nodes(self) -> List[str]:
        """Identify nodes that could improve spectral gap."""
        # Simplified: return nodes with low degree that could benefit from more connections
        candidates = []
        
        for node_id, neighbors in self.adjacency_dict.items():
            if len(neighbors) < self.target_spectral_gap * 10:  # Heuristic
                candidates.append(node_id)
        
        return candidates[:5]  # Limit for performance
    
    def _identify_weak_core_nodes(self) -> List[str]:
        """Identify nodes in lower k-cores that could be strengthened."""
        if not self.current_metrics:
            return []
        
        weak_nodes = []
        for k, nodes in self.current_metrics.k_core_distribution.items():
            if k < self.min_k_core:
                weak_nodes.extend(list(nodes)[:3])  # Limit per k-core
        
        return weak_nodes[:10]  # Overall limit
    
    def _identify_peripheral_nodes(self) -> List[str]:
        """Identify peripheral nodes that could improve expansion."""
        # Find nodes with few connections to different network regions
        peripheral = []
        
        for node_id, neighbors in self.adjacency_dict.items():
            if len(neighbors) <= 2:  # Very low degree
                peripheral.append(node_id)
        
        return peripheral[:8]
    
    async def _heal_spectral_gap(self, action: TopologyHealingAction) -> bool:
        """Execute spectral gap healing by suggesting new connections."""
        try:
            # Suggest additional high-quality neighbors for bridge nodes
            for node_id in action.target_nodes:
                if node_id == self.vrf_selector.node_id:
                    # Trigger reselection for our node
                    await self.vrf_selector.select_neighbors(force_reselection=True)
                # For other nodes, we can only suggest - they need to implement healing
            
            return True
            
        except Exception as e:
            self.logger.error(f"Spectral gap healing failed: {e}")
            return False
    
    async def _heal_k_core(self, action: TopologyHealingAction) -> bool:
        """Execute k-core strengthening by increasing node degrees."""
        try:
            # Similar to spectral gap healing but focused on core strengthening
            weak_nodes = action.target_nodes
            
            if self.vrf_selector.node_id in weak_nodes:
                # Increase our target degree temporarily
                original_target = self.vrf_selector.target_degree
                self.vrf_selector.target_degree = min(
                    original_target + 2,
                    self.vrf_selector.max_degree
                )
                
                await self.vrf_selector.select_neighbors(force_reselection=True)
                
                # Restore original target after some time
                await asyncio.sleep(60)
                self.vrf_selector.target_degree = original_target
            
            return True
            
        except Exception as e:
            self.logger.error(f"K-core healing failed: {e}")
            return False
    
    async def _heal_expansion(self, action: TopologyHealingAction) -> bool:
        """Execute expansion improvement by diversifying connections."""
        try:
            # Focus on connecting to nodes in different network regions
            if self.vrf_selector.node_id in action.target_nodes:
                # Temporarily prioritize distant/diverse nodes
                await self.vrf_selector.select_neighbors(force_reselection=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Expansion healing failed: {e}")
            return False
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                # Update network state and compute metrics
                await self._update_network_state()
                await self._compute_network_metrics()
                
                # Assess health and plan healing if needed
                health = await self.assess_topology_health()
                
                if health in [TopologyHealthStatus.DEGRADED, TopologyHealthStatus.CRITICAL]:
                    self.healing_actions = await self.plan_healing_actions()
                    self.logger.warning(f"Topology health degraded ({health.value}), planned {len(self.healing_actions)} healing actions")
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _healing_loop(self):
        """Background healing execution loop."""
        while True:
            try:
                if self.healing_actions and self.status != TopologyHealthStatus.HEALING:
                    action = self.healing_actions.pop(0)
                    await self.execute_healing_action(action)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Healing loop error: {e}")
                await asyncio.sleep(60)