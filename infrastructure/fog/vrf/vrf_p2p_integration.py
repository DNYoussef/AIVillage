"""
VRF P2P Integration Layer for Fog Networks

Integrates VRF neighbor selection with existing P2P mesh networking infrastructure.
Provides seamless integration with transport managers and mesh protocols.
"""

import asyncio
import logging
import time
from typing import Any

from ..reputation.bayesian_reputation import BayesianReputationEngine
from .topology_manager import TopologyManager
from .vrf_neighbor_selection import NodeInfo, VRFNeighborSelector

logger = logging.getLogger(__name__)


class VRFIntegrationManager:
    """
    Manages integration between VRF neighbor selection and fog P2P networking.

    Provides:
    - VRF-based neighbor discovery and selection
    - Integration with existing P2P infrastructure
    - Reputation-enhanced connection management
    - Topology health monitoring and healing
    """

    def __init__(self, node_id: str, reputation_engine: BayesianReputationEngine | None = None, **kwargs):
        self.node_id = node_id

        # Core components
        self.vrf_selector = VRFNeighborSelector(node_id=node_id, reputation_engine=reputation_engine, **kwargs)

        self.topology_manager = TopologyManager(vrf_selector=self.vrf_selector, **kwargs)

        self.reputation_engine = reputation_engine

        # Connection management
        self.active_connections: dict[str, Any] = {}
        self.connection_metrics: dict[str, dict[str, float]] = {}

        # Configuration
        self.config = {
            "health_check_interval": kwargs.get("health_check_interval", 60.0),
            "metrics_update_interval": kwargs.get("metrics_update_interval", 30.0),
            "reputation_feedback_enabled": kwargs.get("reputation_feedback_enabled", True),
        }

        # Monitoring tasks
        self._health_monitor_task: asyncio.Task | None = None
        self._metrics_task: asyncio.Task | None = None

        self.logger = logging.getLogger(__name__)

    async def start(self) -> bool:
        """Start the VRF integration system."""
        try:
            self.logger.info("Starting VRF integration system...")

            # Start core components
            if not await self.vrf_selector.start():
                raise Exception("Failed to start VRF selector")

            if not await self.topology_manager.start():
                raise Exception("Failed to start topology manager")

            # Start monitoring
            self._health_monitor_task = asyncio.create_task(self._health_monitoring_loop())
            self._metrics_task = asyncio.create_task(self._metrics_update_loop())

            self.logger.info("VRF integration system started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start VRF integration: {e}")
            return False

    async def stop(self):
        """Stop the VRF integration system."""
        self.logger.info("Stopping VRF integration system...")

        # Cancel tasks
        for task in [self._health_monitor_task, self._metrics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop components
        await self.topology_manager.stop()
        await self.vrf_selector.stop()

        self.logger.info("VRF integration system stopped")

    async def add_discovered_node(self, node_info: NodeInfo) -> bool:
        """Add a discovered node to the VRF system."""
        try:
            await self.vrf_selector.add_node(node_info)

            # Initialize metrics tracking
            self.connection_metrics[node_info.node_id] = {
                "connection_attempts": 0,
                "successful_connections": 0,
                "failed_connections": 0,
                "avg_latency": 0.0,
                "last_seen": time.time(),
            }

            self.logger.debug(f"Added node {node_info.node_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add node {node_info.node_id}: {e}")
            return False

    async def get_optimal_neighbors(self, count: int | None = None) -> list[str]:
        """Get optimal neighbors using VRF selection."""
        try:
            neighbors = await self.vrf_selector.select_neighbors()

            if count and len(neighbors) > count:
                # Use reputation to prioritize if available
                if self.reputation_engine:
                    scored = [(nid, self.reputation_engine.get_trust_score(nid)) for nid in neighbors]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    return [nid for nid, _ in scored[:count]]
                else:
                    return neighbors[:count]

            return neighbors

        except Exception as e:
            self.logger.error(f"Failed to get optimal neighbors: {e}")
            return []

    def get_integration_status(self) -> dict[str, Any]:
        """Get integration system status."""
        vrf_status = self.vrf_selector.get_status()
        topology_status = self.topology_manager.get_topology_status()

        return {
            "node_id": self.node_id,
            "vrf_status": vrf_status,
            "topology_status": topology_status,
            "active_connections": len(self.active_connections),
            "total_known_nodes": len(self.vrf_selector.known_nodes),
            "reputation_active": self.reputation_engine is not None,
        }

    async def _health_monitoring_loop(self):
        """Background health monitoring."""
        while True:
            try:
                # Update metrics for active nodes
                current_time = time.time()
                for node_id in list(self.connection_metrics.keys()):
                    if node_id in self.vrf_selector.known_nodes:
                        await self.vrf_selector.update_node_metrics(node_id, last_seen=current_time)

                await asyncio.sleep(self.config["health_check_interval"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)

    async def _metrics_update_loop(self):
        """Background metrics updates."""
        while True:
            try:
                # Update connection metrics
                for node_id, metrics in self.connection_metrics.items():
                    # Simulate latency measurement
                    current_latency = 50.0  # ms

                    if metrics["avg_latency"] == 0:
                        metrics["avg_latency"] = current_latency
                    else:
                        metrics["avg_latency"] = 0.9 * metrics["avg_latency"] + 0.1 * current_latency

                await asyncio.sleep(self.config["metrics_update_interval"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(60)
