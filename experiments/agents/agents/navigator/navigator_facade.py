"""Navigator Facade - Main interface maintaining original PathPolicy compatibility

This facade coordinates all navigation services while maintaining the original
NavigatorAgent interface for backward compatibility and seamless integration.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from .interfaces.routing_interfaces import INavigatorFacadeService, RoutingEvent
from .events.event_bus import get_event_bus, initialize_event_bus

# Import all services
from .services.route_selection_service import RouteSelectionService
from .services.protocol_manager_service import ProtocolManagerService
from .services.network_monitoring_service import NetworkMonitoringService
from .services.qos_manager_service import QoSManagerService
from .services.dtn_handler_service import DTNHandlerService
from .services.energy_optimization_service import EnergyOptimizationService
from .services.security_mixnode_service import SecurityMixnodeService

# Import original types for compatibility
from .path_policy import (
    EnergyMode, MessageContext, NetworkConditions, PathProtocol,
    PeerInfo, RoutingPriority, Receipt
)

logger = logging.getLogger(__name__)


class NavigatorFacade(INavigatorFacadeService):
    """Main Navigator facade coordinating all routing services
    
    This facade maintains the original NavigatorAgent interface while coordinating
    the new service-oriented architecture. It provides:
    
    - Backward compatibility with existing code
    - Centralized service coordination
    - Event-driven inter-service communication
    - Performance monitoring and optimization
    - Unified configuration and status reporting
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        routing_priority: RoutingPriority = RoutingPriority.OFFLINE_FIRST,
        energy_mode: EnergyMode = EnergyMode.BALANCED,
    ):
        self.agent_id = agent_id or f"navigator_{uuid.uuid4().hex[:8]}"
        self.routing_priority = routing_priority
        self.energy_mode = energy_mode
        
        # Service initialization
        self.event_bus = get_event_bus()
        
        # Core services
        self.route_selection = RouteSelectionService()
        self.protocol_manager = ProtocolManagerService()
        self.network_monitoring = NetworkMonitoringService()
        self.qos_manager = QoSManagerService()
        self.dtn_handler = DTNHandlerService()
        self.energy_optimizer = EnergyOptimizationService()
        self.security_mixnode = SecurityMixnodeService()
        
        # Service registry for easy access
        self.services = {
            "route_selection": self.route_selection,
            "protocol_manager": self.protocol_manager,
            "network_monitoring": self.network_monitoring,
            "qos_manager": self.qos_manager,
            "dtn_handler": self.dtn_handler,
            "energy_optimizer": self.energy_optimizer,
            "security_mixnode": self.security_mixnode
        }
        
        # Facade state and compatibility
        self.receipts: List[Receipt] = []
        self.max_receipts = 1000
        self.discovered_peers: Dict[str, PeerInfo] = {}
        
        # Performance tracking
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
        
        # Service coordination
        self.service_startup_complete = False
        self._setup_event_subscriptions()
        
        logger.info(f"NavigatorFacade initialized: {self.agent_id} (priority={routing_priority.value})")
    
    async def initialize(self) -> None:
        """Initialize all services and event bus"""
        try:
            # Initialize event bus
            await initialize_event_bus()
            
            # Start all services
            await asyncio.gather(
                self.network_monitoring.start_monitoring(),
                self.protocol_manager.manage_connections(),
                self.energy_optimizer.start_monitoring(),
                self.security_mixnode.start_service(),
                self.dtn_handler.start_service()
            )
            
            self.service_startup_complete = True
            logger.info("NavigatorFacade initialization complete")
            
        except Exception as e:
            logger.error(f"NavigatorFacade initialization failed: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown all services gracefully"""
        try:
            # Stop all services
            await asyncio.gather(
                self.network_monitoring.stop_monitoring(),
                self.energy_optimizer.stop_monitoring(),
                self.security_mixnode.stop_service(),
                self.dtn_handler.stop_service(),
                return_exceptions=True
            )
            
            logger.info("NavigatorFacade shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during NavigatorFacade shutdown: {e}")
    
    def _setup_event_subscriptions(self) -> None:
        """Set up inter-service event subscriptions"""
        # Route selection events
        self.event_bus.subscribe(
            "route_selected",
            self._handle_route_selected,
            "NavigatorFacade"
        )
        
        # Network change events
        self.event_bus.subscribe(
            "significant_change_detected",
            self._handle_network_change,
            "NavigatorFacade"
        )
        
        # Protocol events
        self.event_bus.subscribe(
            "protocol_switched",
            self._handle_protocol_switch,
            "NavigatorFacade"
        )
        
        # Energy events
        self.event_bus.subscribe(
            "power_management_update",
            self._handle_power_update,
            "NavigatorFacade"
        )
        
        # Security events
        self.event_bus.subscribe(
            "circuit_created",
            self._handle_circuit_created,
            "NavigatorFacade"
        )
    
    async def select_path(
        self,
        destination: str,
        message_context: MessageContext,
        available_protocols: Optional[List[str]] = None,
    ) -> Tuple[PathProtocol, Dict[str, Any]]:
        """Main path selection method - core NavigatorAgent compatibility interface"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Ensure services are initialized
            if not self.service_startup_complete:
                await self.initialize()
            
            # Step 1: Get current network conditions
            network_conditions = await self.network_monitoring.monitor_network_links()
            
            # Step 2: Apply energy optimization
            if available_protocols:
                optimized_protocols = self.energy_optimizer.optimize_for_battery_life(
                    network_conditions.battery_percent,
                    [PathProtocol(p) for p in available_protocols if hasattr(PathProtocol, p.upper())]
                )
                available_protocols = [p.value for p in optimized_protocols]
            
            # Step 3: Core route selection
            selected_protocol, path_scores = await self.route_selection.select_optimal_route(
                destination,
                message_context,
                available_protocols,
                network_conditions
            )
            
            # Step 4: Configure QoS parameters
            qos_config = await self.qos_manager.manage_qos_parameters(
                selected_protocol,
                message_context
            )
            
            # Step 5: Apply privacy/security measures if required
            privacy_config = {}
            if message_context.privacy_required:
                privacy_config = self.security_mixnode.ensure_routing_privacy(
                    selected_protocol,
                    message_context
                )
                
                # Get mixnodes if needed
                if privacy_config.get("mixnode_hops", 0) > 0:
                    mixnodes = await self.security_mixnode.select_privacy_mixnodes(
                        destination,
                        0.8  # Default privacy level
                    )
                    privacy_config["selected_mixnodes"] = mixnodes
            
            # Step 6: Handle store-and-forward if needed
            if selected_protocol == PathProtocol.STORE_FORWARD:
                # Store message for later delivery
                message_stored = await self.dtn_handler.store_message(
                    f"msg_{uuid.uuid4().hex[:8]}",
                    destination,
                    b"message_content",  # Would be actual content
                    message_context
                )
                if not message_stored:
                    # Fallback to best available protocol
                    fallback_protocol = await self.protocol_manager.handle_protocol_fallbacks(
                        selected_protocol,
                        destination
                    )
                    selected_protocol = fallback_protocol
            
            # Step 7: Establish protocol connection if needed
            connection_success = await self._ensure_protocol_connection(selected_protocol, destination)
            
            # Step 8: Generate comprehensive routing metadata
            routing_metadata = self._generate_routing_metadata(
                selected_protocol,
                destination,
                message_context,
                path_scores,
                qos_config,
                privacy_config,
                network_conditions
            )
            
            # Step 9: Create and emit receipt
            execution_time_ms = (time.time() - start_time) * 1000
            receipt = self._create_receipt(
                selected_protocol,
                execution_time_ms,
                path_scores,
                network_conditions
            )
            self.receipts.append(receipt)
            
            # Limit receipt history
            if len(self.receipts) > self.max_receipts:
                self.receipts = self.receipts[-self.max_receipts:]
            
            # Update performance tracking
            self.total_response_time += execution_time_ms
            
            logger.info(
                f"Path selected for {destination}: {selected_protocol.value} "
                f"(time: {execution_time_ms:.1f}ms, score: {path_scores.get(selected_protocol.value, 0.0):.3f})"
            )
            
            return selected_protocol, routing_metadata
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Path selection failed for {destination}: {e}")
            
            # Fallback to store-and-forward
            return PathProtocol.STORE_FORWARD, {
                "protocol": PathProtocol.STORE_FORWARD.value,
                "error": str(e),
                "fallback": True,
                "timestamp": time.time()
            }
    
    async def _ensure_protocol_connection(self, protocol: PathProtocol, destination: str) -> bool:
        """Ensure protocol connection is established"""
        try:
            # Check if we need to establish or switch protocols
            connection_status = self.protocol_manager.get_connection_status()
            
            # For now, assume connection establishment is successful
            # In production, this would actually establish the connection
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish {protocol.value} connection to {destination}: {e}")
            return False
    
    def _generate_routing_metadata(
        self,
        protocol: PathProtocol,
        destination: str,
        context: MessageContext,
        path_scores: Dict[str, float],
        qos_config: Dict[str, Any],
        privacy_config: Dict[str, Any],
        network_conditions: NetworkConditions
    ) -> Dict[str, Any]:
        """Generate comprehensive routing metadata"""
        metadata = {
            # Core routing info
            "protocol": protocol.value,
            "destination": destination,
            "timestamp": time.time(),
            "agent_id": self.agent_id,
            
            # Routing decision
            "path_scores": path_scores,
            "selected_score": path_scores.get(protocol.value, 0.0),
            "routing_priority": self.routing_priority.value,
            "energy_mode": self.energy_mode.value,
            
            # QoS configuration
            "qos_config": qos_config,
            
            # Privacy configuration
            "privacy_config": privacy_config,
            
            # Network conditions
            "network_conditions": {
                "internet_available": network_conditions.internet_available,
                "bluetooth_available": network_conditions.bluetooth_available,
                "wifi_connected": network_conditions.wifi_connected,
                "battery_percent": network_conditions.battery_percent,
                "bandwidth_mbps": network_conditions.bandwidth_mbps,
                "latency_ms": network_conditions.latency_ms,
                "reliability_score": network_conditions.reliability_score
            },
            
            # Message context
            "message_context": {
                "size_bytes": context.size_bytes,
                "priority": context.priority,
                "requires_realtime": context.requires_realtime,
                "privacy_required": context.privacy_required,
                "bandwidth_sensitive": context.bandwidth_sensitive
            }
        }
        
        # Add protocol-specific metadata
        if protocol == PathProtocol.BITCHAT:
            metadata.update({
                "max_hops": 7,
                "mesh_routing": True,
                "energy_efficient": True,
                "offline_capable": True
            })
        elif protocol == PathProtocol.BETANET:
            metadata.update({
                "global_reach": True,
                "mixnode_routing": privacy_config.get("mixnode_hops", 0) > 0,
                "high_bandwidth": True
            })
        elif protocol == PathProtocol.SCION:
            metadata.update({
                "multipath": True,
                "path_aware": True,
                "high_performance": True,
                "failover_support": True
            })
        elif protocol == PathProtocol.STORE_FORWARD:
            metadata.update({
                "delay_tolerant": True,
                "guaranteed_delivery": True,
                "energy_minimal": True
            })
        
        return metadata
    
    def _create_receipt(
        self,
        protocol: PathProtocol,
        execution_time_ms: float,
        path_scores: Dict[str, float],
        network_conditions: NetworkConditions
    ) -> Receipt:
        """Create receipt for bounty reviewers"""
        return Receipt(
            chosen_path=protocol.value,
            switch_latency_ms=execution_time_ms,
            reason=self._determine_selection_reason(protocol),
            timestamp=time.time(),
            scion_available="scion" in path_scores and path_scores["scion"] > 0,
            scion_paths=1 if "scion" in path_scores else 0,
            path_scores=path_scores
        )
    
    def _determine_selection_reason(self, protocol: PathProtocol) -> str:
        """Determine reason for protocol selection"""
        if protocol == PathProtocol.SCION:
            return "scion_high_performance"
        elif protocol == PathProtocol.BETANET:
            return "betanet_internet_available"
        elif protocol == PathProtocol.BITCHAT:
            if self.routing_priority == RoutingPriority.OFFLINE_FIRST:
                return "bitchat_offline_first"
            else:
                return "bitchat_peer_nearby"
        elif protocol == PathProtocol.STORE_FORWARD:
            return "store_forward_fallback"
        else:
            return f"{protocol.value}_selected"
    
    # Event handlers for service coordination
    
    def _handle_route_selected(self, event: RoutingEvent) -> None:
        """Handle route selection events"""
        logger.debug(f"Route selected: {event.data.get('selected_protocol')}")
    
    def _handle_network_change(self, event: RoutingEvent) -> None:
        """Handle network change events"""
        logger.info(f"Network change detected: {event.data.get('change_type')}")
        # Could trigger route recalculation here
    
    def _handle_protocol_switch(self, event: RoutingEvent) -> None:
        """Handle protocol switch events"""
        logger.info(f"Protocol switched: {event.data.get('from_protocol')} -> {event.data.get('to_protocol')}")
    
    def _handle_power_update(self, event: RoutingEvent) -> None:
        """Handle power management updates"""
        actions = event.data.get('actions_taken', [])
        if actions:
            logger.info(f"Power management actions taken: {actions}")
            
            # Update energy mode if needed
            if "enabled_ultra_power_save" in actions:
                self.energy_mode = EnergyMode.POWERSAVE
    
    def _handle_circuit_created(self, event: RoutingEvent) -> None:
        """Handle privacy circuit creation"""
        circuit_id = event.data.get('circuit_id')
        hops = event.data.get('hops', 0)
        logger.debug(f"Anonymity circuit created: {circuit_id} ({hops} hops)")
    
    # Original NavigatorAgent compatibility methods
    
    def get_status(self) -> Dict[str, Any]:
        """Get current Navigator status - original interface compatibility"""
        avg_response_time = (
            self.total_response_time / self.request_count if self.request_count > 0 else 0.0
        )
        
        return {
            "agent_id": self.agent_id,
            "routing_priority": self.routing_priority.value,
            "energy_mode": self.energy_mode.value,
            
            # Service status
            "services_initialized": self.service_startup_complete,
            "active_services": len(self.services),
            
            # Performance metrics
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_response_time_ms": avg_response_time,
            "success_rate": 1.0 - (self.error_count / max(1, self.request_count)),
            
            # Recent receipts
            "recent_receipts": len(self.receipts),
            
            # Service-specific status
            "service_status": self._get_service_status()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status from all services"""
        system_status = {
            "facade": self.get_status(),
            "services": {}
        }
        
        # Collect status from each service
        try:
            system_status["services"]["route_selection"] = self.route_selection.get_performance_metrics()
            system_status["services"]["protocol_manager"] = self.protocol_manager.get_connection_status()
            system_status["services"]["network_monitoring"] = self.network_monitoring.get_monitoring_metrics()
            system_status["services"]["qos_manager"] = self.qos_manager.get_qos_statistics()
            system_status["services"]["dtn_handler"] = self.dtn_handler.get_dtn_statistics()
            system_status["services"]["energy_optimizer"] = self.energy_optimizer.get_energy_statistics()
            system_status["services"]["security_mixnode"] = self.security_mixnode.get_security_statistics()
        except Exception as e:
            logger.error(f"Error collecting service status: {e}")
            system_status["error"] = str(e)
        
        return system_status
    
    def _get_service_status(self) -> Dict[str, str]:
        """Get basic status of all services"""
        return {
            "route_selection": "active",
            "protocol_manager": "active",
            "network_monitoring": "active" if self.network_monitoring.running else "inactive",
            "qos_manager": "active",
            "dtn_handler": "active" if self.dtn_handler.running else "inactive",
            "energy_optimizer": "active" if self.energy_optimizer.running else "inactive",
            "security_mixnode": "active" if self.security_mixnode.running else "inactive"
        }
    
    def update_peer_info(self, peer_id: str, peer_info: PeerInfo) -> None:
        """Update peer information across services - original interface compatibility"""
        self.discovered_peers[peer_id] = peer_info
        
        # Update network monitoring with peer connectivity
        self.network_monitoring.peer_connectivity_history[peer_id].append(time.time())
        
        # Update DTN handler with peer connectivity
        self.dtn_handler.update_peer_connectivity(peer_id, True)
        
        logger.debug(f"Updated peer info for {peer_id}")
    
    def update_routing_success(self, protocol: str, destination: str, success: bool) -> None:
        """Update routing success statistics - original interface compatibility"""
        # Update route selection service
        self.route_selection.route_success_rates[f"{protocol}_{destination}"] = (
            0.9 if success else 0.1
        )
        
        # Update protocol manager
        if success:
            self.protocol_manager.switch_success_rates[protocol] = min(
                1.0, self.protocol_manager.switch_success_rates[protocol] + 0.1
            )
        
        logger.debug(f"Updated routing success for {protocol} to {destination}: {success}")
    
    def set_energy_mode(self, mode: EnergyMode) -> None:
        """Change energy management mode - original interface compatibility"""
        old_mode = self.energy_mode
        self.energy_mode = mode
        
        # Update energy optimizer
        self.energy_optimizer.configure_power_profile(
            {
                EnergyMode.POWERSAVE: self.energy_optimizer.PowerProfile.POWER_SAVER,
                EnergyMode.BALANCED: self.energy_optimizer.PowerProfile.BALANCED,
                EnergyMode.PERFORMANCE: self.energy_optimizer.PowerProfile.PERFORMANCE
            }.get(mode, self.energy_optimizer.PowerProfile.BALANCED)
        )
        
        logger.info(f"Energy mode changed from {old_mode.value} to {mode.value}")
    
    def set_routing_priority(self, priority: RoutingPriority) -> None:
        """Change routing priority mode - original interface compatibility"""
        old_priority = self.routing_priority
        self.routing_priority = priority
        
        logger.info(f"Routing priority changed from {old_priority.value} to {priority.value}")
    
    def enable_global_south_mode(self, enabled: bool = True) -> None:
        """Enable/disable Global South optimizations - original interface compatibility"""
        if enabled:
            self.routing_priority = RoutingPriority.OFFLINE_FIRST
            self.energy_mode = EnergyMode.BALANCED
            logger.info("Global South mode enabled - prioritizing offline-first routing")
        else:
            logger.info("Global South mode disabled")
    
    def get_receipts(self, count: int = 100) -> List[Receipt]:
        """Get recent receipts for bounty reviewers - original interface compatibility"""
        return self.receipts[-count:] if count < len(self.receipts) else self.receipts
    
    def cleanup_cache(self) -> None:
        """Clean up expired cache entries - original interface compatibility"""
        try:
            # Clean up QoS manager
            self.qos_manager.cleanup_expired_flows()
            
            # Clean up route selection cache
            self.route_selection._cleanup_path_cache()
            
            logger.debug("Cache cleanup completed")
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")


# Factory function for creating NavigatorFacade instances
def create_navigator_facade(
    agent_id: Optional[str] = None,
    routing_priority: RoutingPriority = RoutingPriority.OFFLINE_FIRST,
    energy_mode: EnergyMode = EnergyMode.BALANCED,
) -> NavigatorFacade:
    """Create and initialize a NavigatorFacade instance"""
    facade = NavigatorFacade(agent_id, routing_priority, energy_mode)
    return facade


# Maintain backward compatibility with original NavigatorAgent
NavigatorAgent = NavigatorFacade  # Alias for backward compatibility