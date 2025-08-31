"""Protocol Manager Service - Protocol switching and connection management

This service handles protocol switching, connection management, and fallback
handling for BitChat, Betanet, and SCION protocols in the Navigator system.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import time

from ..interfaces.routing_interfaces import IProtocolManagerService, RoutingEvent
from ..events.event_bus import get_event_bus
from ..path_policy import PathProtocol, NetworkConditions, MessageContext

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state for protocol management"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    SWITCHING = "switching"
    FAILED = "failed"
    FALLBACK = "fallback"


@dataclass
class ProtocolConnection:
    """Represents an active protocol connection"""
    protocol: PathProtocol
    destination: str
    state: ConnectionState = ConnectionState.DISCONNECTED
    established_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    connection_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3


class ProtocolManagerService(IProtocolManagerService):
    """Protocol switching and connection management service
    
    Handles:
    - Protocol connection lifecycle management
    - Fast protocol switching (500ms target)
    - Fallback protocol selection and management
    - Connection health monitoring and recovery
    """
    
    def __init__(self):
        self.event_bus = get_event_bus()
        
        # Connection management
        self.active_connections: Dict[str, ProtocolConnection] = {}
        self.connection_pools: Dict[PathProtocol, Set[str]] = {
            protocol: set() for protocol in PathProtocol
        }
        
        # Protocol capabilities and preferences
        self.protocol_capabilities: Dict[PathProtocol, Dict[str, Any]] = {
            PathProtocol.SCION: {
                "supports_multipath": True,
                "supports_realtime": True,
                "supports_privacy": True,
                "requires_internet": True,
                "energy_cost": "medium",
                "typical_latency_ms": 50,
                "reliability": 0.98
            },
            PathProtocol.BETANET: {
                "supports_multipath": False,
                "supports_realtime": True,
                "supports_privacy": True,
                "requires_internet": True,
                "energy_cost": "high",
                "typical_latency_ms": 100,
                "reliability": 0.95
            },
            PathProtocol.BITCHAT: {
                "supports_multipath": True,
                "supports_realtime": False,
                "supports_privacy": False,
                "requires_internet": False,
                "energy_cost": "low",
                "typical_latency_ms": 200,
                "reliability": 0.85
            },
            PathProtocol.STORE_FORWARD: {
                "supports_multipath": False,
                "supports_realtime": False,
                "supports_privacy": False,
                "requires_internet": False,
                "energy_cost": "very_low",
                "typical_latency_ms": 0,  # Delivered when possible
                "reliability": 1.0
            }
        }
        
        # Switch performance tracking
        self.switch_times: Dict[str, List[float]] = defaultdict(list)
        self.switch_success_rates: Dict[str, float] = defaultdict(lambda: 0.9)
        self.target_switch_time_ms = 500
        
        # Fallback hierarchy
        self.fallback_hierarchy: Dict[PathProtocol, List[PathProtocol]] = {
            PathProtocol.SCION: [PathProtocol.BETANET, PathProtocol.BITCHAT, PathProtocol.STORE_FORWARD],
            PathProtocol.BETANET: [PathProtocol.SCION, PathProtocol.BITCHAT, PathProtocol.STORE_FORWARD],
            PathProtocol.BITCHAT: [PathProtocol.BETANET, PathProtocol.SCION, PathProtocol.STORE_FORWARD],
            PathProtocol.STORE_FORWARD: [PathProtocol.BITCHAT, PathProtocol.BETANET, PathProtocol.SCION]
        }
        
        # Connection health monitoring
        self.health_check_interval = 30.0  # 30 seconds
        self.last_health_check = 0.0
        
        logger.info("ProtocolManagerService initialized")
    
    async def switch_protocol(
        self,
        from_protocol: PathProtocol,
        to_protocol: PathProtocol,
        destination: str
    ) -> bool:
        """Switch between protocols for active connections"""
        start_time = time.time()
        connection_key = f"{destination}_{from_protocol.value}"
        
        logger.info(f"Switching protocol for {destination}: {from_protocol.value} -> {to_protocol.value}")
        
        try:
            # Get existing connection
            old_connection = self.active_connections.get(connection_key)
            if not old_connection:
                logger.warning(f"No existing connection found for {connection_key}")
                return await self._establish_new_connection(to_protocol, destination)
            
            # Mark old connection as switching
            old_connection.state = ConnectionState.SWITCHING
            
            # Establish new connection
            new_connection_key = f"{destination}_{to_protocol.value}"
            success = await self._establish_protocol_connection(to_protocol, destination)
            
            if success:
                # Clean up old connection
                await self._cleanup_connection(old_connection)
                del self.active_connections[connection_key]
                
                # Track switch performance
                switch_time_ms = (time.time() - start_time) * 1000
                self._track_switch_performance(from_protocol, to_protocol, switch_time_ms, True)
                
                # Emit switch success event
                self._emit_protocol_event("protocol_switched", {
                    "from_protocol": from_protocol.value,
                    "to_protocol": to_protocol.value,
                    "destination": destination,
                    "switch_time_ms": switch_time_ms,
                    "success": True
                })
                
                logger.info(
                    f"Protocol switch successful: {from_protocol.value} -> {to_protocol.value} "
                    f"in {switch_time_ms:.1f}ms"
                )
                return True
            
            else:
                # Switch failed, restore old connection
                old_connection.state = ConnectionState.CONNECTED
                switch_time_ms = (time.time() - start_time) * 1000
                self._track_switch_performance(from_protocol, to_protocol, switch_time_ms, False)
                
                logger.error(f"Failed to establish {to_protocol.value} connection, keeping {from_protocol.value}")
                return False
        
        except Exception as e:
            logger.error(f"Protocol switch error for {destination}: {e}")
            switch_time_ms = (time.time() - start_time) * 1000
            self._track_switch_performance(from_protocol, to_protocol, switch_time_ms, False)
            return False
    
    async def _establish_new_connection(
        self,
        protocol: PathProtocol,
        destination: str
    ) -> bool:
        """Establish a new protocol connection"""
        connection_key = f"{destination}_{protocol.value}"
        
        if connection_key in self.active_connections:
            return True  # Already connected
        
        return await self._establish_protocol_connection(protocol, destination)
    
    async def _establish_protocol_connection(
        self,
        protocol: PathProtocol,
        destination: str
    ) -> bool:
        """Establish connection for specific protocol"""
        connection_key = f"{destination}_{protocol.value}"
        
        # Create connection object
        connection = ProtocolConnection(
            protocol=protocol,
            destination=destination,
            state=ConnectionState.CONNECTING,
            connection_id=f"{protocol.value}_{hash(destination) % 10000:04d}"
        )
        
        try:
            # Protocol-specific connection logic
            if protocol == PathProtocol.SCION:
                success = await self._establish_scion_connection(connection)
            elif protocol == PathProtocol.BETANET:
                success = await self._establish_betanet_connection(connection)
            elif protocol == PathProtocol.BITCHAT:
                success = await self._establish_bitchat_connection(connection)
            elif protocol == PathProtocol.STORE_FORWARD:
                success = await self._establish_store_forward_connection(connection)
            else:
                success = False
            
            if success:
                connection.state = ConnectionState.CONNECTED
                connection.established_at = time.time()
                connection.last_activity = time.time()
                
                # Add to active connections and pools
                self.active_connections[connection_key] = connection
                self.connection_pools[protocol].add(connection_key)
                
                logger.debug(f"Established {protocol.value} connection to {destination}")
                return True
            else:
                connection.state = ConnectionState.FAILED
                return False
        
        except Exception as e:
            logger.error(f"Failed to establish {protocol.value} connection to {destination}: {e}")
            connection.state = ConnectionState.FAILED
            return False
    
    async def _establish_scion_connection(self, connection: ProtocolConnection) -> bool:
        """Establish SCION protocol connection"""
        # Simulate SCION gateway connection
        await asyncio.sleep(0.1)  # Simulate connection time
        
        # Check if SCION gateway is available (simplified check)
        connection.metadata = {
            "gateway_status": "connected",
            "available_paths": 3,
            "multipath_enabled": True,
            "path_selection": "performance_optimized"
        }
        
        return True  # Assume success for simulation
    
    async def _establish_betanet_connection(self, connection: ProtocolConnection) -> bool:
        """Establish Betanet protocol connection"""
        # Simulate Betanet HTX connection
        await asyncio.sleep(0.15)  # Simulate connection time
        
        connection.metadata = {
            "transport": "HTX",
            "encryption": "enabled",
            "mixnodes_available": True,
            "bandwidth_tier": "standard"
        }
        
        return True  # Assume success for simulation
    
    async def _establish_bitchat_connection(self, connection: ProtocolConnection) -> bool:
        """Establish BitChat protocol connection"""
        # Simulate Bluetooth mesh connection
        await asyncio.sleep(0.05)  # Fast Bluetooth connection
        
        connection.metadata = {
            "transport": "bluetooth_mesh",
            "hop_count": 2,
            "mesh_topology": "adaptive",
            "store_forward_enabled": True
        }
        
        return True  # Assume success for simulation
    
    async def _establish_store_forward_connection(self, connection: ProtocolConnection) -> bool:
        """Establish store-and-forward connection"""
        # Store-and-forward is always available
        await asyncio.sleep(0.01)  # Minimal setup time
        
        connection.metadata = {
            "storage_enabled": True,
            "queue_priority": "normal",
            "max_storage_time_hours": 24,
            "delivery_mode": "opportunistic"
        }
        
        return True  # Always succeeds
    
    async def manage_connections(self) -> Dict[str, Any]:
        """Manage active protocol connections"""
        current_time = time.time()
        
        # Periodic health checks
        if current_time - self.last_health_check > self.health_check_interval:
            await self._perform_health_checks()
            self.last_health_check = current_time
        
        # Clean up stale connections
        stale_connections = []
        for key, connection in self.active_connections.items():
            if current_time - connection.last_activity > 300:  # 5 minutes idle
                stale_connections.append(key)
        
        for key in stale_connections:
            await self._cleanup_stale_connection(key)
        
        # Connection management stats
        connection_stats = {
            "total_connections": len(self.active_connections),
            "connections_by_protocol": {
                protocol.value: len(connections) 
                for protocol, connections in self.connection_pools.items()
            },
            "connection_states": self._get_connection_state_counts(),
            "cleanup_actions": len(stale_connections)
        }
        
        # Emit management event
        self._emit_protocol_event("connections_managed", connection_stats)
        
        return connection_stats
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on active connections"""
        logger.debug("Performing connection health checks")
        
        unhealthy_connections = []
        for key, connection in self.active_connections.items():
            try:
                is_healthy = await self._check_connection_health(connection)
                if not is_healthy:
                    unhealthy_connections.append(key)
            except Exception as e:
                logger.error(f"Health check failed for {key}: {e}")
                unhealthy_connections.append(key)
        
        # Handle unhealthy connections
        for key in unhealthy_connections:
            connection = self.active_connections[key]
            logger.warning(f"Connection {key} is unhealthy, attempting recovery")
            
            # Try to recover connection
            recovered = await self._recover_connection(connection)
            if not recovered:
                # Mark for fallback if recovery fails
                connection.state = ConnectionState.FAILED
                await self._trigger_fallback(connection)
    
    async def _check_connection_health(self, connection: ProtocolConnection) -> bool:
        """Check health of specific connection"""
        # Protocol-specific health checks
        if connection.protocol == PathProtocol.SCION:
            return await self._check_scion_health(connection)
        elif connection.protocol == PathProtocol.BETANET:
            return await self._check_betanet_health(connection)
        elif connection.protocol == PathProtocol.BITCHAT:
            return await self._check_bitchat_health(connection)
        elif connection.protocol == PathProtocol.STORE_FORWARD:
            return True  # Store-and-forward is always healthy
        
        return False
    
    async def _check_scion_health(self, connection: ProtocolConnection) -> bool:
        """Check SCION connection health"""
        # Simulate SCION health check
        await asyncio.sleep(0.05)
        
        # Check if gateway is responsive and paths are available
        gateway_responsive = True  # Simulate
        paths_available = connection.metadata.get("available_paths", 0) > 0
        
        return gateway_responsive and paths_available
    
    async def _check_betanet_health(self, connection: ProtocolConnection) -> bool:
        """Check Betanet connection health"""
        # Simulate Betanet health check
        await asyncio.sleep(0.05)
        
        # Check if HTX transport is responsive
        transport_ok = True  # Simulate
        encryption_ok = connection.metadata.get("encryption") == "enabled"
        
        return transport_ok and encryption_ok
    
    async def _check_bitchat_health(self, connection: ProtocolConnection) -> bool:
        """Check BitChat connection health"""
        # Simulate BitChat health check
        await asyncio.sleep(0.02)
        
        # Check if mesh connectivity is maintained
        mesh_connected = True  # Simulate
        hop_count = connection.metadata.get("hop_count", 999)
        
        return mesh_connected and hop_count <= 7  # Within BitChat hop limit
    
    async def _recover_connection(self, connection: ProtocolConnection) -> bool:
        """Attempt to recover failed connection"""
        logger.info(f"Attempting to recover {connection.protocol.value} connection to {connection.destination}")
        
        connection.retry_count += 1
        if connection.retry_count > connection.max_retries:
            logger.warning(f"Max retries exceeded for {connection.protocol.value} connection")
            return False
        
        # Attempt reconnection
        connection.state = ConnectionState.CONNECTING
        success = await self._establish_protocol_connection(connection.protocol, connection.destination)
        
        if success:
            connection.retry_count = 0  # Reset retry count on success
            logger.info(f"Successfully recovered {connection.protocol.value} connection")
            return True
        
        return False
    
    async def handle_protocol_fallbacks(
        self,
        failed_protocol: PathProtocol,
        destination: str
    ) -> PathProtocol:
        """Handle protocol failures with fallback selection"""
        logger.info(f"Handling fallback for failed {failed_protocol.value} to {destination}")
        
        # Get fallback options for the failed protocol
        fallback_options = self.fallback_hierarchy.get(failed_protocol, [])
        
        # Try each fallback option in order
        for fallback_protocol in fallback_options:
            logger.debug(f"Trying fallback protocol: {fallback_protocol.value}")
            
            # Check if fallback protocol can be established
            if await self._can_establish_protocol(fallback_protocol, destination):
                success = await self._establish_protocol_connection(fallback_protocol, destination)
                
                if success:
                    logger.info(f"Fallback successful: {failed_protocol.value} -> {fallback_protocol.value}")
                    
                    # Emit fallback success event
                    self._emit_protocol_event("fallback_succeeded", {
                        "failed_protocol": failed_protocol.value,
                        "fallback_protocol": fallback_protocol.value,
                        "destination": destination
                    })
                    
                    return fallback_protocol
        
        # If all fallbacks fail, use store-and-forward as last resort
        logger.warning(f"All fallbacks failed for {failed_protocol.value}, using store-and-forward")
        
        await self._establish_protocol_connection(PathProtocol.STORE_FORWARD, destination)
        
        self._emit_protocol_event("fallback_last_resort", {
            "failed_protocol": failed_protocol.value,
            "fallback_protocol": PathProtocol.STORE_FORWARD.value,
            "destination": destination
        })
        
        return PathProtocol.STORE_FORWARD
    
    async def _can_establish_protocol(
        self,
        protocol: PathProtocol,
        destination: str
    ) -> bool:
        """Check if protocol can be established for destination"""
        capabilities = self.protocol_capabilities[protocol]
        
        # Check basic requirements
        if capabilities.get("requires_internet") and not self._is_internet_available():
            return False
        
        # Protocol-specific checks
        if protocol == PathProtocol.BITCHAT and not self._is_bluetooth_available():
            return False
        
        if protocol == PathProtocol.SCION and not self._is_scion_available():
            return False
        
        return True
    
    def _is_internet_available(self) -> bool:
        """Check if internet is available (simplified)"""
        return True  # Simplified check
    
    def _is_bluetooth_available(self) -> bool:
        """Check if Bluetooth is available (simplified)"""
        return True  # Simplified check
    
    def _is_scion_available(self) -> bool:
        """Check if SCION is available (simplified)"""
        return True  # Simplified check
    
    async def _trigger_fallback(self, connection: ProtocolConnection) -> None:
        """Trigger fallback for failed connection"""
        fallback_protocol = await self.handle_protocol_fallbacks(
            connection.protocol,
            connection.destination
        )
        
        # Update connection state
        connection.state = ConnectionState.FALLBACK
        connection.metadata["fallback_protocol"] = fallback_protocol.value
    
    async def _cleanup_connection(self, connection: ProtocolConnection) -> None:
        """Clean up protocol connection"""
        logger.debug(f"Cleaning up {connection.protocol.value} connection to {connection.destination}")
        
        # Protocol-specific cleanup
        if connection.protocol == PathProtocol.SCION:
            await self._cleanup_scion_connection(connection)
        elif connection.protocol == PathProtocol.BETANET:
            await self._cleanup_betanet_connection(connection)
        elif connection.protocol == PathProtocol.BITCHAT:
            await self._cleanup_bitchat_connection(connection)
        
        connection.state = ConnectionState.DISCONNECTED
    
    async def _cleanup_scion_connection(self, connection: ProtocolConnection) -> None:
        """Clean up SCION connection"""
        # Simulate SCION cleanup
        await asyncio.sleep(0.01)
    
    async def _cleanup_betanet_connection(self, connection: ProtocolConnection) -> None:
        """Clean up Betanet connection"""
        # Simulate Betanet cleanup
        await asyncio.sleep(0.01)
    
    async def _cleanup_bitchat_connection(self, connection: ProtocolConnection) -> None:
        """Clean up BitChat connection"""
        # Simulate BitChat cleanup
        await asyncio.sleep(0.005)
    
    async def _cleanup_stale_connection(self, connection_key: str) -> None:
        """Clean up stale connection"""
        if connection_key not in self.active_connections:
            return
        
        connection = self.active_connections[connection_key]
        await self._cleanup_connection(connection)
        
        # Remove from pools and active connections
        self.connection_pools[connection.protocol].discard(connection_key)
        del self.active_connections[connection_key]
        
        logger.info(f"Cleaned up stale connection: {connection_key}")
    
    def _track_switch_performance(
        self,
        from_protocol: PathProtocol,
        to_protocol: PathProtocol,
        switch_time_ms: float,
        success: bool
    ) -> None:
        """Track protocol switch performance"""
        switch_key = f"{from_protocol.value}_to_{to_protocol.value}"
        
        # Track switch times
        self.switch_times[switch_key].append(switch_time_ms)
        if len(self.switch_times[switch_key]) > 100:  # Keep last 100 switches
            self.switch_times[switch_key].pop(0)
        
        # Update success rates using exponential moving average
        alpha = 0.1
        current_rate = self.switch_success_rates[switch_key]
        self.switch_success_rates[switch_key] = (
            alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
        )
    
    def _get_connection_state_counts(self) -> Dict[str, int]:
        """Get counts of connections by state"""
        state_counts = defaultdict(int)
        for connection in self.active_connections.values():
            state_counts[connection.state.value] += 1
        return dict(state_counts)
    
    def _emit_protocol_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit protocol management event"""
        event = RoutingEvent(
            event_type=event_type,
            timestamp=time.time(),
            source_service="ProtocolManagerService",
            data=data
        )
        self.event_bus.publish(event)
    
    def get_protocol_capabilities(self, protocol: PathProtocol) -> Dict[str, Any]:
        """Get capabilities for specific protocol"""
        return self.protocol_capabilities.get(protocol, {})
    
    def get_switch_performance_metrics(self) -> Dict[str, Any]:
        """Get protocol switch performance metrics"""
        avg_switch_times = {}
        for switch_key, times in self.switch_times.items():
            if times:
                avg_switch_times[switch_key] = sum(times) / len(times)
        
        return {
            "average_switch_times_ms": avg_switch_times,
            "switch_success_rates": dict(self.switch_success_rates),
            "target_switch_time_ms": self.target_switch_time_ms,
            "switches_within_target": {
                key: sum(1 for t in times if t <= self.target_switch_time_ms) / len(times) * 100
                for key, times in self.switch_times.items() if times
            }
        }
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            "total_connections": len(self.active_connections),
            "connections_by_protocol": {
                protocol.value: len(connections)
                for protocol, connections in self.connection_pools.items()
            },
            "connection_states": self._get_connection_state_counts(),
            "active_destinations": len(set(
                conn.destination for conn in self.active_connections.values()
            ))
        }