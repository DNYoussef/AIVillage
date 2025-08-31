"""Routing System Interfaces - Core contracts for navigation services

This module defines the fundamental interfaces and contracts that govern
interaction between routing services in the Navigator system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple
import time

from ..path_policy import (
    EnergyMode, MessageContext, NetworkConditions, PathProtocol,
    PeerInfo, RoutingPriority, SCIONPath
)


class IRouteSelectionService(ABC):
    """Interface for core routing algorithm implementations"""
    
    @abstractmethod
    async def select_optimal_route(
        self,
        destination: str,
        context: MessageContext,
        available_protocols: Optional[List[str]] = None,
        network_conditions: Optional[NetworkConditions] = None
    ) -> Tuple[PathProtocol, Dict[str, float]]:
        """Select the optimal route using routing algorithms"""
        pass
    
    @abstractmethod
    def calculate_path_costs(
        self,
        destination: str,
        protocols: List[str],
        conditions: NetworkConditions
    ) -> Dict[str, float]:
        """Calculate costs for different path options"""
        pass
    
    @abstractmethod
    def optimize_routing(self, performance_metrics: Dict[str, Any]) -> None:
        """Optimize routing based on performance feedback"""
        pass


class IProtocolManagerService(ABC):
    """Interface for protocol switching and connection management"""
    
    @abstractmethod
    async def switch_protocol(
        self,
        from_protocol: PathProtocol,
        to_protocol: PathProtocol,
        destination: str
    ) -> bool:
        """Switch between protocols for active connections"""
        pass
    
    @abstractmethod
    async def manage_connections(self) -> Dict[str, Any]:
        """Manage active protocol connections"""
        pass
    
    @abstractmethod
    async def handle_protocol_fallbacks(
        self,
        failed_protocol: PathProtocol,
        destination: str
    ) -> PathProtocol:
        """Handle protocol failures with fallback selection"""
        pass


class INetworkMonitoringService(ABC):
    """Interface for network condition monitoring and link detection"""
    
    @abstractmethod
    async def monitor_network_links(self) -> NetworkConditions:
        """Monitor current network link conditions"""
        pass
    
    @abstractmethod
    async def detect_link_changes(self) -> Tuple[bool, Dict[str, Any]]:
        """Detect significant network link changes"""
        pass
    
    @abstractmethod
    def assess_link_quality(self, protocol: PathProtocol) -> float:
        """Assess quality score for specific protocol link"""
        pass


class IQoSManagerService(ABC):
    """Interface for Quality of Service management and adaptation"""
    
    @abstractmethod
    async def manage_qos_parameters(
        self,
        protocol: PathProtocol,
        context: MessageContext
    ) -> Dict[str, Any]:
        """Manage QoS parameters for protocol and message context"""
        pass
    
    @abstractmethod
    async def adapt_bandwidth_usage(
        self,
        available_bandwidth: float,
        required_bandwidth: float
    ) -> Dict[str, Any]:
        """Adapt bandwidth usage based on availability"""
        pass
    
    @abstractmethod
    def prioritize_traffic(
        self,
        messages: List[MessageContext]
    ) -> List[MessageContext]:
        """Prioritize message traffic based on QoS policies"""
        pass


class IDTNHandlerService(ABC):
    """Interface for Delay-Tolerant Networking and store-and-forward"""
    
    @abstractmethod
    async def store_message(
        self,
        message_id: str,
        destination: str,
        content: bytes,
        context: MessageContext
    ) -> bool:
        """Store message for later forwarding"""
        pass
    
    @abstractmethod
    async def forward_stored_messages(self) -> Dict[str, int]:
        """Forward stored messages when connectivity available"""
        pass
    
    @abstractmethod
    def manage_storage_buffer(self) -> Dict[str, Any]:
        """Manage message storage buffer and cleanup"""
        pass


class IEnergyOptimizationService(ABC):
    """Interface for energy-aware routing and battery optimization"""
    
    @abstractmethod
    def optimize_for_battery_life(
        self,
        current_level: Optional[int],
        routing_options: List[PathProtocol]
    ) -> List[PathProtocol]:
        """Optimize routing choices for battery conservation"""
        pass
    
    @abstractmethod
    def select_energy_efficient_paths(
        self,
        protocols: List[PathProtocol],
        energy_mode: EnergyMode
    ) -> List[PathProtocol]:
        """Select most energy-efficient paths"""
        pass
    
    @abstractmethod
    async def manage_power_consumption(self) -> Dict[str, Any]:
        """Monitor and manage power consumption"""
        pass


class ISecurityMixnodeService(ABC):
    """Interface for privacy-aware mixnode selection and security"""
    
    @abstractmethod
    async def select_privacy_mixnodes(
        self,
        destination: str,
        privacy_level: float = 0.8
    ) -> List[str]:
        """Select mixnodes for privacy routing"""
        pass
    
    @abstractmethod
    def ensure_routing_privacy(
        self,
        protocol: PathProtocol,
        context: MessageContext
    ) -> Dict[str, Any]:
        """Ensure privacy requirements are met"""
        pass
    
    @abstractmethod
    async def manage_anonymity_circuits(self) -> Dict[str, Any]:
        """Manage anonymous routing circuits"""
        pass


@dataclass
class RoutingEvent:
    """Event emitted by routing services for coordination"""
    event_type: str
    timestamp: float
    source_service: str
    data: Dict[str, Any]
    priority: int = 5
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class IEventBusService(Protocol):
    """Interface for inter-service event communication"""
    
    def publish(self, event: RoutingEvent) -> None:
        """Publish event to interested services"""
        pass
    
    def subscribe(
        self,
        event_type: str,
        callback: callable,
        service_name: str
    ) -> str:
        """Subscribe to specific event types"""
        pass
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        pass


class INavigatorFacadeService(ABC):
    """Interface for the main Navigator facade that coordinates services"""
    
    @abstractmethod
    async def select_path(
        self,
        destination: str,
        message_context: MessageContext,
        available_protocols: Optional[List[str]] = None,
    ) -> Tuple[PathProtocol, Dict[str, Any]]:
        """Main path selection method coordinating all services"""
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status from all services"""
        pass
    
    @abstractmethod
    def update_peer_info(self, peer_id: str, peer_info: PeerInfo) -> None:
        """Update peer information across services"""
        pass


class ServiceDependencyError(Exception):
    """Raised when service dependencies are not met"""
    pass


class RoutingConfigurationError(Exception):
    """Raised when routing configuration is invalid"""
    pass