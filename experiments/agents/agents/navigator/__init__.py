"""Navigator Package - Service-oriented routing architecture

This package provides intelligent dual-path routing for BitChat/Betanet networks
with a service-oriented architecture that separates concerns while maintaining
backward compatibility with the original PathPolicy interface.

Architecture:
- NavigatorFacade: Main interface maintaining PathPolicy compatibility
- RouteSelectionService: Core routing algorithms (Dijkstra, A*, mesh routing)
- ProtocolManagerService: Protocol switching and connection management
- NetworkMonitoringService: Network condition monitoring and change detection
- QoSManagerService: Quality of service management and traffic prioritization
- DTNHandlerService: Delay-tolerant networking and store-and-forward
- EnergyOptimizationService: Battery-aware routing and power management
- SecurityMixnodeService: Privacy-aware mixnode selection and security
- EventBusService: Inter-service communication and coordination
"""

from .navigator_facade import NavigatorFacade, NavigatorAgent, create_navigator_facade
from .path_policy import (
    # Core enums and types
    PathProtocol,
    EnergyMode,
    RoutingPriority,
    # Data classes
    NetworkConditions,
    MessageContext,
    PeerInfo,
    Receipt,
    # Utility classes
    LinkChangeDetector,
)

# Service interfaces (for advanced usage)
from .interfaces.routing_interfaces import (
    IRouteSelectionService,
    IProtocolManagerService,
    INetworkMonitoringService,
    IQoSManagerService,
    IDTNHandlerService,
    IEnergyOptimizationService,
    ISecurityMixnodeService,
    INavigatorFacadeService,
)

# Event system
from .events.event_bus import EventBusService, get_event_bus, initialize_event_bus

__version__ = "2.0.0"
__author__ = "AI Village Navigation Team"

# Main exports for external use
__all__ = [
    # Primary interface (backward compatible)
    "NavigatorAgent",
    "NavigatorFacade",
    "create_navigator_facade",
    # Core types and enums
    "PathProtocol",
    "EnergyMode",
    "RoutingPriority",
    "NetworkConditions",
    "MessageContext",
    "PeerInfo",
    "Receipt",
    "LinkChangeDetector",
    # Service interfaces (advanced usage)
    "IRouteSelectionService",
    "IProtocolManagerService",
    "INetworkMonitoringService",
    "IQoSManagerService",
    "IDTNHandlerService",
    "IEnergyOptimizationService",
    "ISecurityMixnodeService",
    "INavigatorFacadeService",
    # Event system
    "EventBusService",
    "get_event_bus",
    "initialize_event_bus",
]

# Service registry for dynamic access
AVAILABLE_SERVICES = [
    "RouteSelectionService",
    "ProtocolManagerService",
    "NetworkMonitoringService",
    "QoSManagerService",
    "DTNHandlerService",
    "EnergyOptimizationService",
    "SecurityMixnodeService",
]

# Architecture metrics
ARCHITECTURE_METRICS = {
    "original_file_size_lines": 1438,
    "services_extracted": 7,
    "max_service_size_lines": 250,
    "coupling_reduction_target": "<20.0",
    "performance_target_maintained": True,
    "backward_compatibility": True,
}


def get_architecture_info() -> dict:
    """Get information about the Navigator architecture"""
    return {
        "version": __version__,
        "architecture": "Service-Oriented with Event-Driven Coordination",
        "services": AVAILABLE_SERVICES,
        "metrics": ARCHITECTURE_METRICS,
        "primary_interface": "NavigatorFacade (NavigatorAgent alias)",
        "event_system": "EventBusService with publish/subscribe model",
        "backward_compatibility": "Full compatibility with original PathPolicy interface",
    }


def get_service_overview() -> dict:
    """Get overview of all navigation services"""
    return {
        "RouteSelectionService": {
            "purpose": "Core routing algorithms",
            "algorithms": ["Dijkstra", "A*", "Mesh routing", "Multi-hop optimization"],
            "size_target": "200-250 lines",
        },
        "ProtocolManagerService": {
            "purpose": "Protocol switching and connection management",
            "protocols": ["BitChat", "Betanet", "SCION"],
            "size_target": "180-220 lines",
        },
        "NetworkMonitoringService": {
            "purpose": "Network condition monitoring and link detection",
            "features": ["Link quality assessment", "Change detection", "Performance tracking"],
            "size_target": "150-180 lines",
        },
        "QoSManagerService": {
            "purpose": "Quality of service management",
            "features": ["Traffic prioritization", "Bandwidth management", "SLA enforcement"],
            "size_target": "120-150 lines",
        },
        "DTNHandlerService": {
            "purpose": "Delay-tolerant networking",
            "features": ["Message storage", "Opportunistic forwarding", "Buffer management"],
            "size_target": "100-130 lines",
        },
        "EnergyOptimizationService": {
            "purpose": "Battery-aware routing",
            "features": ["Power management", "Energy-efficient paths", "Thermal management"],
            "size_target": "100-120 lines",
        },
        "SecurityMixnodeService": {
            "purpose": "Privacy-aware routing",
            "features": ["Mixnode selection", "Anonymity circuits", "Privacy enforcement"],
            "size_target": "80-100 lines",
        },
    }


# Quick start example
def quick_start_example():
    """Example of how to use the new Navigator architecture"""
    example_code = """
    # Simple usage (backward compatible)
    from navigator import NavigatorAgent, MessageContext
    
    navigator = NavigatorAgent()
    await navigator.initialize()
    
    # Select optimal route
    context = MessageContext(size_bytes=1024, priority=7, requires_realtime=True)
    protocol, metadata = await navigator.select_path("destination_peer", context)
    
    print(f"Selected protocol: {protocol.value}")
    print(f"Route metadata: {metadata}")
    
    # Advanced usage with direct service access
    from navigator import NavigatorFacade
    
    navigator = NavigatorFacade()
    await navigator.initialize()
    
    # Access specific services
    energy_stats = navigator.energy_optimizer.get_energy_statistics()
    network_conditions = await navigator.network_monitoring.monitor_network_links()
    
    # Configure services
    navigator.energy_optimizer.configure_power_profile(PowerProfile.POWER_SAVER)
    navigator.security_mixnode.configure_security_policies(
        blocked_countries={"XX"}, 
        min_bandwidth_mbps=2.0
    )
    """

    return example_code


if __name__ == "__main__":
    # Print architecture overview when module is run directly
    import json

    print("Navigator Package Architecture Overview")
    print("=" * 50)
    print(json.dumps(get_architecture_info(), indent=2))

    print("\nService Overview")
    print("=" * 50)
    print(json.dumps(get_service_overview(), indent=2))

    print("\nQuick Start Example")
    print("=" * 50)
    print(quick_start_example())
