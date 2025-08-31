"""
Fog Coordinator Facade

Provides backwards compatibility for the original FogCoordinator interface
while orchestrating the new service-based architecture internally.

This facade maintains 100% API compatibility while leveraging the extracted services.
"""

import asyncio
from datetime import datetime, UTC
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .interfaces.base_service import EventBus
from .interfaces.service_registry import ServiceRegistry, ServiceFactory, ServiceDependency
from .harvesting.fog_harvesting_service import FogHarvestingService
from .routing.fog_routing_service import FogRoutingService
from .marketplace.fog_marketplace_service import FogMarketplaceService
from .tokenomics.fog_tokenomics_service import FogTokenomicsService
from .networking.fog_networking_service import FogNetworkingService
from .monitoring.fog_monitoring_service import FogMonitoringService
from .configuration.fog_configuration_service import FogConfigurationService


logger = logging.getLogger(__name__)


class FogCoordinatorFacade:
    """
    Backwards-compatible facade for the original FogCoordinator.
    
    Orchestrates the new service-based architecture while maintaining
    the same public interface for existing code.
    """
    
    def __init__(
        self,
        node_id: str,
        config_path: Path | None = None,
        enable_harvesting: bool = True,
        enable_onion_routing: bool = True,
        enable_marketplace: bool = True,
        enable_tokens: bool = True,
    ):
        self.node_id = node_id
        self.config_path = config_path
        self.enable_harvesting = enable_harvesting
        self.enable_onion_routing = enable_onion_routing
        self.enable_marketplace = enable_marketplace
        self.enable_tokens = enable_tokens
        
        # Service orchestration
        self.event_bus = EventBus()
        self.service_registry = ServiceRegistry(self.event_bus)
        self.service_factory = None
        
        # Service references (for backwards compatibility)
        self.harvesting_service: Optional[FogHarvestingService] = None
        self.routing_service: Optional[FogRoutingService] = None
        self.marketplace_service: Optional[FogMarketplaceService] = None
        self.tokenomics_service: Optional[FogTokenomicsService] = None
        self.networking_service: Optional[FogNetworkingService] = None
        self.monitoring_service: Optional[FogMonitoringService] = None
        self.configuration_service: Optional[FogConfigurationService] = None
        
        # Backwards compatibility properties
        self.harvest_manager = None
        self.onion_router = None
        self.marketplace = None
        self.token_system = None
        self.resource_manager = None
        self.quorum_manager = None
        self.sla_tier_manager = None
        self.onion_coordinator = None
        
        # System state
        self.is_running = False
        self.stats = {
            "startup_time": None,
            "devices_harvesting": 0,
            "circuits_active": 0,
            "services_offered": 0,
            "tokens_distributed": 0,
            "hidden_services": 0,
        }
        
        # Load configuration
        self.config = self._load_config()
        
        logger.info(f"FogCoordinatorFacade initialized: {node_id}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load fog computing configuration (backwards compatible)"""
        import json
        
        default_config = {
            "node_id": self.node_id,
            "harvest": {
                "min_battery_percent": 20,
                "max_thermal_temp": 45.0,
                "require_charging": True,
                "require_wifi": True,
                "token_rate_per_hour": 10,
            },
            "onion": {
                "num_guards": 3,
                "circuit_lifetime_hours": 1,
                "default_hops": 3,
                "enable_hidden_services": True,
                "enable_mixnet": True,
                "max_circuits": 50,
            },
            "marketplace": {
                "base_token_rate": 100,
                "enable_spot_pricing": True,
                "enable_hidden_services": True,
            },
            "tokens": {
                "initial_supply": 1000000000,
                "reward_rate_per_hour": 10,
                "staking_apy": 0.05,
                "governance_threshold": 1000000,
            },
            "network": {
                "p2p_port": 7777,
                "api_port": 8888,
                "bootstrap_nodes": [],
                "enable_upnp": True,
            },
            "monitoring": {
                "alert_thresholds": {
                    "cpu_usage_percent": 80.0,
                    "memory_usage_percent": 85.0,
                    "disk_usage_percent": 90.0,
                    "error_rate_threshold": 0.05,
                }
            }
        }
        
        # Load from file if specified
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
                logger.info(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        return default_config
    
    async def start(self) -> bool:
        """Start the fog computing coordinator (orchestrates services)"""
        try:
            logger.info("Starting fog computing system with service orchestration...")
            
            # Initialize service factory
            self.service_factory = ServiceFactory(self.service_registry, self.config)
            
            # Create and register all services
            await self._create_and_register_services()
            
            # Start all services in dependency order
            success = await self.service_registry.start_all_services()
            
            if success:
                # Set up backwards compatibility references
                await self._setup_backwards_compatibility()
                
                self.is_running = True
                self.stats["startup_time"] = datetime.now(UTC)
                
                logger.info("Fog computing system started successfully with service orchestration")
                return True
            else:
                logger.error("Failed to start fog computing services")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start fog coordinator facade: {e}")
            return False
    
    async def _create_and_register_services(self):
        """Create and register all fog computing services"""
        
        # Create configuration service first (no dependencies)
        self.configuration_service = self.service_factory.create_service(
            FogConfigurationService,
            "fog_configuration",
            self.config
        )
        
        # Create monitoring service (depends on configuration)
        self.monitoring_service = self.service_factory.create_service(
            FogMonitoringService,
            "fog_monitoring",
            self.config,
            dependencies=[
                ServiceDependency(FogConfigurationService, required=True)
            ]
        )
        
        # Create tokenomics service (depends on configuration and monitoring)
        if self.enable_tokens:
            self.tokenomics_service = self.service_factory.create_service(
                FogTokenomicsService,
                "fog_tokenomics",
                self.config,
                dependencies=[
                    ServiceDependency(FogConfigurationService, required=True),
                    ServiceDependency(FogMonitoringService, required=True)
                ]
            )
        
        # Create networking service (depends on configuration and monitoring)
        self.networking_service = self.service_factory.create_service(
            FogNetworkingService,
            "fog_networking",
            self.config,
            dependencies=[
                ServiceDependency(FogConfigurationService, required=True),
                ServiceDependency(FogMonitoringService, required=True)
            ]
        )
        
        # Create routing service (depends on networking and monitoring)
        if self.enable_onion_routing:
            self.routing_service = self.service_factory.create_service(
                FogRoutingService,
                "fog_routing",
                self.config,
                dependencies=[
                    ServiceDependency(FogConfigurationService, required=True),
                    ServiceDependency(FogNetworkingService, required=True),
                    ServiceDependency(FogMonitoringService, required=True)
                ]
            )
        
        # Create marketplace service (depends on routing, tokenomics, monitoring)
        if self.enable_marketplace:
            dependencies = [
                ServiceDependency(FogConfigurationService, required=True),
                ServiceDependency(FogMonitoringService, required=True)
            ]
            if self.enable_tokens:
                dependencies.append(ServiceDependency(FogTokenomicsService, required=True))
            if self.enable_onion_routing:
                dependencies.append(ServiceDependency(FogRoutingService, required=False))
            
            self.marketplace_service = self.service_factory.create_service(
                FogMarketplaceService,
                "fog_marketplace",
                self.config,
                dependencies=dependencies
            )
        
        # Create harvesting service (depends on marketplace, tokenomics, monitoring)
        if self.enable_harvesting:
            dependencies = [
                ServiceDependency(FogConfigurationService, required=True),
                ServiceDependency(FogMonitoringService, required=True)
            ]
            if self.enable_tokens:
                dependencies.append(ServiceDependency(FogTokenomicsService, required=True))
            if self.enable_marketplace:
                dependencies.append(ServiceDependency(FogMarketplaceService, required=False))
            
            self.harvesting_service = self.service_factory.create_service(
                FogHarvestingService,
                "fog_harvesting",
                self.config,
                dependencies=dependencies
            )
        
        logger.info("All fog computing services created and registered")
    
    async def _setup_backwards_compatibility(self):
        """Set up backwards compatibility references"""
        try:
            # Map service components to original interface
            if self.harvesting_service:
                self.harvest_manager = self.harvesting_service.harvest_manager
                self.resource_manager = self.harvesting_service.resource_manager
            
            if self.routing_service:
                self.onion_router = self.routing_service.onion_router
                self.onion_coordinator = self.routing_service.onion_coordinator
            
            if self.marketplace_service:
                self.marketplace = self.marketplace_service.marketplace
                self.sla_tier_manager = self.marketplace_service.sla_tier_manager
            
            if self.tokenomics_service:
                self.token_system = self.tokenomics_service.token_system
            
            logger.info("Backwards compatibility references established")
            
        except Exception as e:
            logger.error(f"Failed to set up backwards compatibility: {e}")
    
    async def stop(self):
        """Stop the fog computing coordinator"""
        try:
            logger.info("Stopping fog computing system...")
            
            self.is_running = False
            
            # Stop all services in reverse dependency order
            success = await self.service_registry.stop_all_services()
            
            if success:
                logger.info("Fog computing system stopped successfully")
            else:
                logger.warning("Some services failed to stop gracefully")
                
        except Exception as e:
            logger.error(f"Error stopping fog coordinator facade: {e}")
    
    # ============================================================================
    # BACKWARDS COMPATIBLE PUBLIC API
    # ============================================================================
    
    async def register_mobile_device(
        self, device_id: str, capabilities: Dict[str, Any], initial_state: Dict[str, Any]
    ) -> bool:
        """Register a mobile device for fog computing (backwards compatible)"""
        try:
            if self.harvesting_service:
                success = await self.harvesting_service.register_mobile_device(
                    device_id, capabilities, initial_state
                )
                
                # Also create token account if tokenomics enabled
                if success and self.tokenomics_service:
                    device_key = f"device_{device_id}".encode()
                    await self.tokenomics_service.create_account(device_id, device_key, 0.0)
                
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to register mobile device: {e}")
            return False
    
    async def create_hidden_service(self, ports: Dict[int, int], service_type: str = "web") -> str | None:
        """Create a .fog hidden service (backwards compatible)"""
        try:
            if self.routing_service:
                onion_address = await self.routing_service.create_hidden_service(ports, service_type)
                
                # Register in marketplace if enabled
                if onion_address and self.marketplace_service and service_type in ["web", "api", "database"]:
                    offering_data = {
                        "offering_id": f"hidden_{onion_address.split('.')[0]}",
                        "service_type": service_type.upper().replace("WEB", "STATIC_WEBSITE"),
                        "service_tier": "BASIC",
                        "pricing_model": "HOURLY",
                        "base_price": "0.01",
                        "regions": ["fog_network"],
                        "uptime_guarantee": 99.0,
                    }
                    
                    await self.marketplace_service.register_service_offering(
                        self.node_id, offering_data
                    )
                
                return onion_address
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create hidden service: {e}")
            return None
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive fog system status (backwards compatible)"""
        try:
            status = {
                "node_id": self.node_id,
                "is_running": self.is_running,
                "startup_time": self.stats["startup_time"].isoformat() if self.stats["startup_time"] else None,
                "components": {
                    "harvest_manager": self.harvest_manager is not None,
                    "onion_router": self.onion_router is not None,
                    "marketplace": self.marketplace is not None,
                    "token_system": self.token_system is not None,
                    "resource_manager": self.resource_manager is not None,
                    "quorum_manager": self.quorum_manager is not None,
                    "sla_tier_manager": self.sla_tier_manager is not None,
                    "onion_coordinator": self.onion_coordinator is not None,
                },
                "statistics": self.stats.copy(),
            }
            
            # Add service-specific stats
            if self.harvesting_service:
                status["harvest"] = await self.harvesting_service.get_harvesting_stats()
            
            if self.routing_service:
                status["onion"] = await self.routing_service.get_routing_stats()
            
            if self.marketplace_service:
                status["marketplace"] = await self.marketplace_service.get_marketplace_stats()
            
            if self.tokenomics_service:
                status["tokens"] = await self.tokenomics_service.get_tokenomics_stats()
            
            if self.networking_service:
                status["networking"] = await self.networking_service.get_networking_stats()
            
            if self.monitoring_service:
                status["monitoring"] = await self.monitoring_service.get_monitoring_stats()
            
            # Get service registry status
            status["service_orchestration"] = self.service_registry.get_service_status()
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    async def process_fog_request(self, request_type: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process various types of fog computing requests (backwards compatible)"""
        try:
            if request_type == "compute_task":
                if self.harvesting_service:
                    assigned_device = await self.harvesting_service.assign_compute_task(request_data)
                    return {"success": assigned_device is not None, "assigned_device": assigned_device}
            
            elif request_type == "service_request":
                if self.marketplace_service:
                    contract_id = await self.marketplace_service.submit_service_request(
                        request_data["customer_id"], request_data
                    )
                    return {"success": contract_id is not None, "contract_id": contract_id}
            
            elif request_type == "token_transfer":
                if self.tokenomics_service:
                    success = await self.tokenomics_service.transfer_tokens(
                        request_data["from_account"],
                        request_data["to_account"],
                        request_data["amount"],
                        request_data.get("description", "")
                    )
                    return {"success": success}
            
            elif request_type == "provision_sla_service":
                if self.marketplace_service:
                    result = await self.marketplace_service.provision_sla_service(
                        request_data["service_id"],
                        request_data.get("tier", "bronze"),
                        request_data.get("available_devices", []),
                        request_data.get("service_config", {})
                    )
                    return result
            
            elif request_type == "validate_sla_compliance":
                if self.marketplace_service:
                    result = await self.marketplace_service.validate_sla_compliance(
                        request_data["service_id"], request_data["metrics"]
                    )
                    return result
            
            elif request_type == "submit_privacy_task":
                if self.routing_service:
                    success = await self.routing_service.submit_privacy_aware_task(request_data)
                    return {"success": success, "task_id": request_data["task_id"]}
            
            elif request_type == "send_private_gossip":
                if self.routing_service:
                    message = request_data["message"]
                    if isinstance(message, str):
                        message = message.encode()
                    
                    success = await self.routing_service.send_private_message(
                        request_data["recipient_id"],
                        message,
                        request_data.get("privacy_level", "PRIVATE")
                    )
                    return {"success": success}
            
            return {"success": False, "error": f"Unknown request type: {request_type}"}
            
        except Exception as e:
            logger.error(f"Failed to process fog request: {e}")
            return {"success": False, "error": str(e)}


# Factory function for backwards compatibility
def create_fog_coordinator(
    node_id: str,
    config_path: Path | None = None,
    enable_harvesting: bool = True,
    enable_onion_routing: bool = True,
    enable_marketplace: bool = True,
    enable_tokens: bool = True,
) -> FogCoordinatorFacade:
    """Create a fog coordinator instance (backwards compatible factory)"""
    return FogCoordinatorFacade(
        node_id=node_id,
        config_path=config_path,
        enable_harvesting=enable_harvesting,
        enable_onion_routing=enable_onion_routing,
        enable_marketplace=enable_marketplace,
        enable_tokens=enable_tokens
    )