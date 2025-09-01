"""
Fog Computing Integration Coordinator

Orchestrates the complete fog computing system by integrating:
- Mobile compute harvesting
- Onion routing privacy layer
- Fog marketplace for services
- Token economics and rewards
- P2P networking (BitChat/Betanet)
- Hidden service hosting

This is the main entry point for the fog computing network.
"""

import asyncio
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any

# Import fog computing components
from ..compute.harvest_manager import FogHarvestManager, HarvestPolicy
from ..edge.mobile.resource_manager import MobileResourceManager
from ..marketplace.fog_marketplace import FogMarketplace, ServiceTier, ServiceType
from ..privacy.onion_routing import NodeType, OnionRouter
from ..quorum.quorum_manager import QuorumManager
from ..scheduler.enhanced_sla_tiers import EnhancedSLATierManager, SLAMetrics, SLATier
from ..tokenomics.fog_token_system import FogTokenSystem

# Privacy integration imports moved to avoid circular dependency

logger = logging.getLogger(__name__)


class FogCoordinator:
    """
    Master coordinator for the AI Village fog computing system.

    Integrates all fog computing components into a cohesive system that:
    - Harvests idle mobile compute during charging
    - Routes traffic anonymously through onion routing
    - Provides AWS-alternative marketplace services
    - Rewards contributors with tokenomics
    - Hosts censorship-resistant hidden services
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

        # Component instances
        self.harvest_manager: FogHarvestManager | None = None
        self.onion_router: OnionRouter | None = None
        self.marketplace: FogMarketplace | None = None
        self.token_system: FogTokenSystem | None = None
        self.resource_manager: MobileResourceManager | None = None
        self.quorum_manager: QuorumManager | None = None
        self.sla_tier_manager: EnhancedSLATierManager | None = None
        self.onion_coordinator: "FogOnionCoordinator | None" = None

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

        logger.info(f"FogCoordinator initialized: {node_id}")

    def _load_config(self) -> dict[str, Any]:
        """Load fog computing configuration"""

        default_config = {
            # Harvest configuration
            "harvest": {
                "min_battery_percent": 20,
                "max_thermal_temp": 45.0,
                "require_charging": True,
                "require_wifi": True,
                "token_rate_per_hour": 10,
            },
            # Onion routing configuration
            "onion": {"num_guards": 3, "circuit_lifetime_hours": 1, "default_hops": 3, "enable_hidden_services": True},
            # Marketplace configuration
            "marketplace": {"base_token_rate": 100, "enable_spot_pricing": True, "enable_hidden_services": True},
            # Token system configuration
            "tokens": {
                "initial_supply": 1000000000,  # 1 billion
                "reward_rate_per_hour": 10,
                "staking_apy": 0.05,
                "governance_threshold": 1000000,
            },
            # Network configuration
            "network": {"p2p_port": 7777, "api_port": 8888, "bootstrap_nodes": [], "enable_upnp": True},
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
        """Start the fog computing coordinator"""

        try:
            logger.info("Starting fog computing system...")

            # Initialize token system first (needed by other components)
            if self.enable_tokens:
                await self._initialize_token_system()

            # Initialize components in parallel
            tasks = []

            if self.enable_harvesting:
                tasks.append(self._initialize_harvest_manager())

            if self.enable_onion_routing:
                tasks.append(self._initialize_onion_router())

            if self.enable_marketplace:
                tasks.append(self._initialize_marketplace())

            # Initialize mobile resource manager
            tasks.append(self._initialize_resource_manager())

            # Initialize quorum management and SLA tiers
            tasks.append(self._initialize_quorum_manager())
            tasks.append(self._initialize_sla_tier_manager())

            # Initialize onion coordinator if onion routing is enabled
            if self.enable_onion_routing:
                tasks.append(self._initialize_onion_coordinator())

            # Wait for all components to initialize
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for initialization failures
            failed_components = []
            component_names = ["harvest", "onion", "marketplace", "resource", "quorum", "sla_tiers"]
            if self.enable_onion_routing:
                component_names.append("onion_coordinator")
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    component_name = component_names[i] if i < len(component_names) else f"component_{i}"
                    failed_components.append(component_name)
                    logger.error(f"Failed to initialize {component_name}: {result}")

            if failed_components:
                logger.error(f"Failed to initialize components: {failed_components}")
                return False

            # Connect components
            await self._connect_components()

            # Start background tasks
            await self._start_background_tasks()

            self.is_running = True
            self.stats["startup_time"] = datetime.now(UTC)

            logger.info("Fog computing system started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start fog coordinator: {e}")
            return False

    async def _initialize_token_system(self):
        """Initialize the token system"""
        config = self.config["tokens"]

        self.token_system = FogTokenSystem(
            initial_supply=config["initial_supply"],
            reward_rate_per_hour=config["reward_rate_per_hour"],
            staking_apy=config["staking_apy"],
            governance_threshold=config["governance_threshold"],
        )

        # Create system accounts
        system_key = b"system_key_placeholder"  # In production, use proper crypto
        await self.token_system.create_account("system", system_key, 0)
        await self.token_system.create_account("treasury", system_key, config["initial_supply"] * 0.1)

        logger.info("Token system initialized")

    async def _initialize_harvest_manager(self):
        """Initialize the compute harvest manager"""
        config = self.config["harvest"]

        harvest_policy = HarvestPolicy(
            min_battery_percent=config["min_battery_percent"],
            max_cpu_temp=config["max_thermal_temp"],
            require_charging=config["require_charging"],
            require_wifi=config["require_wifi"],
        )

        self.harvest_manager = FogHarvestManager(
            node_id=self.node_id, policy=harvest_policy, token_rate_per_hour=config["token_rate_per_hour"]
        )

        logger.info("Harvest manager initialized")

    async def _initialize_onion_router(self):
        """Initialize the onion routing system"""
        config = self.config["onion"]

        # Determine node types based on configuration
        node_types = {NodeType.MIDDLE}  # All nodes are middle relays by default

        # Add additional types based on capabilities
        if config.get("enable_exit", False):
            node_types.add(NodeType.EXIT)

        if config.get("enable_guard", True):  # Most fog nodes can be guards
            node_types.add(NodeType.GUARD)

        self.onion_router = OnionRouter(
            node_id=self.node_id,
            node_types=node_types,
            enable_hidden_services=config["enable_hidden_services"],
            num_guards=config["num_guards"],
            circuit_lifetime_hours=config["circuit_lifetime_hours"],
        )

        # Fetch network consensus
        await self.onion_router.fetch_consensus()

        logger.info(f"Onion router initialized with types: {node_types}")

    async def _initialize_marketplace(self):
        """Initialize the fog marketplace"""
        config = self.config["marketplace"]

        self.marketplace = FogMarketplace(
            marketplace_id=f"fog-market-{self.node_id}",
            base_token_rate=config["base_token_rate"],
            enable_hidden_services=config["enable_hidden_services"],
            enable_spot_pricing=config["enable_spot_pricing"],
        )

        logger.info("Marketplace initialized")

    async def _initialize_resource_manager(self):
        """Initialize the mobile resource manager"""
        self.resource_manager = MobileResourceManager(
            harvest_enabled=self.enable_harvesting, token_rewards_enabled=self.enable_tokens
        )

        logger.info("Resource manager initialized")

    async def _initialize_quorum_manager(self):
        """Initialize the quorum manager for infrastructure diversity"""
        self.quorum_manager = QuorumManager()
        logger.info("Quorum manager initialized")

    async def _initialize_sla_tier_manager(self):
        """Initialize the enhanced SLA tier manager"""
        self.sla_tier_manager = EnhancedSLATierManager(self.quorum_manager)
        logger.info("SLA tier manager initialized")

    async def _initialize_onion_coordinator(self):
        """Initialize the fog onion coordinator"""
        from .fog_onion_coordinator import FogOnionCoordinator, PrivacyLevel

        self.onion_coordinator = FogOnionCoordinator(
            node_id=f"onion-{self.node_id}",
            fog_coordinator=self,
            enable_mixnet=True,
            default_privacy_level=PrivacyLevel.PRIVATE,
            max_circuits=50,
        )

        success = await self.onion_coordinator.start()
        if success:
            logger.info("Fog onion coordinator initialized")
        else:
            raise RuntimeError("Failed to start fog onion coordinator")

    async def _connect_components(self):
        """Connect fog computing components together"""

        # Connect resource manager to harvest manager
        if self.resource_manager and self.harvest_manager:
            await self.resource_manager.set_p2p_coordinator(self.harvest_manager)

        # Connect resource manager to marketplace
        if self.resource_manager and self.marketplace:
            await self.resource_manager.set_marketplace_client(self.marketplace)

        # Connect harvest manager to token system for rewards
        if self.harvest_manager and self.token_system:
            self.harvest_manager.token_system = self.token_system

        logger.info("Components connected successfully")

    async def _start_background_tasks(self):
        """Start background maintenance tasks"""

        # Start circuit rotation for onion router
        if self.onion_router:
            asyncio.create_task(self._circuit_rotation_task())

        # Start reward distribution task
        if self.token_system and self.harvest_manager:
            asyncio.create_task(self._reward_distribution_task())

        # Start marketplace pricing updates
        if self.marketplace:
            asyncio.create_task(self._marketplace_update_task())

        # Start statistics collection
        asyncio.create_task(self._stats_collection_task())

        # Start SLA monitoring task
        if self.sla_tier_manager:
            asyncio.create_task(self._sla_monitoring_task())

        logger.info("Background tasks started")

    async def _circuit_rotation_task(self):
        """Background task to rotate onion circuits"""
        while self.is_running:
            try:
                if self.onion_router:
                    rotated = await self.onion_router.rotate_circuits()
                    if rotated > 0:
                        logger.debug(f"Rotated {rotated} onion circuits")

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Circuit rotation error: {e}")
                await asyncio.sleep(60)

    async def _reward_distribution_task(self):
        """Background task to distribute token rewards"""
        while self.is_running:
            try:
                if self.harvest_manager and self.token_system:
                    # Get network stats
                    stats = await self.harvest_manager.get_network_stats()

                    # Process pending rewards
                    active_device_count = stats.get("active_devices", 0)
                    if isinstance(active_device_count, int) and active_device_count > 0:
                        # This would integrate with actual contribution tracking
                        logger.info(f"Processing rewards for {active_device_count} active devices")
                        pass
                    else:
                        logger.debug("No active devices found for reward distribution")

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Reward distribution error: {e}")
                await asyncio.sleep(600)

    async def _marketplace_update_task(self):
        """Background task to update marketplace pricing"""
        while self.is_running:
            try:
                if self.marketplace:
                    # Update dynamic pricing for all service types
                    for service_type in ServiceType:
                        await self.marketplace._update_dynamic_pricing(service_type)

                await asyncio.sleep(900)  # Update every 15 minutes

            except Exception as e:
                logger.error(f"Marketplace update error: {e}")
                await asyncio.sleep(300)

    async def _stats_collection_task(self):
        """Background task to collect system statistics"""
        while self.is_running:
            try:
                # Update stats from all components
                if self.harvest_manager:
                    harvest_stats = await self.harvest_manager.get_network_stats()
                    self.stats["devices_harvesting"] = harvest_stats.get("active_devices", 0)

                if self.onion_router:
                    onion_stats = self.onion_router.get_stats()
                    self.stats["circuits_active"] = onion_stats.get("active_circuits", 0)
                    self.stats["hidden_services"] = onion_stats.get("hidden_services", 0)

                if self.marketplace:
                    market_stats = self.marketplace.get_market_stats()
                    self.stats["services_offered"] = market_stats.total_offerings

                if self.token_system:
                    token_stats = self.token_system.get_network_stats()
                    self.stats["tokens_distributed"] = token_stats["total_rewards_distributed"]

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Stats collection error: {e}")
                await asyncio.sleep(60)

    async def _sla_monitoring_task(self):
        """Background task to monitor SLA compliance and diversity requirements"""
        while self.is_running:
            try:
                if self.sla_tier_manager:
                    # Get all active services
                    all_services = self.sla_tier_manager.get_all_services_status()

                    # Check compliance for each service
                    for tier_name, services in all_services.get("services_by_tier", {}).items():
                        for service in services:
                            service_id = service["service_id"]

                            # Mock current metrics (in production, collect from actual monitoring)
                            current_metrics = SLAMetrics(
                                p95_latency_ms=100.0,  # Would come from actual metrics
                                uptime_percentage=99.5,
                                error_rate_percentage=0.05,
                                throughput_ops_per_second=1000.0,
                            )

                            # Validate SLA compliance
                            compliance_result = await self.sla_tier_manager.validate_sla_compliance(
                                service_id, current_metrics
                            )

                            if not compliance_result["compliant"]:
                                logger.warning(
                                    f"SLA violations for service {service_id}: {compliance_result['violations']}"
                                )

                                # For Gold tier, attempt rebalancing if diversity is compromised
                                if service.get("tier") == "gold" and not compliance_result.get("diversity_valid", True):
                                    logger.info(f"Attempting rebalance for Gold tier service {service_id}")
                                    # In production, would get available devices from device registry
                                    available_devices = []  # Mock empty - would populate with real devices

                                    if available_devices:
                                        rebalance_result = await self.sla_tier_manager.rebalance_service(
                                            service_id, available_devices
                                        )
                                        if rebalance_result["success"]:
                                            logger.info(f"Successfully rebalanced service {service_id}")

                await asyncio.sleep(600)  # Check every 10 minutes

            except Exception as e:
                logger.error(f"SLA monitoring error: {e}")
                await asyncio.sleep(300)

    async def stop(self):
        """Stop the fog computing coordinator"""
        logger.info("Stopping fog computing system...")

        self.is_running = False

        # Stop components gracefully
        if self.harvest_manager:
            # Stop all active harvesting sessions
            for device_id in list(self.harvest_manager.active_sessions.keys()):
                await self.harvest_manager.stop_harvesting(device_id, "system_shutdown")

        if self.onion_router:
            # Close all circuits
            for circuit_id in list(self.onion_router.circuits.keys()):
                await self.onion_router.close_circuit(circuit_id)

        if self.onion_coordinator:
            # Stop onion coordinator
            await self.onion_coordinator.stop()

        logger.info("Fog computing system stopped")

    # ============================================================================
    # PUBLIC API METHODS
    # ============================================================================

    async def register_mobile_device(
        self, device_id: str, capabilities: dict[str, Any], initial_state: dict[str, Any]
    ) -> bool:
        """Register a mobile device for fog computing"""

        success = True

        # Register with harvest manager
        if self.harvest_manager:
            from ..compute.harvest_manager import DeviceCapabilities

            device_caps = DeviceCapabilities(
                device_id=device_id,
                device_type=capabilities.get("device_type", "smartphone"),
                cpu_cores=capabilities.get("cpu_cores", 4),
                cpu_freq_mhz=capabilities.get("cpu_freq_mhz", 2000),
                cpu_architecture=capabilities.get("cpu_architecture", "arm64"),
                ram_total_mb=capabilities.get("ram_total_mb", 4096),
                ram_available_mb=capabilities.get("ram_available_mb", 2048),
                storage_total_gb=capabilities.get("storage_total_gb", 64),
                storage_available_gb=capabilities.get("storage_available_gb", 32),
                has_gpu=capabilities.get("has_gpu", False),
                network_speed_mbps=capabilities.get("network_speed_mbps", 50.0),
            )

            success &= await self.harvest_manager.register_device(device_id, device_caps, initial_state)

        # Register with marketplace as potential provider
        if self.marketplace and self.resource_manager:
            success &= await self.resource_manager.register_as_fog_provider(capabilities)

        # Create token account if needed
        if self.token_system and device_id not in self.token_system.accounts:
            device_key = f"device_{device_id}".encode()
            await self.token_system.create_account(device_id, device_key, 0)

        return success

    async def create_hidden_service(self, ports: dict[int, int], service_type: str = "web") -> str | None:
        """Create a .fog hidden service"""

        if not self.onion_router:
            logger.error("Onion routing not enabled")
            return None

        try:
            hidden_service = await self.onion_router.create_hidden_service(ports=ports, descriptor_cookie=None)

            # Register in marketplace if it's a commercial service
            if self.marketplace and service_type in ["web", "api", "database"]:
                service_tier = ServiceTier.BASIC
                if service_type == "web":
                    market_service_type = ServiceType.STATIC_WEBSITE
                elif service_type == "api":
                    market_service_type = ServiceType.SERVERLESS_FUNCTION
                else:
                    market_service_type = ServiceType.DATABASE

                from decimal import Decimal

                from ..marketplace.fog_marketplace import ServiceOffering

                offering = ServiceOffering(
                    offering_id=f"hidden_{hidden_service.service_id}",
                    provider_id=self.node_id,
                    service_type=market_service_type,
                    service_tier=service_tier,
                    pricing_model=ServiceOffering.__dataclass_fields__["pricing_model"].default,
                    base_price=Decimal("0.01"),  # 0.01 tokens per hour
                    regions=["fog_network"],
                    uptime_guarantee=99.0,
                )

                await self.marketplace.register_offering(self.node_id, offering)

            logger.info(f"Created hidden service: {hidden_service.onion_address}")
            return hidden_service.onion_address

        except Exception as e:
            logger.error(f"Failed to create hidden service: {e}")
            return None

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive fog system status"""

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

        # Add component-specific stats
        if self.harvest_manager:
            status["harvest"] = await self.harvest_manager.get_network_stats()

        if self.onion_router:
            status["onion"] = self.onion_router.get_stats()

        if self.marketplace:
            status["marketplace"] = self.marketplace.get_market_stats().__dict__

        if self.token_system:
            status["tokens"] = self.token_system.get_network_stats()

        if self.onion_coordinator:
            status["privacy"] = await self.onion_coordinator.get_coordinator_stats()

        return status

    async def process_fog_request(self, request_type: str, request_data: dict[str, Any]) -> dict[str, Any]:
        """Process various types of fog computing requests"""
        # Import privacy types here to avoid circular dependency
        from .fog_onion_coordinator import PrivacyAwareTask, PrivacyLevel

        try:
            if request_type == "compute_task":
                # Route compute task to harvest manager
                if self.harvest_manager:
                    assigned_device = await self.harvest_manager.assign_task(request_data)
                    return {"success": assigned_device is not None, "assigned_device": assigned_device}

            elif request_type == "service_request":
                # Handle marketplace service request
                if self.marketplace:
                    from ..marketplace.fog_marketplace import ServiceRequest

                    service_request = ServiceRequest(
                        request_id=request_data["request_id"],
                        customer_id=request_data["customer_id"],
                        service_type=ServiceType[request_data["service_type"]],
                        service_tier=ServiceTier[request_data.get("service_tier", "BASIC")],
                    )

                    contract_id = await self.marketplace.submit_request(request_data["customer_id"], service_request)

                    return {"success": contract_id is not None, "contract_id": contract_id}

            elif request_type == "token_transfer":
                # Handle token transfer
                if self.token_system:
                    success = await self.token_system.transfer(
                        request_data["from_account"],
                        request_data["to_account"],
                        request_data["amount"],
                        request_data.get("description", ""),
                    )
                    return {"success": success}

            elif request_type == "provision_sla_service":
                # Handle SLA tier service provisioning
                if self.sla_tier_manager:
                    tier = SLATier[request_data.get("tier", "BRONZE")]
                    available_devices = request_data.get("available_devices", [])
                    service_config = request_data.get("service_config", {})

                    result = await self.sla_tier_manager.provision_service(
                        service_id=request_data["service_id"],
                        tier=tier,
                        available_devices=available_devices,
                        service_config=service_config,
                    )

                    return result

            elif request_type == "validate_sla_compliance":
                # Handle SLA compliance validation
                if self.sla_tier_manager:
                    service_id = request_data["service_id"]
                    metrics = SLAMetrics(**request_data["metrics"])

                    result = await self.sla_tier_manager.validate_sla_compliance(service_id, metrics)
                    return result

            elif request_type == "get_quorum_status":
                # Handle quorum status requests
                if self.quorum_manager:
                    request_data.get("device_profiles", [])
                    # Convert dict profiles to InfrastructureProfile objects if needed
                    # This would need proper deserialization in production
                    status = self.quorum_manager.get_quorum_status_summary([])
                    return {"success": True, "status": status}

            elif request_type == "submit_privacy_task":
                # Handle privacy-aware task submission
                if self.onion_coordinator:
                    privacy_task = PrivacyAwareTask(
                        task_id=request_data["task_id"],
                        privacy_level=PrivacyLevel[request_data.get("privacy_level", "PRIVATE")],
                        task_data=(
                            request_data["task_data"].encode()
                            if isinstance(request_data["task_data"], str)
                            else request_data["task_data"]
                        ),
                        compute_requirements=request_data.get("compute_requirements", {}),
                        client_id=request_data["client_id"],
                        require_onion_circuit=request_data.get("require_onion_circuit", True),
                        require_mixnet=request_data.get("require_mixnet", False),
                        min_circuit_hops=request_data.get("min_circuit_hops", 3),
                    )

                    success = await self.onion_coordinator.submit_privacy_aware_task(privacy_task)
                    return {"success": success, "task_id": request_data["task_id"]}

            elif request_type == "create_privacy_service":
                # Handle privacy-aware service creation
                if self.onion_coordinator:
                    service = await self.onion_coordinator.create_privacy_aware_service(
                        service_id=request_data["service_id"],
                        service_type=request_data["service_type"],
                        privacy_level=PrivacyLevel[request_data.get("privacy_level", "PRIVATE")],
                        ports=request_data.get("ports", {}),
                        authentication_required=request_data.get("authentication_required", False),
                    )

                    if service:
                        return {
                            "success": True,
                            "service_id": service.service_id,
                            "onion_address": service.onion_address,
                            "privacy_level": service.privacy_level.value,
                        }
                    else:
                        return {"success": False, "error": "Failed to create privacy-aware service"}

            elif request_type == "send_private_gossip":
                # Handle private gossip communication
                if self.onion_coordinator:
                    success = await self.onion_coordinator.send_private_gossip(
                        recipient_id=request_data["recipient_id"],
                        message=(
                            request_data["message"].encode()
                            if isinstance(request_data["message"], str)
                            else request_data["message"]
                        ),
                        privacy_level=PrivacyLevel[request_data.get("privacy_level", "PRIVATE")],
                    )
                    return {"success": success}

            return {"success": False, "error": f"Unknown request type: {request_type}"}

        except Exception as e:
            logger.error(f"Failed to process fog request: {e}")
            return {"success": False, "error": str(e)}
