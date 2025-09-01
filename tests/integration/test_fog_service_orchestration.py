"""
Integration Tests for Fog Service Orchestration

Comprehensive tests to validate the extracted service architecture including:
- Service initialization and dependency resolution
- Event-driven communication between services
- Backwards compatibility with original FogCoordinator
- Performance and coupling validation
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
import json

from infrastructure.fog.services import (
    FogCoordinatorFacade,
    create_fog_coordinator,
    ServiceRegistry,
    ServiceFactory,
    EventBus,
    FogTokenomicsService,
    FogMonitoringService,
    FogConfigurationService,
)


class TestFogServiceOrchestration:
    """Test suite for fog service orchestration"""

    @pytest.fixture
    async def event_bus(self):
        """Create event bus for testing"""
        return EventBus()

    @pytest.fixture
    async def service_registry(self, event_bus):
        """Create service registry for testing"""
        return ServiceRegistry(event_bus)

    @pytest.fixture
    async def test_config(self):
        """Create test configuration"""
        return {
            "node_id": "test_node",
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
            },
        }

    @pytest.fixture
    async def fog_coordinator(self, test_config):
        """Create fog coordinator facade for testing"""
        coordinator = FogCoordinatorFacade(
            node_id="test_node",
            config_path=None,
            enable_harvesting=True,
            enable_onion_routing=True,
            enable_marketplace=True,
            enable_tokens=True,
        )
        coordinator.config = test_config
        yield coordinator

        # Cleanup
        if coordinator.is_running:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_service_registry_initialization(self, service_registry):
        """Test service registry initialization"""
        assert service_registry is not None
        assert len(service_registry.services) == 0
        assert len(service_registry.dependencies) == 0
        assert service_registry.startup_order == []

    @pytest.mark.asyncio
    async def test_service_creation_and_registration(self, service_registry, test_config):
        """Test service creation and registration"""
        factory = ServiceFactory(service_registry, test_config)

        # Create configuration service
        config_service = factory.create_service(FogConfigurationService, "fog_configuration", test_config)

        # Verify service registration
        assert "fog_configuration" in service_registry.services
        assert service_registry.get_service("fog_configuration") == config_service
        assert service_registry.get_service_by_type(FogConfigurationService) == config_service

    @pytest.mark.asyncio
    async def test_dependency_resolution(self, service_registry, test_config):
        """Test service dependency resolution"""
        factory = ServiceFactory(service_registry, test_config)

        # Create services with dependencies
        factory.create_service(FogConfigurationService, "fog_configuration", test_config)

        factory.create_service(
            FogMonitoringService,
            "fog_monitoring",
            test_config,
            dependencies=[factory.registry.ServiceDependency(FogConfigurationService, required=True)],
        )

        # Resolve dependencies
        startup_order = service_registry.resolve_dependencies()

        # Configuration should start before monitoring
        config_index = startup_order.index("fog_configuration")
        monitoring_index = startup_order.index("fog_monitoring")
        assert config_index < monitoring_index

    @pytest.mark.asyncio
    async def test_event_driven_communication(self, event_bus):
        """Test event-driven communication between services"""
        events_received = []

        async def test_handler(event):
            events_received.append(event)

        # Subscribe to test events
        event_bus.subscribe("test_event", test_handler)

        # Publish event
        from infrastructure.fog.services.interfaces.base_service import ServiceEvent

        test_event = ServiceEvent("test_event", "test_service", {"data": "test"})
        await event_bus.publish(test_event)

        # Verify event was received
        assert len(events_received) == 1
        assert events_received[0].event_type == "test_event"
        assert events_received[0].source_service == "test_service"
        assert events_received[0].data["data"] == "test"

    @pytest.mark.asyncio
    async def test_fog_coordinator_facade_initialization(self, fog_coordinator):
        """Test fog coordinator facade initialization"""
        assert fog_coordinator.node_id == "test_node"
        assert fog_coordinator.enable_harvesting is True
        assert fog_coordinator.enable_onion_routing is True
        assert fog_coordinator.enable_marketplace is True
        assert fog_coordinator.enable_tokens is True
        assert fog_coordinator.is_running is False
        assert fog_coordinator.event_bus is not None
        assert fog_coordinator.service_registry is not None

    @pytest.mark.asyncio
    async def test_fog_coordinator_start_stop(self, fog_coordinator):
        """Test fog coordinator start and stop"""
        # Mock service initialization to avoid external dependencies
        with patch.multiple(
            "infrastructure.fog.services.harvesting.fog_harvesting_service.FogHarvestingService",
            initialize=AsyncMock(return_value=True),
            cleanup=AsyncMock(return_value=True),
            health_check=AsyncMock(),
        ), patch.multiple(
            "infrastructure.fog.services.routing.fog_routing_service.FogRoutingService",
            initialize=AsyncMock(return_value=True),
            cleanup=AsyncMock(return_value=True),
            health_check=AsyncMock(),
        ):
            # Start coordinator
            success = await fog_coordinator.start()
            assert success is True
            assert fog_coordinator.is_running is True
            assert fog_coordinator.stats["startup_time"] is not None

            # Verify services are created
            assert fog_coordinator.configuration_service is not None
            assert fog_coordinator.monitoring_service is not None

            # Stop coordinator
            await fog_coordinator.stop()
            assert fog_coordinator.is_running is False

    @pytest.mark.asyncio
    async def test_backwards_compatibility_api(self, fog_coordinator):
        """Test backwards compatibility with original FogCoordinator API"""
        # Mock service dependencies
        with patch.multiple(
            "infrastructure.fog.services.harvesting.fog_harvesting_service.FogHarvestingService",
            initialize=AsyncMock(return_value=True),
            register_mobile_device=AsyncMock(return_value=True),
            health_check=AsyncMock(),
        ):
            await fog_coordinator.start()

            # Test device registration (original API)
            device_id = "test_device"
            capabilities = {"device_type": "smartphone", "cpu_cores": 4, "ram_total_mb": 4096}
            initial_state = {"battery_percent": 80, "charging": True}

            success = await fog_coordinator.register_mobile_device(device_id, capabilities, initial_state)
            assert success is True

            # Test system status (original API)
            status = await fog_coordinator.get_system_status()
            assert "node_id" in status
            assert "is_running" in status
            assert "components" in status
            assert "statistics" in status

            await fog_coordinator.stop()

    @pytest.mark.asyncio
    async def test_service_health_monitoring(self, event_bus, test_config):
        """Test service health monitoring"""
        # Create monitoring service
        monitoring_service = FogMonitoringService("test_monitoring", test_config, event_bus)

        # Initialize service
        success = await monitoring_service.initialize()
        assert success is True

        # Register a service for monitoring
        await monitoring_service.register_service_for_monitoring("test_service", "test_type")

        # Verify service is tracked
        assert "test_service" in monitoring_service.tracked_services
        assert monitoring_service.metrics["services_monitored"] == 1

        # Test health check
        health = await monitoring_service.health_check()
        assert health.service_name == "test_monitoring"
        assert health.status is not None

        await monitoring_service.cleanup()

    @pytest.mark.asyncio
    async def test_configuration_management(self, event_bus, test_config):
        """Test configuration management service"""
        config_service = FogConfigurationService("test_config", test_config, event_bus)

        # Initialize service
        success = await config_service.initialize()
        assert success is True

        # Test configuration retrieval
        harvest_config = await config_service.get_configuration("harvest")
        assert harvest_config is not None
        assert harvest_config["min_battery_percent"] == 20

        # Test configuration update
        success = await config_service.update_configuration("harvest.min_battery_percent", 25, validate=True)
        assert success is True

        # Verify update
        updated_value = await config_service.get_configuration("harvest.min_battery_percent")
        assert updated_value == 25

        await config_service.cleanup()

    @pytest.mark.asyncio
    async def test_token_economics_service(self, event_bus, test_config):
        """Test tokenomics service functionality"""
        tokenomics_service = FogTokenomicsService("test_tokenomics", test_config, event_bus)

        # Initialize service
        success = await tokenomics_service.initialize()
        assert success is True

        # Test account creation
        success = await tokenomics_service.create_account("test_account", b"test_key", 100.0)
        assert success is True

        # Test balance retrieval
        balance = await tokenomics_service.get_account_balance("test_account")
        assert balance == 100.0

        # Test token transfer
        await tokenomics_service.create_account("recipient", b"recipient_key", 0.0)
        success = await tokenomics_service.transfer_tokens("test_account", "recipient", 50.0, "test transfer")
        assert success is True

        # Test reward distribution
        success = await tokenomics_service.distribute_reward("recipient", 25.0, "test reward")
        assert success is True

        await tokenomics_service.cleanup()

    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, fog_coordinator):
        """Test performance metrics collection"""
        with patch("psutil.cpu_percent", return_value=50.0), patch("psutil.virtual_memory") as mock_memory, patch(
            "psutil.disk_usage"
        ) as mock_disk:

            # Mock system metrics
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.percent = 70.0

            # Start coordinator
            with patch.multiple(
                "infrastructure.fog.services.monitoring.fog_monitoring_service.FogMonitoringService",
                initialize=AsyncMock(return_value=True),
                health_check=AsyncMock(),
            ):
                await fog_coordinator.start()

                # Get system status with metrics
                status = await fog_coordinator.get_system_status()

                # Verify metrics structure
                assert "service_orchestration" in status
                assert "monitoring" in status

                await fog_coordinator.stop()

    @pytest.mark.asyncio
    async def test_coupling_reduction_validation(self):
        """Test that coupling has been reduced as expected"""
        from infrastructure.fog.services import SERVICE_COUPLING_METRICS

        # Verify coupling metrics
        assert SERVICE_COUPLING_METRICS["average_coupling"] < 15.0
        assert SERVICE_COUPLING_METRICS["FogHarvestingService"] < 15.0
        assert SERVICE_COUPLING_METRICS["FogRoutingService"] < 15.0
        assert SERVICE_COUPLING_METRICS["FogMarketplaceService"] < 15.0
        assert SERVICE_COUPLING_METRICS["FogTokenomicsService"] < 15.0
        assert SERVICE_COUPLING_METRICS["FogNetworkingService"] < 15.0
        assert SERVICE_COUPLING_METRICS["FogMonitoringService"] < 15.0
        assert SERVICE_COUPLING_METRICS["FogConfigurationService"] < 15.0

        # Verify significant coupling reduction (target: >70% reduction)
        original_coupling = 39.8
        current_coupling = SERVICE_COUPLING_METRICS["average_coupling"]
        reduction_percentage = ((original_coupling - current_coupling) / original_coupling) * 100

        assert reduction_percentage > 70.0
        print(f"Coupling reduction achieved: {reduction_percentage:.1f}%")

    @pytest.mark.asyncio
    async def test_service_isolation_and_testability(self, event_bus, test_config):
        """Test that services are properly isolated and testable"""
        # Create individual service without dependencies
        config_service = FogConfigurationService("isolated_config", test_config, event_bus)

        # Service should initialize independently
        success = await config_service.initialize()
        assert success is True

        # Service should function independently
        value = await config_service.get_configuration("harvest.min_battery_percent")
        assert value == 20

        # Service should clean up independently
        success = await config_service.cleanup()
        assert success is True

    @pytest.mark.asyncio
    async def test_factory_function_compatibility(self):
        """Test factory function for backwards compatibility"""
        coordinator = create_fog_coordinator(
            node_id="factory_test",
            enable_harvesting=True,
            enable_onion_routing=False,
            enable_marketplace=True,
            enable_tokens=True,
        )

        assert isinstance(coordinator, FogCoordinatorFacade)
        assert coordinator.node_id == "factory_test"
        assert coordinator.enable_harvesting is True
        assert coordinator.enable_onion_routing is False
        assert coordinator.enable_marketplace is True
        assert coordinator.enable_tokens is True

    @pytest.mark.asyncio
    async def test_concurrent_service_operations(self, fog_coordinator):
        """Test concurrent operations across services"""
        # Mock services for concurrent testing
        with patch.multiple(
            "infrastructure.fog.services.harvesting.fog_harvesting_service.FogHarvestingService",
            initialize=AsyncMock(return_value=True),
            register_mobile_device=AsyncMock(return_value=True),
        ), patch.multiple(
            "infrastructure.fog.services.tokenomics.fog_tokenomics_service.FogTokenomicsService",
            initialize=AsyncMock(return_value=True),
            create_account=AsyncMock(return_value=True),
        ):
            await fog_coordinator.start()

            # Perform concurrent operations
            tasks = []

            # Register multiple devices concurrently
            for i in range(5):
                device_id = f"device_{i}"
                capabilities = {"device_type": "smartphone", "cpu_cores": 4}
                initial_state = {"battery_percent": 80}

                task = fog_coordinator.register_mobile_device(device_id, capabilities, initial_state)
                tasks.append(task)

            # Wait for all operations to complete
            results = await asyncio.gather(*tasks)

            # Verify all operations succeeded
            assert all(results)

            await fog_coordinator.stop()

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, fog_coordinator):
        """Test error handling and recovery mechanisms"""
        # Mock service that fails initialization
        with patch.multiple(
            "infrastructure.fog.services.harvesting.fog_harvesting_service.FogHarvestingService",
            initialize=AsyncMock(side_effect=Exception("Mock initialization failure")),
        ):
            # Start should handle initialization failure gracefully
            success = await fog_coordinator.start()
            assert success is False  # Should fail due to service initialization error
            assert fog_coordinator.is_running is False

    @pytest.mark.asyncio
    async def test_configuration_file_integration(self, tmp_path):
        """Test configuration file integration"""
        # Create temporary config file
        config_data = {
            "node_id": "file_test",
            "harvest": {"min_battery_percent": 30},
            "tokens": {"initial_supply": 2000000000},
        }

        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Create coordinator with config file
        coordinator = FogCoordinatorFacade(node_id="file_test", config_path=config_file)

        # Verify config was loaded
        assert coordinator.config["harvest"]["min_battery_percent"] == 30
        assert coordinator.config["tokens"]["initial_supply"] == 2000000000


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
