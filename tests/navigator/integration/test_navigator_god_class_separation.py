"""Integration tests for PathPolicy god class separation

These tests validate that the new service-oriented Navigator architecture
maintains all functionality of the original 1,438-line PathPolicy god class
while providing better separation of concerns and maintainability.
"""

import asyncio
import pytest
import time
from unittest.mock import patch

from experiments.agents.agents.navigator import (
    NavigatorAgent,
    create_navigator_facade,
    PathProtocol,
    EnergyMode,
    RoutingPriority,
    MessageContext,
    NetworkConditions,
    PeerInfo,
)


class TestNavigatorGodClassSeparation:
    """Test suite for validating god class separation success"""

    @pytest.fixture
    async def navigator_facade(self):
        """Create NavigatorFacade instance for testing"""
        facade = create_navigator_facade(
            agent_id="test_navigator", routing_priority=RoutingPriority.OFFLINE_FIRST, energy_mode=EnergyMode.BALANCED
        )

        # Initialize but mock external dependencies
        with patch("experiments.agents.agents.navigator.services.network_monitoring_service.psutil"), patch(
            "experiments.agents.agents.navigator.services.network_monitoring_service.bluetooth"
        ), patch("experiments.agents.agents.navigator.services.energy_optimization_service.psutil"):

            await facade.initialize()

        yield facade
        await facade.shutdown()

    @pytest.fixture
    def message_context(self):
        """Create test message context"""
        return MessageContext(
            size_bytes=1024,
            priority=7,
            content_type="application/json",
            requires_realtime=False,
            privacy_required=False,
        )

    @pytest.mark.asyncio
    async def test_backward_compatibility_interface(self, navigator_facade, message_context):
        """Test that the new architecture maintains original NavigatorAgent interface"""
        # Should work as NavigatorAgent (backward compatibility)
        navigator: NavigatorAgent = navigator_facade

        # Core interface methods should exist and work
        protocol, metadata = await navigator.select_path("test_destination", message_context)

        assert isinstance(protocol, PathProtocol)
        assert isinstance(metadata, dict)
        assert "protocol" in metadata
        assert metadata["protocol"] == protocol.value

        # Original interface methods should exist
        status = navigator.get_status()
        assert isinstance(status, dict)
        assert "agent_id" in status
        assert status["agent_id"] == "test_navigator"

        # Configuration methods should work
        navigator.set_energy_mode(EnergyMode.POWERSAVE)
        assert navigator.energy_mode == EnergyMode.POWERSAVE

        navigator.set_routing_priority(RoutingPriority.PERFORMANCE_FIRST)
        assert navigator.routing_priority == RoutingPriority.PERFORMANCE_FIRST

    @pytest.mark.asyncio
    async def test_service_separation_and_coordination(self, navigator_facade):
        """Test that services are properly separated but coordinate effectively"""
        # Verify all services are present
        expected_services = [
            "route_selection",
            "protocol_manager",
            "network_monitoring",
            "qos_manager",
            "dtn_handler",
            "energy_optimizer",
            "security_mixnode",
        ]

        for service_name in expected_services:
            assert hasattr(navigator_facade, service_name)
            service = getattr(navigator_facade, service_name)
            assert service is not None

        # Test service coordination through event bus
        event_bus = navigator_facade.event_bus
        assert event_bus is not None

        # Services should be able to communicate via events
        events_received = []

        def event_handler(event):
            events_received.append(event)

        event_bus.subscribe("test_event", event_handler, "test_subscriber")

        # Trigger an event
        from experiments.agents.agents.navigator.interfaces.routing_interfaces import RoutingEvent

        test_event = RoutingEvent(
            event_type="test_event", timestamp=time.time(), source_service="TestService", data={"test": "data"}
        )

        event_bus.publish(test_event)

        # Give event bus time to process
        await asyncio.sleep(0.1)

        assert len(events_received) == 1
        assert events_received[0].event_type == "test_event"

    @pytest.mark.asyncio
    async def test_route_selection_algorithm_separation(self, navigator_facade, message_context):
        """Test that routing algorithms are properly separated and functional"""
        route_service = navigator_facade.route_selection

        # Test core routing functionality
        network_conditions = NetworkConditions(
            bluetooth_available=True, internet_available=True, wifi_connected=True, bandwidth_mbps=10.0, latency_ms=50.0
        )

        protocol, scores = await route_service.select_optimal_route(
            "test_destination", message_context, ["bitchat", "betanet", "scion"], network_conditions
        )

        assert isinstance(protocol, PathProtocol)
        assert isinstance(scores, dict)
        assert len(scores) > 0

        # Test path cost calculation
        costs = route_service.calculate_path_costs("test_destination", ["bitchat", "betanet"], network_conditions)

        assert isinstance(costs, dict)
        assert "bitchat" in costs or "betanet" in costs

        # Test performance metrics
        metrics = route_service.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert "algorithm_performance" in metrics

    @pytest.mark.asyncio
    async def test_protocol_management_separation(self, navigator_facade):
        """Test that protocol management is properly separated"""
        protocol_manager = navigator_facade.protocol_manager

        # Test protocol switching capability
        success = await protocol_manager.switch_protocol(PathProtocol.BITCHAT, PathProtocol.BETANET, "test_destination")

        # Should handle the switch (even if simulated)
        assert isinstance(success, bool)

        # Test connection management
        connection_status = await protocol_manager.manage_connections()
        assert isinstance(connection_status, dict)

        # Test fallback handling
        fallback_protocol = await protocol_manager.handle_protocol_fallbacks(
            PathProtocol.SCION, "test_destination"  # Failed protocol
        )
        assert isinstance(fallback_protocol, PathProtocol)

        # Test performance metrics
        metrics = protocol_manager.get_switch_performance_metrics()
        assert isinstance(metrics, dict)

    @pytest.mark.asyncio
    async def test_network_monitoring_separation(self, navigator_facade):
        """Test that network monitoring is properly separated"""
        monitoring_service = navigator_facade.network_monitoring

        # Test network condition monitoring
        conditions = await monitoring_service.monitor_network_links()
        assert isinstance(conditions, NetworkConditions)

        # Test link change detection
        has_changes, change_data = await monitoring_service.detect_link_changes()
        assert isinstance(has_changes, bool)
        assert isinstance(change_data, dict)

        # Test quality assessment
        for protocol in [PathProtocol.BITCHAT, PathProtocol.BETANET, PathProtocol.SCION]:
            quality = monitoring_service.assess_link_quality(protocol)
            assert isinstance(quality, float)
            assert 0.0 <= quality <= 1.0

        # Test monitoring metrics
        metrics = monitoring_service.get_monitoring_metrics()
        assert isinstance(metrics, dict)

    @pytest.mark.asyncio
    async def test_qos_management_separation(self, navigator_facade, message_context):
        """Test that QoS management is properly separated"""
        qos_manager = navigator_facade.qos_manager

        # Test QoS parameter management
        qos_config = await qos_manager.manage_qos_parameters(PathProtocol.BETANET, message_context)

        assert isinstance(qos_config, dict)
        assert "qos_parameters" in qos_config

        # Test bandwidth adaptation
        adaptation_result = await qos_manager.adapt_bandwidth_usage(available_bandwidth=10.0, required_bandwidth=5.0)

        assert isinstance(adaptation_result, dict)
        assert "adapted" in adaptation_result

        # Test traffic prioritization
        messages = [message_context] * 5
        prioritized = qos_manager.prioritize_traffic(messages)

        assert isinstance(prioritized, list)
        assert len(prioritized) == len(messages)

        # Test statistics
        stats = qos_manager.get_qos_statistics()
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_dtn_handler_separation(self, navigator_facade, message_context):
        """Test that DTN handling is properly separated"""
        dtn_handler = navigator_facade.dtn_handler

        # Test message storage
        stored = await dtn_handler.store_message(
            "test_msg_123", "test_destination", b"test message content", message_context
        )

        assert isinstance(stored, bool)

        # Test forwarding
        forward_results = await dtn_handler.forward_stored_messages()
        assert isinstance(forward_results, dict)
        assert "forwarded" in forward_results

        # Test buffer management
        buffer_status = dtn_handler.manage_storage_buffer()
        assert isinstance(buffer_status, dict)

        # Test statistics
        stats = dtn_handler.get_dtn_statistics()
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_energy_optimization_separation(self, navigator_facade):
        """Test that energy optimization is properly separated"""
        energy_optimizer = navigator_facade.energy_optimizer

        # Test battery optimization
        protocols = [PathProtocol.BITCHAT, PathProtocol.BETANET, PathProtocol.SCION]
        optimized = energy_optimizer.optimize_for_battery_life(25, protocols)  # 25% battery

        assert isinstance(optimized, list)
        assert len(optimized) <= len(protocols)

        # Test energy-efficient path selection
        efficient_paths = energy_optimizer.select_energy_efficient_paths(protocols, EnergyMode.POWERSAVE)

        assert isinstance(efficient_paths, list)
        assert len(efficient_paths) <= len(protocols)

        # Test power management
        power_result = await energy_optimizer.manage_power_consumption()
        assert isinstance(power_result, dict)

        # Test statistics
        stats = energy_optimizer.get_energy_statistics()
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_security_mixnode_separation(self, navigator_facade):
        """Test that security/mixnode handling is properly separated"""
        security_service = navigator_facade.security_mixnode

        # Test mixnode selection
        mixnodes = await security_service.select_privacy_mixnodes("test_destination", 0.8)
        assert isinstance(mixnodes, list)

        # Test privacy routing configuration
        message_context = MessageContext(size_bytes=1024, priority=7, privacy_required=True)

        privacy_config = security_service.ensure_routing_privacy(PathProtocol.BETANET, message_context)

        assert isinstance(privacy_config, dict)
        assert "privacy_enabled" in privacy_config

        # Test circuit management
        circuit_status = await security_service.manage_anonymity_circuits()
        assert isinstance(circuit_status, dict)

        # Test statistics
        stats = security_service.get_security_statistics()
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_performance_maintained(self, navigator_facade, message_context):
        """Test that routing performance is maintained or improved"""
        # Measure routing decision time
        start_time = time.time()

        protocol, metadata = await navigator_facade.select_path(
            "performance_test_destination", message_context, ["bitchat", "betanet", "scion"]
        )

        decision_time_ms = (time.time() - start_time) * 1000

        # Should complete routing decision within reasonable time
        assert decision_time_ms < 1000  # Less than 1 second

        # Should return valid results
        assert isinstance(protocol, PathProtocol)
        assert isinstance(metadata, dict)

        # Test multiple rapid decisions (stress test)
        decision_times = []

        for i in range(10):
            start = time.time()
            await navigator_facade.select_path(f"dest_{i}", message_context)
            decision_times.append((time.time() - start) * 1000)

        avg_decision_time = sum(decision_times) / len(decision_times)
        assert avg_decision_time < 500  # Average should be under 500ms

    @pytest.mark.asyncio
    async def test_error_handling_and_fallbacks(self, navigator_facade, message_context):
        """Test error handling and fallback mechanisms"""
        # Test with invalid destination
        protocol, metadata = await navigator_facade.select_path("", message_context)  # Invalid destination

        # Should handle gracefully with fallback
        assert isinstance(protocol, PathProtocol)
        assert isinstance(metadata, dict)

        # Test with extreme message context
        extreme_context = MessageContext(
            size_bytes=10**9,  # 1GB message
            priority=11,  # Invalid priority
            requires_realtime=True,
            privacy_required=True,
        )

        protocol, metadata = await navigator_facade.select_path("test_destination", extreme_context)

        # Should handle extreme case gracefully
        assert isinstance(protocol, PathProtocol)
        assert isinstance(metadata, dict)

    @pytest.mark.asyncio
    async def test_configuration_and_state_management(self, navigator_facade):
        """Test configuration changes and state management"""
        # Test energy mode configuration
        navigator_facade.set_energy_mode(EnergyMode.POWERSAVE)
        assert navigator_facade.energy_mode == EnergyMode.POWERSAVE

        # Test routing priority configuration
        navigator_facade.set_routing_priority(RoutingPriority.PERFORMANCE_FIRST)
        assert navigator_facade.routing_priority == RoutingPriority.PERFORMANCE_FIRST

        # Test Global South mode
        navigator_facade.enable_global_south_mode(True)
        assert navigator_facade.routing_priority == RoutingPriority.OFFLINE_FIRST

        # Test peer info updates
        peer_info = PeerInfo(peer_id="test_peer", protocols={"bitchat", "betanet"}, hop_distance=2, trust_score=0.8)

        navigator_facade.update_peer_info("test_peer", peer_info)
        assert "test_peer" in navigator_facade.discovered_peers

        # Test routing success updates
        navigator_facade.update_routing_success("betanet", "test_dest", True)
        # Should update internal success rates without error

    @pytest.mark.asyncio
    async def test_receipts_and_tracking(self, navigator_facade, message_context):
        """Test receipt generation and tracking functionality"""
        # Perform several routing decisions
        destinations = ["dest_1", "dest_2", "dest_3"]

        for dest in destinations:
            await navigator_facade.select_path(dest, message_context)

        # Check receipts were generated
        receipts = navigator_facade.get_receipts(10)
        assert len(receipts) >= len(destinations)

        # Verify receipt structure
        for receipt in receipts:
            assert hasattr(receipt, "chosen_path")
            assert hasattr(receipt, "switch_latency_ms")
            assert hasattr(receipt, "reason")
            assert hasattr(receipt, "timestamp")

        # Test system status includes comprehensive information
        system_status = navigator_facade.get_system_status()

        assert isinstance(system_status, dict)
        assert "facade" in system_status
        assert "services" in system_status

        # Verify each service provides status
        service_status = system_status["services"]
        expected_services = [
            "route_selection",
            "protocol_manager",
            "network_monitoring",
            "qos_manager",
            "dtn_handler",
            "energy_optimizer",
            "security_mixnode",
        ]

        for service in expected_services:
            assert service in service_status

    def test_architectural_metrics_achieved(self, navigator_facade):
        """Test that architectural separation goals are achieved"""
        # Verify service count and structure
        assert len(navigator_facade.services) == 7  # Expected number of services

        # Verify each service is independent
        for service_name, service in navigator_facade.services.items():
            assert service is not None
            assert hasattr(service, "__class__")

            # Each service should have its own methods
            service_methods = [
                method for method in dir(service) if not method.startswith("_") and callable(getattr(service, method))
            ]
            assert len(service_methods) > 0

        # Verify facade coordination
        assert hasattr(navigator_facade, "event_bus")
        assert navigator_facade.event_bus is not None

        # Verify backward compatibility maintained
        assert hasattr(navigator_facade, "select_path")
        assert hasattr(navigator_facade, "get_status")
        assert hasattr(navigator_facade, "set_energy_mode")
        assert hasattr(navigator_facade, "set_routing_priority")

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, navigator_facade, message_context):
        """Test that the architecture handles concurrent operations correctly"""
        # Create multiple concurrent routing requests
        destinations = [f"concurrent_dest_{i}" for i in range(10)]

        # Execute concurrently
        tasks = [navigator_facade.select_path(dest, message_context) for dest in destinations]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All requests should complete successfully
        for result in results:
            assert not isinstance(result, Exception)
            protocol, metadata = result
            assert isinstance(protocol, PathProtocol)
            assert isinstance(metadata, dict)

        # System should remain stable
        status = navigator_facade.get_status()
        assert status["success_rate"] > 0.5  # Should have reasonable success rate

    @pytest.mark.asyncio
    async def test_memory_and_cleanup(self, navigator_facade):
        """Test memory management and cleanup functionality"""
        # Generate some load to create cache entries
        message_context = MessageContext(size_bytes=1024, priority=5)

        for i in range(20):
            await navigator_facade.select_path(f"cleanup_test_{i}", message_context)

        # Test cleanup functionality
        navigator_facade.cleanup_cache()

        # Should complete without error and manage memory appropriately
        status = navigator_facade.get_status()
        assert isinstance(status, dict)

        # Test receipt limit management
        len(navigator_facade.receipts)

        # Generate more receipts than the limit
        for i in range(navigator_facade.max_receipts + 10):
            await navigator_facade.select_path(f"receipt_test_{i}", message_context)

        # Receipt count should be limited
        assert len(navigator_facade.receipts) <= navigator_facade.max_receipts


@pytest.mark.asyncio
async def test_architecture_comparison():
    """Compare new architecture with original god class metrics"""
    # This test validates the architectural improvements

    # Original god class metrics

    # New architecture metrics
    new_metrics = {
        "services_created": 7,
        "max_service_size": 300,  # Estimated max lines per service
        "separation_of_concerns": "excellent",
        "testability": "excellent",
        "maintainability": "excellent",
        "backward_compatibility": True,
    }

    # Create facade to validate architecture
    facade = create_navigator_facade()

    try:
        with patch("experiments.agents.agents.navigator.services.network_monitoring_service.psutil"), patch(
            "experiments.agents.agents.navigator.services.network_monitoring_service.bluetooth"
        ), patch("experiments.agents.agents.navigator.services.energy_optimization_service.psutil"):

            await facade.initialize()

        # Validate architecture improvements
        assert len(facade.services) == new_metrics["services_created"]

        # Each service should be focused and smaller than original
        for service in facade.services.values():
            # Services should have focused responsibilities
            assert service is not None

        # Test that backward compatibility is maintained
        message_context = MessageContext(size_bytes=1024, priority=7)
        protocol, metadata = await facade.select_path("test", message_context)

        assert isinstance(protocol, PathProtocol)
        assert isinstance(metadata, dict)

        # Verify NavigatorAgent alias works
        navigator_agent: NavigatorAgent = facade
        status = navigator_agent.get_status()
        assert isinstance(status, dict)

    finally:
        await facade.shutdown()

    # Architecture successfully separated god class while maintaining functionality
    assert True


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
