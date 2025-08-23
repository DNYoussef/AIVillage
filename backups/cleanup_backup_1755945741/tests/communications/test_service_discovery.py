"""Tests for service discovery registry and cleanup behavior."""

import time

from src.communications.service_discovery import ServiceInfo, ServiceRegistry


def test_service_cleanup_and_discovery():
    registry = ServiceRegistry()

    # Register two services of the same type
    registry.register_service(
        ServiceInfo(
            agent_id="agent1",
            service_type="test",
            host="localhost",
            port=8000,
            capabilities=[],
            metadata={},
            last_heartbeat=time.time(),
        )
    )
    registry.register_service(
        ServiceInfo(
            agent_id="agent2",
            service_type="test",
            host="localhost",
            port=8001,
            capabilities=[],
            metadata={},
            last_heartbeat=time.time(),
        )
    )

    # Ensure both services are initially discoverable
    assert len(registry.discover_services("test")) == 2

    # Mark second service as stale
    registry.services["agent2:test"].last_heartbeat = time.time() - registry.heartbeat_timeout - 1

    # Cleanup stale services
    cleaned = registry.cleanup_stale_services()
    assert cleaned == 1

    # Only the active service should remain discoverable
    active_services = registry.discover_services("test")
    assert len(active_services) == 1
    assert active_services[0].agent_id == "agent1"

    # Discovering all services should yield the same result
    all_services = registry.discover_services()
    assert len(all_services) == 1
    assert all_services[0].agent_id == "agent1"
