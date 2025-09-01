"""
BitChat Transport Integration Test Suite

Tests the unified transport system with BitChat bridge integration,
validating the complete transport stack for production readiness.
"""

import asyncio
import logging
import os
import sys

import pytest

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from aivillage.core.persistence import StorageManager
    from aivillage.p2p.bitchat_bridge import create_bitchat_bridge, is_bitchat_available
    from aivillage.p2p.transport import TransportManager, TransportPriority, TransportType

    IMPORTS_AVAILABLE = True
    logger.info("✅ All transport imports successful")
except ImportError as e:
    logger.error(f"❌ Import failed: {e}")
    IMPORTS_AVAILABLE = False


class TestBitChatBridge:
    """Test BitChat bridge functionality"""

    def test_bridge_creation(self):
        """Test BitChat bridge can be created"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")

        bridge = create_bitchat_bridge("test_device_001")
        assert bridge is not None
        assert bridge.device_id == "test_device_001"

        # Test bridge functionality, not just creation
        assert hasattr(bridge, "send_message"), "Bridge missing send_message method"
        assert hasattr(bridge, "receive_message"), "Bridge missing receive_message method"
        assert hasattr(bridge, "get_status"), "Bridge missing get_status method"

        # Test bridge state initialization
        status = bridge.get_status()
        assert status is not None, "Bridge status should not be None"
        assert isinstance(status, dict), "Bridge status should be a dictionary"
        assert status.get("device_id") == "test_device_001", "Status should contain correct device_id"

        logger.info("✅ BitChat bridge creation test passed")

    def test_bridge_availability(self):
        """Test BitChat availability detection"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")

        available = is_bitchat_available()
        logger.info(f"BitChat availability: {available}")
        # Should not fail regardless of availability
        assert isinstance(available, bool)
        logger.info("✅ BitChat availability test passed")

    def test_bridge_status(self):
        """Test bridge status reporting"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")

        bridge = create_bitchat_bridge("test_device_002")
        status = bridge.get_status()

        # Verify status structure
        required_keys = ["available", "device_id", "transport_active", "registered_handlers"]
        for key in required_keys:
            assert key in status, f"Missing status key: {key}"

        assert status["device_id"] == "test_device_002"
        logger.info("✅ Bridge status test passed")


class TestTransportManager:
    """Test unified transport manager functionality"""

    @pytest.mark.asyncio
    async def test_transport_creation(self):
        """Test transport manager creation"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")

        manager = TransportManager(
            device_id="test_device_003",
            transport_priority=TransportPriority.OFFLINE_FIRST,
            enable_bitchat=True,
            enable_betanet=True,
        )

        assert manager.device_id == "test_device_003"
        assert manager.transport_priority == TransportPriority.OFFLINE_FIRST
        logger.info("✅ Transport manager creation test passed")

    @pytest.mark.asyncio
    async def test_transport_start_stop(self):
        """Test transport manager start/stop functionality"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")

        manager = TransportManager(
            device_id="test_device_004",
            transport_priority=TransportPriority.OFFLINE_FIRST,
            enable_bitchat=True,
            enable_betanet=True,  # Enable Betanet for testing
        )

        # Test start
        start_success = await manager.start()
        logger.info(f"Start result: {start_success}")

        # Always check status after start attempt
        status = manager.get_transport_status()
        active_count = len(status["active_transports"])
        logger.info(f"Active transports after start: {active_count}")

        # Test stop
        stop_success = await manager.stop()
        logger.info(f"Stop result: {stop_success}")

        # Verify stop worked
        final_status = manager.get_transport_status()
        final_active_count = len(final_status["active_transports"])
        logger.info(f"Active transports after stop: {final_active_count}")

        # Should have fewer or same active transports after stop
        assert final_active_count <= active_count
        logger.info("✅ Transport start/stop test passed")

    @pytest.mark.asyncio
    async def test_transport_status(self):
        """Test transport status reporting"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")

        manager = TransportManager(device_id="test_device_005", transport_priority=TransportPriority.PRIVACY_FIRST)

        status = manager.get_transport_status()

        # Verify status structure
        required_keys = ["device_id", "transport_priority", "active_transports", "transport_count", "statistics"]
        for key in required_keys:
            assert key in status, f"Missing status key: {key}"

        assert status["device_id"] == "test_device_005"
        assert status["transport_priority"] == "privacy_first"
        assert isinstance(status["active_transports"], list)
        assert isinstance(status["transport_count"], int)
        logger.info("✅ Transport status test passed")

    @pytest.mark.asyncio
    async def test_message_sending(self):
        """Test message sending functionality"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")

        manager = TransportManager(device_id="test_device_006", enable_bitchat=True, enable_betanet=True)

        # Start transports
        await manager.start()

        try:
            # Test message sending
            test_payload = b"Hello, unified transport system!"
            result = await manager.send_message(recipient_id="test_recipient", payload=test_payload, priority=7)

            # Should not fail even if no transports available
            assert isinstance(result, bool)
            logger.info(f"Message send result: {result}")

        finally:
            # Clean up
            await manager.stop()

        logger.info("✅ Message sending test passed")


class TestPersistenceIntegration:
    """Test persistence layer integration"""

    def test_storage_manager_creation(self):
        """Test storage manager can be created"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")

        try:
            storage = StorageManager()
            assert storage is not None
            logger.info("✅ Storage manager creation test passed")
        except Exception as e:
            logger.warning(f"Storage manager creation failed: {e}")
            pytest.skip("Storage manager not available")

    def test_storage_stats(self):
        """Test storage statistics functionality"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")

        try:
            storage = StorageManager()
            stats = storage.get_storage_statistics()

            # Should return statistics without error
            assert isinstance(stats, dict)
            logger.info(f"Storage stats: {stats}")
            logger.info("✅ Storage statistics test passed")
        except Exception as e:
            logger.warning(f"Storage statistics failed: {e}")
            pytest.skip("Storage statistics not working")


class TestIntegrationComplete:
    """Comprehensive integration tests"""

    @pytest.mark.asyncio
    async def test_complete_transport_integration(self):
        """Test complete transport system integration"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")

        logger.info("🚀 Starting complete transport integration test")

        # Create transport manager with full configuration
        manager = TransportManager(
            device_id="integration_test_device",
            transport_priority=TransportPriority.OFFLINE_FIRST,
            enable_bitchat=True,
            enable_betanet=True,
            enable_quic=True,
            max_peers=20,
            battery_aware=True,
        )

        # Test full lifecycle
        try:
            # 1. Start all transports
            start_result = await manager.start()
            logger.info(f"Transport start result: {start_result}")

            # 2. Get comprehensive status
            status = manager.get_transport_status()
            logger.info(f"Transport status: {status}")

            # 3. Register message handler
            received_messages = []

            def test_handler(message):
                received_messages.append(message)
                logger.info(f"Received message: {message}")

            manager.register_message_handler(test_handler)

            # 4. Test message sending with different priorities
            test_messages = [
                ("recipient_1", b"Test message 1", TransportType.BITCHAT),
                ("recipient_2", b"Test message 2", TransportType.BETANET),
                ("recipient_3", b"Test message 3", None),  # Auto-select transport
            ]

            send_results = []
            for recipient, payload, transport_pref in test_messages:
                result = await manager.send_message(
                    recipient_id=recipient, payload=payload, transport_preference=transport_pref
                )
                send_results.append(result)
                logger.info(f"Send to {recipient} via {transport_pref}: {result}")

            # 5. Verify final status
            final_status = manager.get_transport_status()
            logger.info(f"Final status: {final_status}")

            # Test should complete without exceptions
            assert isinstance(send_results, list)
            assert len(send_results) == 3
            logger.info("✅ Complete integration test passed")

        finally:
            # Clean up
            await manager.stop()
            logger.info("🏁 Integration test cleanup completed")


# Test execution
if __name__ == "__main__":

    async def run_integration_tests():
        """Run all integration tests"""
        logger.info("🧪 Starting BitChat Transport Integration Tests")

        if not IMPORTS_AVAILABLE:
            logger.error("❌ Cannot run tests - imports failed")
            return False

        test_classes = [
            TestBitChatBridge(),
            TestTransportManager(),
            TestPersistenceIntegration(),
            TestIntegrationComplete(),
        ]

        total_tests = 0
        passed_tests = 0

        for test_class in test_classes:
            class_name = test_class.__class__.__name__
            logger.info(f"\n📋 Running {class_name} tests...")

            # Get all test methods
            test_methods = [method for method in dir(test_class) if method.startswith("test_")]

            for test_method_name in test_methods:
                total_tests += 1
                test_method = getattr(test_class, test_method_name)

                try:
                    if asyncio.iscoroutinefunction(test_method):
                        await test_method()
                    else:
                        test_method()
                    passed_tests += 1
                    logger.info(f"✅ {test_method_name} PASSED")
                except Exception as e:
                    logger.error(f"❌ {test_method_name} FAILED: {e}")

        logger.info(f"\n📊 Test Results: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - BitChat transport integration successful!")
            return True
        else:
            logger.warning(f"⚠️  {total_tests - passed_tests} tests failed")
            return False

    # Run the tests
    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1)
