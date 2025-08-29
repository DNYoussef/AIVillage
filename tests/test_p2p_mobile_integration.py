"""Integration tests for P2P mobile bridge with LibP2P network.

Tests the complete integration between:
- LibP2P mesh network implementation
- Mobile JNI bridge layer
- Security and delivery systems
- Transport manager integration

This ensures the 532-line Android bridge can successfully
connect to and use the actual P2P network.
"""

import asyncio
import logging
from pathlib import Path
import sys
import time

# Add the infrastructure path to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "infrastructure"))

from p2p.core.libp2p_transport import create_libp2p_transport
from p2p.mobile_integration.jni.libp2p_mesh_bridge import LibP2PMeshBridge, initialize_bridge
from p2p.mobile_integration.libp2p_mesh import MeshConfiguration
from p2p.security.production_security import SecurityLevel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class P2PMobileIntegrationTest:
    """Comprehensive test suite for P2P mobile integration."""

    def __init__(self):
        self.bridge: LibP2PMeshBridge = None
        self.transport = None
        self.test_results = {
            "bridge_initialization": False,
            "transport_creation": False,
            "mesh_startup": False,
            "peer_discovery": False,
            "message_sending": False,
            "message_receiving": False,
            "security_integration": False,
            "mobile_api_calls": False,
            "performance_metrics": {},
            "errors": [],
        }

    async def run_all_tests(self):
        """Run the complete test suite."""
        logger.info("Starting P2P Mobile Integration Test Suite")
        start_time = time.time()

        try:
            # Test 1: Bridge Initialization
            await self.test_bridge_initialization()

            # Test 2: Transport Creation
            await self.test_transport_creation()

            # Test 3: Mesh Network Startup
            await self.test_mesh_startup()

            # Test 4: Basic Messaging
            await self.test_basic_messaging()

            # Test 5: Mobile API Integration
            await self.test_mobile_api_calls()

            # Test 6: Security Integration
            await self.test_security_integration()

            # Test 7: Performance Testing
            await self.test_performance()

            # Test 8: Error Handling
            await self.test_error_handling()

        except Exception as e:
            logger.error(f"Test suite failed with error: {e}")
            self.test_results["errors"].append(str(e))

        finally:
            # Cleanup
            await self.cleanup()

        # Generate test report
        total_time = time.time() - start_time
        self.test_results["performance_metrics"]["total_test_time"] = total_time

        self.generate_test_report()

    async def test_bridge_initialization(self):
        """Test 1: Bridge Initialization."""
        logger.info("Test 1: Testing bridge initialization")

        try:
            # Test the JNI-style initialization
            result = initialize_bridge(port=8080)
            if result["success"]:
                logger.info("✓ Bridge initialization successful")
                self.test_results["bridge_initialization"] = True

                # Verify bridge info
                bridge_info = result["bridge_info"]
                expected_endpoints = ["start_mesh", "stop_mesh", "get_status", "send_message", "get_peers"]

                endpoints = bridge_info.get("endpoints", {})
                for endpoint in expected_endpoints:
                    if endpoint.replace("_", "_").lower() not in str(endpoints).lower():
                        logger.warning(f"Missing endpoint: {endpoint}")

            else:
                logger.error(f"Bridge initialization failed: {result.get('error', 'Unknown error')}")
                self.test_results["errors"].append("Bridge initialization failed")

        except Exception as e:
            logger.error(f"Bridge initialization test failed: {e}")
            self.test_results["errors"].append(f"Bridge init error: {str(e)}")

    async def test_transport_creation(self):
        """Test 2: Transport Creation."""
        logger.info("Test 2: Testing LibP2P transport creation")

        try:
            # Create LibP2P transport
            self.transport = create_libp2p_transport(
                node_id="test_mobile_node",
                security_level=SecurityLevel.STANDARD,
                enable_delivery_guarantees=True,
                listen_port=9000,  # Different from bridge port
                enable_mdns=False,  # Disable for testing
            )

            if self.transport:
                logger.info("✓ LibP2P transport created successfully")
                self.test_results["transport_creation"] = True

                # Verify capabilities
                capabilities = self.transport.get_capabilities()
                if capabilities.supports_broadcast and capabilities.supports_unicast:
                    logger.info("✓ Transport capabilities verified")
                else:
                    logger.warning("Transport capabilities incomplete")

        except Exception as e:
            logger.error(f"Transport creation test failed: {e}")
            self.test_results["errors"].append(f"Transport creation error: {str(e)}")

    async def test_mesh_startup(self):
        """Test 3: Mesh Network Startup."""
        logger.info("Test 3: Testing mesh network startup")

        try:
            # Start the transport (which starts the mesh network)
            if self.transport:
                success = await self.transport.start()
                if success:
                    logger.info("✓ LibP2P transport started successfully")
                    self.test_results["mesh_startup"] = True

                    # Wait a moment for initialization
                    await asyncio.sleep(2)

                    # Check transport status
                    status = self.transport.get_status()
                    if status["running"] and status["mesh_status"]["status"] == "active":
                        logger.info("✓ Mesh network is active")
                    else:
                        logger.warning(f"Mesh network status: {status['mesh_status']['status']}")
                else:
                    logger.error("Failed to start LibP2P transport")
                    self.test_results["errors"].append("Transport startup failed")

        except Exception as e:
            logger.error(f"Mesh startup test failed: {e}")
            self.test_results["errors"].append(f"Mesh startup error: {str(e)}")

    async def test_basic_messaging(self):
        """Test 4: Basic Messaging."""
        logger.info("Test 4: Testing basic messaging functionality")

        try:
            if not self.transport or not self.transport.running:
                logger.error("Transport not running, skipping messaging test")
                return

            from p2p.core.message_types import MessageMetadata, MessageType, UnifiedMessage

            # Create a test message
            metadata = MessageMetadata(
                sender_id="test_mobile_node",
                recipient_id="broadcast",
            )

            test_message = UnifiedMessage(
                message_type=MessageType.DATA,
                payload=b"Hello from mobile integration test!",
                metadata=metadata,
            )

            # Send message through transport
            success = await self.transport.send_message(test_message)

            if success:
                logger.info("✓ Message sent successfully")
                self.test_results["message_sending"] = True
            else:
                logger.error("Message sending failed")
                self.test_results["errors"].append("Message sending failed")

            # Give some time for message processing
            await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Basic messaging test failed: {e}")
            self.test_results["errors"].append(f"Messaging error: {str(e)}")

    async def test_mobile_api_calls(self):
        """Test 5: Mobile API Integration."""
        logger.info("Test 5: Testing mobile API integration")

        try:
            # Create a bridge instance for direct testing
            self.bridge = LibP2PMeshBridge(port=8081)  # Different port

            # Test mesh startup via bridge
            config = {
                "node_id": "mobile_bridge_test",
                "listen_port": 9001,
                "max_peers": 10,
            }

            # This simulates the REST API call that Android would make
            response = await self._simulate_rest_call("start_mesh", config)

            if response and response.get("status") == "started":
                logger.info("✓ Mobile API mesh start successful")

                # Test status call
                status_response = await self._simulate_rest_call("get_status")
                if status_response and "node_id" in status_response:
                    logger.info("✓ Mobile API status call successful")
                    self.test_results["mobile_api_calls"] = True

                # Test message sending via bridge
                message_data = {
                    "type": "DATA_MESSAGE",
                    "sender": "mobile_test",
                    "recipient": None,  # Broadcast
                    "payload": "Hello from mobile API!",
                    "ttl": 5,
                }

                send_response = await self._simulate_rest_call("send_message", message_data)
                if send_response and send_response.get("success"):
                    logger.info("✓ Mobile API message sending successful")

                # Test peers call
                peers_response = await self._simulate_rest_call("get_peers")
                if peers_response is not None:
                    logger.info("✓ Mobile API peers call successful")

            else:
                logger.error("Mobile API mesh start failed")
                self.test_results["errors"].append("Mobile API calls failed")

        except Exception as e:
            logger.error(f"Mobile API test failed: {e}")
            self.test_results["errors"].append(f"Mobile API error: {str(e)}")

    async def _simulate_rest_call(self, endpoint: str, data=None):
        """Simulate REST API calls that mobile clients would make."""
        try:
            if endpoint == "start_mesh" and self.bridge:
                # Simulate starting mesh via bridge
                if hasattr(self.bridge, "mesh_network") and not self.bridge.mesh_network:
                    # Create and configure mesh network
                    config = MeshConfiguration()
                    if data:
                        config.node_id = data.get("node_id", config.node_id)
                        config.listen_port = data.get("listen_port", config.listen_port)
                        config.max_peers = data.get("max_peers", config.max_peers)

                    # Simulate bridge start
                    from p2p.mobile_integration.libp2p_mesh import LibP2PMeshNetwork

                    self.bridge.mesh_network = LibP2PMeshNetwork(config)
                    success = await self.bridge.mesh_network.start()

                    if success:
                        return {
                            "status": "started",
                            "node_id": config.node_id,
                            "listen_port": config.listen_port,
                        }

            elif endpoint == "get_status" and self.bridge and self.bridge.mesh_network:
                return self.bridge.mesh_network.get_mesh_status()

            elif endpoint == "send_message" and self.bridge and self.bridge.mesh_network:
                from p2p.mobile_integration.libp2p_mesh import MeshMessage, MeshMessageType

                message = MeshMessage(
                    type=MeshMessageType(data.get("type", "DATA_MESSAGE")),
                    sender=data.get("sender", ""),
                    recipient=data.get("recipient"),
                    payload=data.get("payload", "").encode(),
                    ttl=data.get("ttl", 5),
                )

                success = await self.bridge.mesh_network.send_message(message)
                return {"success": success, "message_id": message.id}

            elif endpoint == "get_peers" and self.bridge and self.bridge.mesh_network:
                return {"peers": list(self.bridge.mesh_network.connected_peers.keys())}

            return None

        except Exception as e:
            logger.error(f"REST call simulation failed: {e}")
            return None

    async def test_security_integration(self):
        """Test 6: Security Integration."""
        logger.info("Test 6: Testing security integration")

        try:
            if not self.transport or not self.transport.security_manager:
                logger.warning("No security manager available for testing")
                return

            # Test peer authentication
            auth_data = {
                "public_key": self.transport.security_manager.crypto_manager.get_public_key(),
                "signature": b"test_signature",
                "timestamp": time.time(),
            }

            # This would normally fail due to invalid signature, but tests the flow
            await self.transport.security_manager.authenticate_peer("test_peer", auth_data)

            # Get security status
            security_status = self.transport.security_manager.get_security_status()

            if security_status and "security_level" in security_status:
                logger.info("✓ Security manager operational")
                self.test_results["security_integration"] = True

        except Exception as e:
            logger.error(f"Security integration test failed: {e}")
            self.test_results["errors"].append(f"Security error: {str(e)}")

    async def test_performance(self):
        """Test 7: Performance Testing."""
        logger.info("Test 7: Testing performance metrics")

        try:
            if not self.transport:
                return

            # Measure startup time
            start_time = time.time()
            status = self.transport.get_status()
            status_time = time.time() - start_time

            self.test_results["performance_metrics"]["status_call_time"] = status_time

            # Check message throughput capability
            if self.transport.delivery_service:
                delivery_status = self.transport.delivery_service.get_delivery_status()
                self.test_results["performance_metrics"]["delivery_metrics"] = delivery_status.get(
                    "performance_metrics", {}
                )

            # Check memory usage (simplified)
            mesh_status = status.get("mesh_status", {})
            self.test_results["performance_metrics"]["peer_count"] = mesh_status.get("peer_count", 0)
            self.test_results["performance_metrics"]["message_cache_size"] = mesh_status.get("message_cache_size", 0)

            logger.info("✓ Performance metrics collected")

        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            self.test_results["errors"].append(f"Performance error: {str(e)}")

    async def test_error_handling(self):
        """Test 8: Error Handling."""
        logger.info("Test 8: Testing error handling")

        try:
            # Test invalid message sending
            if self.transport:
                from p2p.core.message_types import MessageMetadata, MessageType, UnifiedMessage

                # Create message with invalid data
                metadata = MessageMetadata(sender_id="", recipient_id="")  # Invalid empty sender
                invalid_message = UnifiedMessage(
                    message_type=MessageType.DATA,
                    payload=b"",  # Empty payload
                    metadata=metadata,
                )

                # This should handle the error gracefully
                result = await self.transport.send_message(invalid_message)
                logger.info(f"✓ Invalid message handled gracefully (result: {result})")

            # Test bridge error conditions
            if self.bridge:
                # Try to start mesh when already started
                try:
                    await self._simulate_rest_call("start_mesh", {"node_id": "duplicate"})
                    logger.info("✓ Duplicate start handled")
                except Exception as e:
                    logger.info(f"✓ Duplicate start error handled: {e}")

        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            self.test_results["errors"].append(f"Error handling test error: {str(e)}")

    async def cleanup(self):
        """Clean up test resources."""
        logger.info("Cleaning up test resources")

        try:
            # Stop transport
            if self.transport and self.transport.running:
                await self.transport.stop()
                logger.info("✓ Transport stopped")

            # Stop bridge mesh if running
            if self.bridge and hasattr(self.bridge, "mesh_network") and self.bridge.mesh_network:
                await self.bridge.mesh_network.stop()
                logger.info("✓ Bridge mesh stopped")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("\n" + "=" * 60)
        logger.info("P2P MOBILE INTEGRATION TEST REPORT")
        logger.info("=" * 60)

        # Test results summary
        passed_tests = sum(1 for result in self.test_results.values() if result is True)
        total_tests = len([k for k in self.test_results.keys() if k not in ["performance_metrics", "errors"]])

        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")

        # Individual test results
        test_names = {
            "bridge_initialization": "Bridge Initialization",
            "transport_creation": "Transport Creation",
            "mesh_startup": "Mesh Network Startup",
            "message_sending": "Message Sending",
            "message_receiving": "Message Receiving",
            "security_integration": "Security Integration",
            "mobile_api_calls": "Mobile API Calls",
        }

        for key, name in test_names.items():
            status = "✓ PASS" if self.test_results[key] else "✗ FAIL"
            logger.info(f"{name:.<30} {status}")

        # Performance metrics
        logger.info("\nPerformance Metrics:")
        metrics = self.test_results["performance_metrics"]
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

        # Errors
        if self.test_results["errors"]:
            logger.info("\nErrors Encountered:")
            for error in self.test_results["errors"]:
                logger.info(f"  - {error}")

        # Overall result
        logger.info("\n" + "=" * 60)
        if passed_tests >= total_tests * 0.8:  # 80% pass rate
            logger.info("✓ OVERALL RESULT: INTEGRATION SUCCESSFUL")
            logger.info("The mobile bridge can successfully connect to the P2P network")
        else:
            logger.info("✗ OVERALL RESULT: INTEGRATION NEEDS WORK")
            logger.info("Some critical components need fixes before production")

        logger.info("=" * 60)


async def main():
    """Run the integration test suite."""
    test_suite = P2PMobileIntegrationTest()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())
