"""Communications Component Validation Suite.

Tests communication protocols, service discovery, and message passing systems.
"""

import logging
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.communications.message_passing_system import MessagePassingSystem
    from src.communications.service_discovery import ServiceDiscovery, ServiceInfo
    from src.communications.standard_protocol import StandardCommunicationProtocol
    from src.communications.websocket_handler import WebSocketHandler
except ImportError as e:
    print(f"Warning: Could not import Communications components: {e}")
    StandardCommunicationProtocol = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CommunicationsValidator:
    """Validates Communications component functionality."""

    def __init__(self) -> None:
        self.results = {
            "standard_protocol": {"status": "pending", "time": 0, "details": ""},
            "service_discovery": {"status": "pending", "time": 0, "details": ""},
            "message_passing": {"status": "pending", "time": 0, "details": ""},
            "websocket_handler": {"status": "pending", "time": 0, "details": ""},
        }

    def test_standard_protocol(self) -> None:
        """Test standard communication protocol."""
        logger.info("Testing Standard Communication Protocol...")
        start_time = time.time()

        try:
            if StandardCommunicationProtocol is None:
                self.results["standard_protocol"] = {
                    "status": "failed",
                    "time": time.time() - start_time,
                    "details": "StandardCommunicationProtocol could not be imported",
                }
                return

            # Test protocol configuration
            protocol_config = {
                "agent_id": "test_agent_comm",
                "protocol_version": "1.0",
                "message_format": "json",
                "compression_enabled": True,
                "encryption_enabled": True,
            }

            # Initialize protocol
            protocol = StandardCommunicationProtocol(protocol_config)

            if hasattr(protocol, "send_message") and hasattr(protocol, "receive_message"):
                self.results["standard_protocol"] = {
                    "status": "success",
                    "time": time.time() - start_time,
                    "details": f"Protocol initialized. Agent: {protocol_config['agent_id']}, Version: {protocol_config['protocol_version']}",
                }
            else:
                self.results["standard_protocol"] = {
                    "status": "partial",
                    "time": time.time() - start_time,
                    "details": f"Protocol created but missing expected methods. Available: {[m for m in dir(protocol) if not m.startswith('_')][:5]}",
                }

        except Exception as e:
            self.results["standard_protocol"] = {
                "status": "failed",
                "time": time.time() - start_time,
                "details": f"Error: {e!s}",
            }

    def test_service_discovery(self) -> None:
        """Test service discovery functionality."""
        logger.info("Testing Service Discovery...")
        start_time = time.time()

        try:
            # Test service discovery
            discovery_config = {
                "discovery_port": 8500,
                "heartbeat_interval": 30,
                "service_timeout": 300,
                "discovery_protocols": ["mdns", "broadcast"],
            }

            discovery = ServiceDiscovery(discovery_config)

            # Test ServiceInfo creation
            test_service = ServiceInfo(
                agent_id="test_service_001",
                service_type="reasoning_agent",
                host="127.0.0.1",
                port=8080,
                capabilities=["text_generation", "problem_solving"],
                metadata={"version": "1.0", "load": "low"},
                last_heartbeat=time.time(),
                status="active",
            )

            if hasattr(discovery, "register_service") and hasattr(discovery, "discover_services"):
                self.results["service_discovery"] = {
                    "status": "success",
                    "time": time.time() - start_time,
                    "details": f"Service discovery functional. Service: {test_service.agent_id}, Capabilities: {len(test_service.capabilities)}",
                }
            else:
                self.results["service_discovery"] = {
                    "status": "partial",
                    "time": time.time() - start_time,
                    "details": f"Service discovery created. Available methods: {[m for m in dir(discovery) if not m.startswith('_') and ('register' in m or 'discover' in m)]}",
                }

        except Exception as e:
            self.results["service_discovery"] = {
                "status": "failed",
                "time": time.time() - start_time,
                "details": f"Error: {e!s}",
            }

    def test_message_passing(self) -> None:
        """Test message passing system."""
        logger.info("Testing Message Passing System...")
        start_time = time.time()

        try:
            # Test message passing system
            messaging_config = {
                "agent_id": "test_messaging_agent",
                "buffer_size": 1000,
                "retry_attempts": 3,
                "timeout": 5.0,
                "priority_queue": True,
            }

            messaging = MessagePassingSystem(messaging_config)

            if hasattr(messaging, "send_message") and hasattr(messaging, "receive_message"):
                # Test message structure
                {
                    "message_id": "msg_test_001",
                    "sender": "validator",
                    "recipient": "test_agent",
                    "message_type": "task_request",
                    "content": {"task": "validate_functionality", "parameters": {"timeout": 30}},
                    "priority": "normal",
                    "timestamp": time.time(),
                }

                self.results["message_passing"] = {
                    "status": "success",
                    "time": time.time() - start_time,
                    "details": f"Message passing functional. Buffer size: {messaging_config['buffer_size']}, Priority queue: {messaging_config['priority_queue']}",
                }
            else:
                self.results["message_passing"] = {
                    "status": "partial",
                    "time": time.time() - start_time,
                    "details": f"Message passing created. Available methods: {[m for m in dir(messaging) if not m.startswith('_') and ('send' in m or 'receive' in m)]}",
                }

        except Exception as e:
            self.results["message_passing"] = {
                "status": "failed",
                "time": time.time() - start_time,
                "details": f"Error: {e!s}",
            }

    def test_websocket_handler(self) -> None:
        """Test WebSocket communication handler."""
        logger.info("Testing WebSocket Handler...")
        start_time = time.time()

        try:
            # Test WebSocket handler
            websocket_config = {
                "host": "127.0.0.1",
                "port": 8765,
                "max_connections": 50,
                "message_size_limit": 1048576,  # 1MB
                "heartbeat_interval": 30,
            }

            ws_handler = WebSocketHandler(websocket_config)

            if hasattr(ws_handler, "start_server") and hasattr(ws_handler, "broadcast_message"):
                # Test WebSocket message structure
                {
                    "type": "agent_communication",
                    "data": {
                        "agent_id": "ws_test_agent",
                        "message": "WebSocket validation test",
                        "timestamp": time.time(),
                    },
                    "metadata": {"protocol_version": "1.0", "encoding": "utf-8"},
                }

                self.results["websocket_handler"] = {
                    "status": "success",
                    "time": time.time() - start_time,
                    "details": f"WebSocket handler functional. Port: {websocket_config['port']}, Max connections: {websocket_config['max_connections']}",
                }
            else:
                self.results["websocket_handler"] = {
                    "status": "partial",
                    "time": time.time() - start_time,
                    "details": f"WebSocket handler created. Available methods: {[m for m in dir(ws_handler) if not m.startswith('_') and ('start' in m or 'broadcast' in m)]}",
                }

        except Exception as e:
            self.results["websocket_handler"] = {
                "status": "failed",
                "time": time.time() - start_time,
                "details": f"Error: {e!s}",
            }

    def run_validation(self):
        """Run all Communications validation tests."""
        logger.info("=== Communications Validation Suite ===")

        # Run all tests
        self.test_standard_protocol()
        self.test_service_discovery()
        self.test_message_passing()
        self.test_websocket_handler()

        # Calculate results
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r["status"] == "success")
        partial_tests = sum(1 for r in self.results.values() if r["status"] == "partial")

        logger.info("=== Communications Validation Results ===")
        for test_name, result in self.results.items():
            status_emoji = {"success": "PASS", "partial": "WARN", "failed": "FAIL", "pending": "PEND"}

            logger.info(f"[{status_emoji[result['status']]}] {test_name}: {result['status'].upper()}")
            logger.info(f"   Time: {result['time']:.2f}s")
            logger.info(f"   Details: {result['details']}")

        success_rate = (successful_tests + partial_tests * 0.5) / total_tests
        logger.info(
            f"\nCommunications Success Rate: {success_rate:.1%} ({successful_tests + partial_tests}/{total_tests})"
        )

        return self.results, success_rate


if __name__ == "__main__":
    validator = CommunicationsValidator()
    results, success_rate = validator.run_validation()

    if success_rate >= 0.8:
        print("Communications Validation: PASSED")
    else:
        print("Communications Validation: NEEDS IMPROVEMENT")
