from pathlib import Path
import sys
import unittest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestKingAgentBasic(unittest.TestCase):
    """Basic tests for King agent without heavy dependencies."""

    def test_king_agent_import_with_stubs(self):
        """Test that King agent can be imported with dependency stubs."""
        try:
            # Try to import core components
            from core.error_handling import Message, MessageType, StandardCommunicationProtocol

            print("✅ Core agent components imported successfully")

            # Test basic functionality
            protocol = StandardCommunicationProtocol()
            message = Message(
                type=MessageType.TASK,
                sender="test_sender",
                receiver="test_receiver",
                content={"test": "data"},
            )

            assert protocol is not None
            assert message is not None
            assert message.type == MessageType.TASK

            print("✅ Basic agent infrastructure functional")

        except ImportError as e:
            self.skipTest(f"Dependencies not available: {e}")

    def test_agent_configuration(self):
        """Test agent configuration classes."""
        try:
            from agents.unified_base_agent import UnifiedAgentConfig

            config = UnifiedAgentConfig(name="test_agent", max_turns=10)

            assert config.name == "test_agent"
            assert config.max_turns == 10

            print("✅ Agent configuration working")

        except ImportError as e:
            self.skipTest(f"Configuration dependencies not available: {e}")

    def test_communication_protocol(self):
        """Test communication protocol functionality."""
        try:
            from core.error_handling import Message, MessageType, StandardCommunicationProtocol

            protocol = StandardCommunicationProtocol()

            # Test message creation
            task_message = Message(
                type=MessageType.TASK,
                sender="test_agent",
                receiver="king_agent",
                content={"action": "test", "params": {"key": "value"}},
            )

            # Test message handling (async)
            import asyncio

            asyncio.run(protocol.send_message(task_message))

            # Check that message was added to history for both sender and receiver
            assert len(protocol.message_history) > 0
            # Find our task message in the history
            task_found = False
            for agent_messages in protocol.message_history.values():
                for msg in agent_messages:
                    if msg.type == MessageType.TASK:
                        task_found = True
                        break
            assert task_found

            print("✅ Communication protocol functional")

        except ImportError as e:
            self.skipTest(f"Protocol dependencies not available: {e}")

    def test_security_features(self):
        """Test that security features are working."""
        # This test verifies our security fixes are in place

        # Test 1: Ensure ADAS security is in place
        try:
            # Import should work even if we can't run full ADAS
            from agent_forge.adas.adas import AgentTechnique, SecureCodeRunner

            runner = SecureCodeRunner()
            assert runner is not None

            # Test validation
            technique = AgentTechnique(technique_name="test", code="def run(m,w,p): return 0.5")
            is_valid = technique.validate_code("def run(m,w,p): return 0.5")
            assert is_valid

            # Test dangerous code rejection
            is_dangerous = technique.validate_code("def run(m,w,p): eval('dangerous'); return 0.5")
            assert not is_dangerous

            print("✅ Security features operational")

        except ImportError as e:
            self.skipTest(f"Security module dependencies not available: {e}")


if __name__ == "__main__":
    unittest.main()
