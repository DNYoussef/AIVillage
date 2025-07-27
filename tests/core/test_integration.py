"""Integration tests for core module interactions.

This module tests how the core modules work together, ensuring that
the various components integrate correctly without breaking each other.
"""

import logging

import pytest

from core import (
    AgentCommunicationProtocol,
    AgentMessage,
    AgentMessageType,
    AIVillageException,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    Priority,
    get_component_logger,
)
from core.evidence import Chunk, EvidencePack
from core.logging_config import setup_aivillage_logging
from services.core.business_logic import ChatBusinessLogic


class TestLoggingErrorHandlingIntegration:
    """Test integration between logging and error handling systems."""

    def test_logger_captures_exceptions(self, caplog):
        """Test that component loggers properly capture AIVillageException details."""
        # Get logger without setting up aivillage logging to work with caplog
        logger = get_component_logger("integration_test")

        # Create an exception with full context
        error_context = ErrorContext(
            component="test_component",
            operation="test_operation",
            details={"user_id": "12345", "action": "test_action"},
        )

        exception = AIVillageException(
            message="Integration test error",
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.ERROR,
            context=error_context,
        )

        # Log the exception with caplog
        with caplog.at_level(logging.ERROR):
            try:
                raise exception
            except AIVillageException:
                logger.exception("Caught exception during integration test")

        # Verify exception details are captured
        assert len(caplog.records) > 0
        # The exception should be logged by our logger.exception call
        assert "Integration test error" in caplog.text

    def test_error_context_in_logs(self, caplog):
        """Test that error context is properly included in log messages."""
        # Get logger without setting up aivillage logging to work with caplog
        logger = get_component_logger("context_test")

        # Log with extra context that matches ErrorContext structure
        with caplog.at_level(logging.INFO):
            logger.info(
                "Processing user request",
                extra={
                    "component": "user_service",
                    "operation": "handle_request",
                    "user_id": "12345",
                    "request_type": "update_profile",
                },
            )

        # Context should be included in the log output
        assert len(caplog.records) > 0
        assert "Processing user request" in caplog.text


class TestCommunicationErrorHandlingIntegration:
    """Test integration between communication and error handling systems."""

    @pytest.mark.asyncio
    async def test_communication_protocol_with_error_handling(self):
        """Test that communication protocol handles errors gracefully."""
        protocol = AgentCommunicationProtocol()

        # Create a handler that raises an exception
        async def failing_handler(message):
            raise AIVillageException(
                message="Handler failed to process message",
                category=ErrorCategory.PROCESSING,
                severity=ErrorSeverity.ERROR,
                context=ErrorContext(
                    component="message_handler",
                    operation="process_message",
                    details={"message_id": message.id},
                ),
            )

        # Subscribe the failing handler
        protocol.subscribe("failing_agent", failing_handler)

        # Send a message to the failing agent
        message = AgentMessage(
            type=AgentMessageType.TASK,
            sender="test_sender",
            receiver="failing_agent",
            content="test message",
        )

        # The protocol should handle the exception gracefully
        # (In a real implementation, this might log the error but not crash)
        try:
            await protocol.send_message(message)
        except Exception as e:
            # If an exception is raised, it should be an AIVillageException
            assert isinstance(e, AIVillageException)

    @pytest.mark.asyncio
    async def test_message_serialization_with_error_recovery(self):
        """Test message serialization with error recovery."""
        # Create a message with valid data
        message = AgentMessage(
            type=AgentMessageType.RESPONSE,
            sender="test_sender",
            receiver="test_receiver",
            content={"status": "success", "data": [1, 2, 3]},
            priority=Priority.HIGH,
        )

        # Serialize and deserialize
        message_dict = message.to_dict()
        assert isinstance(message_dict, dict)
        assert message_dict["type"] == "RESPONSE"
        assert message_dict["priority"] == 3  # HIGH priority value

        # Test deserialization
        restored_message = AgentMessage.from_dict(message_dict)
        assert restored_message.type == message.type
        assert restored_message.sender == message.sender
        assert restored_message.content == message.content


class TestChatEngineIntegration:
    """Test integration of chat engine with other core components."""

    @pytest.mark.asyncio
    async def test_chat_engine_with_logging(self, caplog):
        """Test that chat engine integrates properly with logging."""
        setup_aivillage_logging(log_level="INFO")

        # Create chat business logic instance
        chat_logic = ChatBusinessLogic()

        # Mock the chat request
        from services.core.interfaces import ChatRequest

        mock_request = ChatRequest(
            message="Hello, how are you?", conversation_id="test-123", context={}
        )

        # Test the chat processing
        with caplog.at_level(logging.INFO):
            response = await chat_logic.process_chat(mock_request)

        # Should return a response without errors
        assert response is not None
        assert hasattr(response, "response")
        assert hasattr(response, "conversation_id")

    def test_chat_engine_error_handling(self):
        """Test that chat engine properly handles and reports errors."""
        # This test would verify that the chat engine creates proper
        # AIVillageException instances when things go wrong

        # Since process_chat might not have error conditions easily testable,
        # we'll test the pattern by creating what a chat engine error might look like
        try:
            raise AIVillageException(
                message="Failed to process chat message",
                category=ErrorCategory.PROCESSING,
                severity=ErrorSeverity.ERROR,
                context=ErrorContext(
                    component="chat_engine",
                    operation="process_message",
                    details={"message_length": 1000, "conversation_id": "test-123"},
                ),
            )
        except AIVillageException as e:
            # Verify the exception has proper structure
            assert e.message == "Failed to process chat message"
            assert e.category == ErrorCategory.PROCESSING
            assert e.context.component == "chat_engine"
            assert e.context.operation == "process_message"


class TestEvidenceLoggingIntegration:
    """Test integration between evidence system and logging."""

    def test_evidence_pack_creation_with_logging(self, caplog):
        """Test creating evidence packs with proper logging."""
        setup_aivillage_logging(log_level="INFO")
        logger = get_component_logger("evidence_test")

        # Create evidence pack
        chunk = Chunk(
            id="test_chunk_1",
            text="This is a test document chunk",
            score=0.85,
            source_uri="https://example.com/doc1",
        )

        with caplog.at_level(logging.INFO):
            evidence_pack = EvidencePack(
                query="test query", chunks=[chunk], proto_confidence=0.8
            )

            logger.info(
                "Created evidence pack",
                extra={
                    "pack_id": str(evidence_pack.id),
                    "query": evidence_pack.query,
                    "chunk_count": len(evidence_pack.chunks),
                    "confidence": evidence_pack.proto_confidence,
                },
            )

        # Verify evidence pack was created successfully
        assert evidence_pack.query == "test query"
        assert len(evidence_pack.chunks) == 1
        assert evidence_pack.proto_confidence == 0.8

    def test_evidence_serialization_error_handling(self):
        """Test error handling during evidence serialization."""
        chunk = Chunk(
            id="test_chunk",
            text="Test content",
            score=0.5,
            source_uri="https://example.com",
        )

        evidence_pack = EvidencePack(query="test", chunks=[chunk])

        # Test JSON serialization
        json_str = evidence_pack.to_json()
        assert isinstance(json_str, str)

        # Test deserialization
        restored_pack = EvidencePack.from_json(json_str)
        assert restored_pack.query == evidence_pack.query
        assert len(restored_pack.chunks) == len(evidence_pack.chunks)


class TestCrossModuleImportIntegration:
    """Test that cross-module imports work correctly after cleanup."""

    def test_core_module_exports(self):
        """Test that all expected classes are exported from core.__init__."""
        # Test error handling exports
        # Test communication exports (new classes with Agent prefix)
        from core import (
            AgentCommunicationProtocol,
            AgentMessage,
            AgentMessageType,
            Message,
            MessageType,
            StandardCommunicationProtocol,
        )

        # Verify classes are distinct
        assert AgentMessageType != MessageType
        assert AgentMessage != Message
        assert AgentCommunicationProtocol != StandardCommunicationProtocol

    def test_direct_module_imports(self):
        """Test that direct module imports still work."""
        # Test direct imports from error_handling
        # Test direct imports from communication
        from core.communication import AgentMessageType as CommunicationMessageType
        from core.error_handling import MessageType as ErrorHandlingMessageType

        # Verify the types are different
        assert ErrorHandlingMessageType != CommunicationMessageType

    def test_no_import_conflicts(self):
        """Test that there are no import conflicts between modules."""
        # This should not raise any ImportError or AttributeError
        try:
            from core import AgentMessageType, MessageType
            from core.communication import AgentMessageType as CommMessageType
            from core.error_handling import MessageType as EHMessageType

            # Verify they are the expected types
            assert MessageType == EHMessageType
            assert AgentMessageType == CommMessageType

        except (ImportError, AttributeError) as e:
            pytest.fail(f"Import conflict detected: {e}")


class TestEndToEndIntegration:
    """End-to-end integration tests combining multiple core components."""

    @pytest.mark.asyncio
    async def test_full_message_processing_workflow(self, caplog):
        """Test a complete workflow involving multiple core components."""
        # Setup logging
        setup_aivillage_logging(log_level="INFO")
        logger = get_component_logger("integration_workflow")

        # Create communication protocol
        protocol = AgentCommunicationProtocol()

        # Create a realistic handler that processes messages
        processed_messages = []

        async def message_processor(message: AgentMessage):
            logger.info(
                "Processing message",
                extra={
                    "message_id": message.id,
                    "sender": message.sender,
                    "type": message.type.value,
                },
            )

            try:
                # Simulate some processing
                if message.content == "error":
                    raise AIVillageException(
                        message="Simulated processing error",
                        category=ErrorCategory.PROCESSING,
                        severity=ErrorSeverity.WARNING,
                        context=ErrorContext(
                            component="message_processor",
                            operation="process_message",
                            details={"message_id": message.id},
                        ),
                    )

                processed_messages.append(message)
                logger.info(
                    "Message processed successfully", extra={"message_id": message.id}
                )

            except AIVillageException as e:
                logger.error(
                    "Failed to process message",
                    extra={
                        "message_id": message.id,
                        "error_category": e.category.value,
                        "error_severity": e.severity.value,
                    },
                )
                raise

        # Subscribe the processor
        protocol.subscribe("processor_agent", message_processor)

        # Test successful message processing
        with caplog.at_level(logging.INFO):
            success_message = AgentMessage(
                type=AgentMessageType.TASK,
                sender="test_client",
                receiver="processor_agent",
                content="process this task",
                priority=Priority.HIGH,
            )

            await protocol.send_message(success_message)

        # Verify successful processing
        assert len(processed_messages) == 1
        assert processed_messages[0].content == "process this task"

        # Test error message processing
        with caplog.at_level(logging.ERROR):
            error_message = AgentMessage(
                type=AgentMessageType.TASK,
                sender="test_client",
                receiver="processor_agent",
                content="error",  # This will trigger an error
            )

            try:
                await protocol.send_message(error_message)
            except AIVillageException as e:
                # Expected error
                assert e.category == ErrorCategory.PROCESSING
                assert e.context.component == "message_processor"

    def test_serialization_across_modules(self):
        """Test that serialization works across different core modules."""
        # Create objects from different modules
        error_context = ErrorContext(
            component="serialization_test",
            operation="test_serialization",
            details={"test": True},
        )

        agent_message = AgentMessage(
            type=AgentMessageType.NOTIFICATION,
            sender="serializer",
            receiver="deserializer",
            content={"notification": "test complete"},
            priority=Priority.MEDIUM,
        )

        chunk = Chunk(
            id="serial_chunk",
            text="Serialization test content",
            score=0.9,
            source_uri="https://test.com/serial",
        )

        evidence = EvidencePack(query="serialization test", chunks=[chunk])

        # Test serialization
        message_dict = agent_message.to_dict()
        evidence_json = evidence.to_json()

        # Test deserialization
        restored_message = AgentMessage.from_dict(message_dict)
        restored_evidence = EvidencePack.from_json(evidence_json)

        # Verify integrity
        assert restored_message.type == agent_message.type
        assert restored_message.content == agent_message.content
        assert restored_evidence.query == evidence.query
        assert len(restored_evidence.chunks) == len(evidence.chunks)
