"""Simplified tests for core logging configuration.

This module provides basic tests for the logging configuration module,
focusing on the key functionality that can be tested reliably.
"""

import json
import logging
from unittest.mock import patch

from core.logging_config import (
    AIVillageLoggerAdapter,
    StructuredFormatter,
    get_component_logger,
    setup_aivillage_logging,
)


class TestStructuredFormatter:
    """Test the StructuredFormatter class."""

    def test_format_basic_message(self):
        """Test formatting a basic log message."""
        formatter = StructuredFormatter()

        # Create a logger and handler to generate real log records
        logger = logging.getLogger("test_formatter")
        logger.setLevel(logging.INFO)

        # Create a custom handler to capture the record
        class RecordCapture(logging.Handler):
            def __init__(self):
                super().__init__()
                self.record = None

            def emit(self, record):
                self.record = record

        handler = RecordCapture()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Log a message
        logger.info("Test message")

        # Format the captured record
        formatted = formatter.format(handler.record)

        # Parse and verify
        log_data = json.loads(formatted)
        assert log_data["level"] == "INFO"
        assert log_data["component"] == "test_formatter"
        assert log_data["message"] == "Test message"
        assert "timestamp" in log_data

        # Cleanup
        logger.removeHandler(handler)


class TestGetComponentLogger:
    """Test the get_component_logger function."""

    def test_get_component_logger_basic(self):
        """Test getting a basic component logger."""
        logger = get_component_logger("test_component")

        assert isinstance(logger, AIVillageLoggerAdapter)
        assert logger.name == "AIVillage.test_component"

    def test_get_component_logger_with_dots(self):
        """Test getting a logger with dots in name."""
        logger = get_component_logger("main.sub.component")

        assert logger.name == "AIVillage.main.sub.component"

    def test_multiple_calls_return_same_underlying_logger(self):
        """Test that multiple calls use the same underlying logger."""
        logger1 = get_component_logger("test_component")
        logger2 = get_component_logger("test_component")

        # The adapters may be different but they wrap the same logger
        assert logger1.logger is logger2.logger


class TestSetupAIVillageLogging:
    """Test the setup_aivillage_logging function."""

    def test_basic_setup(self):
        """Test basic logging setup."""
        logger = setup_aivillage_logging()

        assert isinstance(logger, logging.Logger)
        assert logger.name == "AIVillage"

    def test_setup_with_custom_level(self):
        """Test setup with custom log level."""
        logger = setup_aivillage_logging(log_level="DEBUG")

        # The logger should be configured
        assert logger.isEnabledFor(logging.DEBUG)

    @patch("logging.config.dictConfig")
    def test_configuration_structure(self, mock_dictConfig):
        """Test that proper configuration is passed."""
        setup_aivillage_logging()

        # Verify dictConfig was called
        assert mock_dictConfig.called

        # Get the configuration
        config = mock_dictConfig.call_args[0][0]

        # Verify basic structure
        assert config["version"] == 1
        assert "formatters" in config
        assert "handlers" in config
        assert "loggers" in config


class TestLoggingIntegration:
    """Integration tests for the logging system."""

    def test_component_logger_works(self):
        """Test that component logger can log messages."""
        # Setup logging
        setup_aivillage_logging(log_level="INFO")

        # Get a logger
        logger = get_component_logger("integration_test")

        # Should be able to log without errors
        logger.info("Test info message")
        logger.debug("Test debug message")  # Won't show at INFO level
        logger.warning("Test warning message")
        logger.error("Test error message")

    def test_logger_with_extra_context(self):
        """Test logging with extra context."""
        setup_aivillage_logging()

        logger = get_component_logger("context_test")

        # Log with extra fields - should not raise error
        logger.info("User action", extra={"user_id": "123", "action": "login"})


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_component_name(self):
        """Test logger with empty component name."""
        logger = get_component_logger("")

        # Should still work
        assert isinstance(logger, AIVillageLoggerAdapter)
        assert logger.name == "AIVillage."

    def test_special_characters_in_name(self):
        """Test logger with special characters."""
        logger = get_component_logger("test-component_2.0")

        assert isinstance(logger, AIVillageLoggerAdapter)
        assert logger.name == "AIVillage.test-component_2.0"

    @patch("core.logging_config.Path.mkdir", side_effect=PermissionError("No permission"))
    def test_setup_handles_permission_error(self, mock_mkdir):
        """Test that setup handles permission errors gracefully."""
        # Should handle permission error and fall back to console logging only
        logger = setup_aivillage_logging(log_dir="/restricted_path", enable_file=True)

        # Should still return a logger
        assert isinstance(logger, logging.Logger)

        # Should have called mkdir to try creating the directory
        mock_mkdir.assert_called_once()

        # The function should continue without file logging when permission is denied
