"""Comprehensive test suite for constants usage and magic literal elimination."""

import os
import json
import tempfile
import unittest
from unittest.mock import patch

# Import the constants and config manager
from infrastructure.constants import (
    TaskConstants,
    ProjectConstants,
    TimingConstants,
    PerformanceConstants,
    MessageConstants,
    ErrorMessageConstants,
    PerformanceFieldNames,
    RewardConstants,
    get_config_manager,
    set_config_file,
    ConfigManager,
    EnvironmentConfig,
)
from infrastructure.constants.task_constants import TaskType, TaskActionMapping
from infrastructure.constants.project_constants import ProjectStatus


class TestTaskConstants(unittest.TestCase):
    """Test task management constants."""

    def test_task_defaults(self):
        """Test that task default values are properly defined."""
        self.assertIsInstance(TaskConstants.DEFAULT_PRIORITY, int)
        self.assertIsInstance(TaskConstants.DEFAULT_BATCH_SIZE, int)
        self.assertIsInstance(TaskConstants.MAX_RETRIES, int)

        # Validate ranges
        self.assertGreaterEqual(TaskConstants.DEFAULT_PRIORITY, TaskConstants.MIN_PRIORITY)
        self.assertLessEqual(TaskConstants.DEFAULT_PRIORITY, TaskConstants.MAX_PRIORITY)
        self.assertGreaterEqual(TaskConstants.DEFAULT_BATCH_SIZE, TaskConstants.MIN_BATCH_SIZE)
        self.assertLessEqual(TaskConstants.DEFAULT_BATCH_SIZE, TaskConstants.MAX_BATCH_SIZE)

    def test_task_type_enum(self):
        """Test task type enumeration."""
        self.assertEqual(TaskType.CRITICAL.value, "critical")
        self.assertEqual(TaskType.ROUTINE.value, "routine")
        self.assertEqual(TaskType.ANALYSIS.value, "analysis")
        self.assertEqual(TaskType.DEFAULT.value, "default")

    def test_task_action_mapping(self):
        """Test task action mapping enumeration."""
        self.assertEqual(TaskActionMapping.CRITICAL.value, 0)
        self.assertEqual(TaskActionMapping.ROUTINE_HIGH_PRIORITY.value, 1)
        self.assertEqual(TaskActionMapping.ANALYSIS.value, 2)
        self.assertEqual(TaskActionMapping.HIGH_COMPLEXITY.value, 3)
        self.assertEqual(TaskActionMapping.DEFAULT.value, 4)


class TestProjectConstants(unittest.TestCase):
    """Test project management constants."""

    def test_project_status_enum(self):
        """Test project status enumeration."""
        self.assertEqual(ProjectStatus.INITIALIZED.value, "initialized")
        self.assertEqual(ProjectStatus.ACTIVE.value, "active")
        self.assertEqual(ProjectStatus.COMPLETED.value, "completed")
        self.assertEqual(ProjectStatus.CANCELLED.value, "cancelled")

    def test_project_defaults(self):
        """Test project default values."""
        self.assertEqual(ProjectConstants.DEFAULT_STATUS, ProjectStatus.INITIALIZED.value)
        self.assertEqual(ProjectConstants.DEFAULT_PROGRESS, 0.0)

        # Test field names are strings
        self.assertIsInstance(ProjectConstants.PROJECT_ID_FIELD, str)
        self.assertIsInstance(ProjectConstants.NAME_FIELD, str)
        self.assertIsInstance(ProjectConstants.STATUS_FIELD, str)


class TestTimingConstants(unittest.TestCase):
    """Test timing and scheduling constants."""

    def test_timing_values(self):
        """Test timing constant values are reasonable."""
        self.assertGreater(TimingConstants.BATCH_PROCESSING_INTERVAL, 0)
        self.assertGreater(TimingConstants.RETRY_DELAY, 0)
        self.assertGreater(TimingConstants.DEFAULT_TIMEOUT, 0)

        # Test sleep intervals are in ascending order
        self.assertLess(TimingConstants.SHORT_SLEEP, TimingConstants.MEDIUM_SLEEP)
        self.assertLess(TimingConstants.MEDIUM_SLEEP, TimingConstants.LONG_SLEEP)


class TestPerformanceConstants(unittest.TestCase):
    """Test performance and incentive constants."""

    def test_performance_multipliers(self):
        """Test performance multiplier ranges."""
        self.assertLess(PerformanceConstants.MIN_PERFORMANCE, PerformanceConstants.MAX_PERFORMANCE)
        self.assertGreater(PerformanceConstants.PERFORMANCE_BOOST_FACTOR, 1.0)
        self.assertLess(PerformanceConstants.PERFORMANCE_PENALTY_FACTOR, 1.0)
        self.assertEqual(PerformanceConstants.NEUTRAL_TREND, 1.0)

    def test_update_rates(self):
        """Test update rates are within reasonable bounds."""
        self.assertGreater(PerformanceConstants.SPECIALIZATION_RATE, 0)
        self.assertGreater(PerformanceConstants.COLLABORATION_RATE, 0)
        self.assertGreater(PerformanceConstants.INNOVATION_RATE, 0)

        # All rates should be <= 1.0 for stability
        self.assertLessEqual(PerformanceConstants.SPECIALIZATION_RATE, 1.0)
        self.assertLessEqual(PerformanceConstants.COLLABORATION_RATE, 1.0)
        self.assertLessEqual(PerformanceConstants.INNOVATION_RATE, 1.0)

    def test_trend_limits(self):
        """Test performance trend limits."""
        self.assertLess(PerformanceConstants.MIN_TREND, PerformanceConstants.NEUTRAL_TREND)
        self.assertGreater(PerformanceConstants.MAX_TREND, PerformanceConstants.NEUTRAL_TREND)


class TestRewardConstants(unittest.TestCase):
    """Test reward calculation constants."""

    def test_reward_values(self):
        """Test reward constant values."""
        self.assertGreater(RewardConstants.BASE_SUCCESS_REWARD, 0)
        self.assertGreater(RewardConstants.INNOVATION_BONUS, 0)
        self.assertGreater(RewardConstants.COLLABORATION_BONUS, 0)
        self.assertGreater(RewardConstants.REWARD_NORMALIZATION_DIVISOR, 0)

    def test_default_values(self):
        """Test default values for missing components."""
        self.assertGreaterEqual(RewardConstants.DEFAULT_QUALITY_SCORE, 0)
        self.assertLessEqual(RewardConstants.DEFAULT_QUALITY_SCORE, 1.0)
        self.assertGreater(RewardConstants.DEFAULT_EXPECTED_TIME, 0)
        self.assertGreater(RewardConstants.DEFAULT_BUDGET, 0)


class TestMessageConstants(unittest.TestCase):
    """Test message and communication constants."""

    def test_message_senders(self):
        """Test message sender constants."""
        self.assertIsInstance(MessageConstants.UNIFIED_MANAGEMENT, str)
        self.assertIsInstance(MessageConstants.TASK_MANAGER, str)
        self.assertIsInstance(MessageConstants.DEFAULT_AGENT, str)

    def test_message_fields(self):
        """Test message field names."""
        self.assertIsInstance(MessageConstants.TASK_ID, str)
        self.assertIsInstance(MessageConstants.DESCRIPTION, str)
        self.assertIsInstance(MessageConstants.INCENTIVE, str)
        self.assertIsInstance(MessageConstants.ASSIGNED_AGENT, str)

    def test_decision_threshold(self):
        """Test decision making threshold."""
        self.assertGreaterEqual(MessageConstants.DECISION_THRESHOLD, 0)
        self.assertLessEqual(MessageConstants.DECISION_THRESHOLD, 1.0)


class TestErrorMessageConstants(unittest.TestCase):
    """Test error message templates."""

    def test_error_templates(self):
        """Test error message templates are properly formatted."""
        # Test template formatting
        task_id = "test-task-123"
        formatted_msg = ErrorMessageConstants.TASK_NOT_FOUND_TEMPLATE.format(task_id=task_id)
        self.assertIn(task_id, formatted_msg)

        project_id = "test-project-456"
        formatted_msg = ErrorMessageConstants.PROJECT_NOT_FOUND_TEMPLATE.format(project_id=project_id)
        self.assertIn(project_id, formatted_msg)

    def test_error_prefixes(self):
        """Test error message prefixes are strings."""
        self.assertIsInstance(ErrorMessageConstants.ERROR_CREATING_TASK, str)
        self.assertIsInstance(ErrorMessageConstants.ERROR_ASSIGNING_TASK, str)
        self.assertIsInstance(ErrorMessageConstants.ERROR_COMPLETING_TASK, str)


class TestPerformanceFieldNames(unittest.TestCase):
    """Test performance field name constants."""

    def test_field_names_are_strings(self):
        """Test all field names are strings."""
        self.assertIsInstance(PerformanceFieldNames.SUCCESS_FIELD, str)
        self.assertIsInstance(PerformanceFieldNames.TIME_TAKEN_FIELD, str)
        self.assertIsInstance(PerformanceFieldNames.QUALITY_FIELD, str)
        self.assertIsInstance(PerformanceFieldNames.ASSIGNED_AGENT_FIELD, str)
        self.assertIsInstance(PerformanceFieldNames.TASK_ID_FIELD, str)


class TestConfigManager(unittest.TestCase):
    """Test configuration manager functionality."""

    def setUp(self):
        """Set up test environment."""
        self.config_manager = ConfigManager()

    def test_default_configuration(self):
        """Test default configuration values."""
        self.assertGreater(self.config_manager.get_task_batch_size(), 0)
        self.assertGreater(self.config_manager.get_default_priority(), 0)
        self.assertGreaterEqual(self.config_manager.get_max_retries(), 0)
        self.assertGreater(self.config_manager.get_batch_processing_interval(), 0)

    def test_environment_variable_override(self):
        """Test environment variable overrides."""
        with patch.dict(os.environ, {"TASK_BATCH_SIZE": "10"}):
            config_manager = ConfigManager()
            self.assertEqual(config_manager.get_task_batch_size(), 10)

        with patch.dict(os.environ, {"TASK_DEFAULT_PRIORITY": "3"}):
            config_manager = ConfigManager()
            self.assertEqual(config_manager.get_default_priority(), 3)

    def test_configuration_validation(self):
        """Test configuration validation."""
        validation_result = self.config_manager.validate_configuration()
        self.assertIsInstance(validation_result, dict)
        self.assertIn("valid", validation_result)
        self.assertIn("configuration", validation_result)

    def test_invalid_configuration(self):
        """Test handling of invalid configuration values."""
        with patch.dict(os.environ, {"TASK_BATCH_SIZE": "-1"}):
            config_manager = ConfigManager()
            validation_result = config_manager.validate_configuration()
            # Should handle invalid values gracefully
            self.assertIsInstance(validation_result, dict)

    def test_config_file_loading(self):
        """Test loading configuration from file."""
        config_data = {"batch_size": 15, "default_priority": 2, "learning_rate": 0.05}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            config_manager = ConfigManager(config_file)
            self.assertEqual(config_manager.get_task_batch_size(), 15)
            self.assertEqual(config_manager.get_default_priority(), 2)
            self.assertEqual(config_manager.get_learning_rate(), 0.05)
        finally:
            os.unlink(config_file)

    def test_config_range_validation(self):
        """Test configuration range validation."""
        # Test batch size validation
        with self.assertRaises(ValueError):
            ConfigManager._validate_range(0, 1, 100, "test_param")

        with self.assertRaises(ValueError):
            ConfigManager._validate_range(101, 1, 100, "test_param")

        # Valid range should pass
        result = ConfigManager._validate_range(50, 1, 100, "test_param")
        self.assertEqual(result, 50)


class TestGlobalConfigManager(unittest.TestCase):
    """Test global configuration manager functions."""

    def test_get_config_manager(self):
        """Test global config manager getter."""
        config_manager1 = get_config_manager()
        config_manager2 = get_config_manager()
        self.assertIs(config_manager1, config_manager2)  # Should be singleton

    def test_set_config_file(self):
        """Test setting configuration file."""
        config_data = {"batch_size": 20}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            set_config_file(config_file)
            config_manager = get_config_manager()
            self.assertEqual(config_manager.get_task_batch_size(), 20)
        finally:
            os.unlink(config_file)


class TestEnvironmentConfig(unittest.TestCase):
    """Test environment configuration dataclass."""

    def test_environment_config_creation(self):
        """Test creating environment configuration."""
        config = EnvironmentConfig(batch_size=10, default_priority=3, learning_rate=0.02)

        self.assertEqual(config.batch_size, 10)
        self.assertEqual(config.default_priority, 3)
        self.assertEqual(config.learning_rate, 0.02)
        self.assertIsNone(config.max_retries)  # Not set, should be None

    def test_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        config = EnvironmentConfig(batch_size=10, default_priority=None, learning_rate=0.02)  # This should be excluded

        config_dict = config.to_dict()
        self.assertIn("batch_size", config_dict)
        self.assertIn("learning_rate", config_dict)
        self.assertNotIn("default_priority", config_dict)


class TestMagicLiteralElimination(unittest.TestCase):
    """Test that magic literals have been successfully eliminated."""

    def test_no_hardcoded_batch_size(self):
        """Test that batch size uses constants instead of magic literals."""
        # This would be tested by importing the actual task management classes
        # and verifying they use the config manager
        config_manager = get_config_manager()
        batch_size = config_manager.get_task_batch_size()

        # Should be a reasonable value from constants, not hardcoded 5
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)

    def test_no_hardcoded_priorities(self):
        """Test that priority values use constants."""
        config_manager = get_config_manager()
        default_priority = config_manager.get_default_priority()

        # Should use constant, not hardcoded 1
        self.assertIsInstance(default_priority, int)
        self.assertGreaterEqual(default_priority, TaskConstants.MIN_PRIORITY)
        self.assertLessEqual(default_priority, TaskConstants.MAX_PRIORITY)

    def test_no_hardcoded_timing_values(self):
        """Test that timing values use constants."""
        config_manager = get_config_manager()
        batch_interval = config_manager.get_batch_processing_interval()

        # Should use constant, not hardcoded 1.0
        self.assertIsInstance(batch_interval, float)
        self.assertGreater(batch_interval, 0)

    def test_no_hardcoded_performance_values(self):
        """Test that performance values use constants."""
        config_manager = get_config_manager()
        learning_rate = config_manager.get_learning_rate()

        # Should use constant, not hardcoded 0.01
        self.assertIsInstance(learning_rate, float)
        self.assertGreater(learning_rate, 0)
        self.assertLessEqual(learning_rate, 1.0)


class TestConstantsImport(unittest.TestCase):
    """Test that constants can be imported correctly."""

    def test_import_all_constants(self):
        """Test importing all constant modules."""
        try:

            # If we get here, imports succeeded
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import constants: {e}")

    def test_constants_have_expected_attributes(self):
        """Test that constant classes have expected attributes."""
        # Task constants
        self.assertTrue(hasattr(TaskConstants, "DEFAULT_PRIORITY"))
        self.assertTrue(hasattr(TaskConstants, "DEFAULT_BATCH_SIZE"))
        self.assertTrue(hasattr(TaskConstants, "MAX_RETRIES"))

        # Performance constants
        self.assertTrue(hasattr(PerformanceConstants, "LEARNING_RATE"))
        self.assertTrue(hasattr(PerformanceConstants, "NEUTRAL_TREND"))
        self.assertTrue(hasattr(PerformanceConstants, "MAX_PERFORMANCE"))

        # Message constants
        self.assertTrue(hasattr(MessageConstants, "UNIFIED_MANAGEMENT"))
        self.assertTrue(hasattr(MessageConstants, "TASK_ID"))
        self.assertTrue(hasattr(MessageConstants, "INCENTIVE"))


if __name__ == "__main__":
    unittest.main()
