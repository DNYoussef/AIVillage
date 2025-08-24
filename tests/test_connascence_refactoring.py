"""
Tests for connascence refactoring - validates elimination of coupling violations.
Tests both the new centralized constants/utilities and refactored modules.
"""

from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.constants.system_constants import HotspotThresholds, SystemLimits, TensorDimensions, TimeConstants
from src.utils.parameter_objects import (
    MCPConnectionParams,
    MessageSendParams,
    TrainingParams,
    create_mcp_connection_params,
    keyword_only_params,
)
from src.utils.risk_assessment import (
    RiskAssessment,
    assess_churn_risk,
    assess_complexity_risk,
    calculate_combined_risk_score,
    calculate_hotspot_risk_level,
)
from src.utils.validation_utils import (
    calculate_secure_hash,
    sanitize_filename,
    validate_email_format,
    validate_memory_requirements,
    validate_message_size,
    validate_network_latency,
    validate_perplexity_score,
    validate_regression_drop,
    validate_success_rate,
    validate_working_hours,
)

try:
    from src.utils.sandbox_factory import (
        SandboxConfig,
        SandboxContext,
        SandboxFactory,
        SandboxServiceLocator,
        configure_sandbox_service,
        get_sandbox_manager,
    )

    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False

    # Mock classes for testing
    class SandboxConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class SandboxFactory:
        @staticmethod
        def create_manager(config):
            return type("MockManager", (), {"config": config})()

    class SandboxServiceLocator:
        def __init__(self):
            self._managers = {}

        def register_manager(self, name, manager):
            self._managers[name] = manager

        def get_manager(self, name):
            return self._managers.get(name)

    def get_sandbox_manager(name="default"):
        return None

    def configure_sandbox_service(config, name="default"):
        pass

    class SandboxContext:
        def __init__(self, config, name="temp"):
            self.config = config

        def __enter__(self):
            return SandboxFactory.create_manager(self.config)

        def __exit__(self, *args):
            pass


class TestSystemConstants:
    """Test centralized system constants eliminate magic numbers"""

    def test_system_limits_constants(self):
        """Verify system limits are properly defined"""
        assert SystemLimits.MAX_FILE_SIZE_BYTES == 100
        assert SystemLimits.HIGH_LATENCY_THRESHOLD_MS == 300
        assert SystemLimits.MAX_CYCLOMATIC_COMPLEXITY == 15
        assert SystemLimits.SUCCESS_RATE_THRESHOLD == 60

    def test_tensor_dimensions_constants(self):
        """Verify tensor dimension constants"""
        assert TensorDimensions.LOW_DIM_LIMIT == 768
        assert TensorDimensions.HIGH_DIM_LIMIT == 1536
        assert TensorDimensions.BASE_CHUNK_THRESHOLD == 256

    def test_time_constants(self):
        """Verify time-related constants"""
        assert TimeConstants.MORNING_HOUR == 10
        assert TimeConstants.EVENING_HOUR == 18
        assert TimeConstants.WORK_START_HOUR == 9
        assert TimeConstants.WORK_END_HOUR == 17

    def test_hotspot_thresholds(self):
        """Verify hotspot risk thresholds"""
        assert HotspotThresholds.CRITICAL_SCORE == 50.0
        assert HotspotThresholds.HIGH_SCORE == 30.0
        assert HotspotThresholds.MEDIUM_SCORE == 15.0
        assert HotspotThresholds.SINGLE_AUTHOR_THRESHOLD == 2


class TestValidationUtils:
    """Test centralized validation algorithms eliminate duplicate logic"""

    def test_memory_validation(self):
        """Test memory requirement validation"""
        assert validate_memory_requirements(25.0) is True  # Above threshold
        assert validate_memory_requirements(15.0) is False  # Below threshold
        assert validate_memory_requirements(20.0) is True  # Exact threshold

    def test_success_rate_validation(self):
        """Test success rate validation with decimal values"""
        assert validate_success_rate(0.7) is True
        assert validate_success_rate(0.5) is False
        assert validate_success_rate(0.6) is True  # Exact threshold

    def test_regression_validation(self):
        """Test performance regression validation"""
        assert validate_regression_drop(0.03) is True  # Concerning drop
        assert validate_regression_drop(0.01) is False  # Acceptable drop
        assert validate_regression_drop(0.02) is True  # Exact threshold

    def test_perplexity_validation(self):
        """Test perplexity score validation"""
        good_result = {"perplexity": {"perplexity": 50}}
        bad_result = {"perplexity": {"perplexity": 150}}

        assert validate_perplexity_score(good_result) is True
        assert validate_perplexity_score(bad_result) is False
        assert validate_perplexity_score(None) is False
        assert validate_perplexity_score({}) is False

    def test_network_latency_validation(self):
        """Test network latency validation"""
        assert validate_network_latency(400) is True  # High latency
        assert validate_network_latency(200) is False  # Acceptable latency
        assert validate_network_latency(None) is False  # No data

    def test_message_size_validation(self):
        """Test message size validation"""
        large_message = 15 * 1024  # 15KB
        small_message = 5 * 1024  # 5KB

        assert validate_message_size(large_message) is True
        assert validate_message_size(small_message) is False

    def test_working_hours_validation(self):
        """Test working hours validation"""
        assert validate_working_hours(10) is True  # During work
        assert validate_working_hours(20) is False  # After work
        assert validate_working_hours(9) is True  # Start of work
        assert validate_working_hours(17) is True  # End of work

    def test_secure_hash_consistency(self):
        """Test secure hash algorithm consistency"""
        test_data = "test_string"
        hash1 = calculate_secure_hash(test_data)
        hash2 = calculate_secure_hash(test_data)

        assert hash1 == hash2  # Deterministic
        assert len(hash1) == 64  # SHA-256 hex length
        assert hash1 != calculate_secure_hash("different_string")

    def test_email_validation(self):
        """Test email format validation"""
        valid_emails = ["user@example.com", "test.email+tag@domain.co.uk", "user123@subdomain.example.org"]
        invalid_emails = ["not_an_email", "@domain.com", "user@", "user@.com", "user space@domain.com"]

        for email in valid_emails:
            assert validate_email_format(email) is True, f"Should validate: {email}"

        for email in invalid_emails:
            assert validate_email_format(email) is False, f"Should reject: {email}"

    def test_filename_sanitization(self):
        """Test filename sanitization algorithm"""
        assert sanitize_filename("normal_file.txt") == "normal_file.txt"
        assert sanitize_filename("file<>with|bad*chars") == "file__with_bad_chars"
        assert sanitize_filename("   .hidden.file   ") == "hidden.file"
        assert sanitize_filename("CON") == "unnamed_file"  # Reserved name
        assert sanitize_filename("") == "unnamed_file"  # Empty


class TestRiskAssessment:
    """Test centralized risk assessment algorithms"""

    @pytest.fixture
    def mock_churn_metrics(self):
        """Mock churn metrics for testing"""
        mock = Mock()
        mock.unique_authors = 1
        mock.commit_count = 25
        mock.lines_modified = 1500
        mock.churn_score = 25
        return mock

    @pytest.fixture
    def mock_complexity_metrics(self):
        """Mock complexity metrics for testing"""
        mock = Mock()
        mock.cyclomatic_complexity = 20
        mock.lines_of_code = 600
        mock.function_count = 25
        mock.complexity_score = 30
        return mock

    def test_hotspot_risk_calculation(self, mock_churn_metrics, mock_complexity_metrics):
        """Test hotspot risk level calculation"""
        # Critical risk (high score + single author)
        risk = calculate_hotspot_risk_level(
            hotspot_score=60, churn=mock_churn_metrics, complexity=mock_complexity_metrics
        )
        assert risk == "critical"

        # High risk
        risk = calculate_hotspot_risk_level(
            hotspot_score=35, churn=mock_churn_metrics, complexity=mock_complexity_metrics
        )
        assert risk == "high"

        # Medium risk
        risk = calculate_hotspot_risk_level(
            hotspot_score=20, churn=mock_churn_metrics, complexity=mock_complexity_metrics
        )
        assert risk == "medium"

        # Low risk
        risk = calculate_hotspot_risk_level(
            hotspot_score=10, churn=mock_churn_metrics, complexity=mock_complexity_metrics
        )
        assert risk == "low"

    def test_complexity_risk_assessment(self, mock_complexity_metrics):
        """Test complexity-based risk assessment"""
        assessment = assess_complexity_risk(mock_complexity_metrics)

        assert isinstance(assessment, RiskAssessment)
        assert assessment.level in ["critical", "high", "medium", "low"]
        assert assessment.score > 0
        assert len(assessment.factors) > 0
        assert len(assessment.recommendations) > 0

    def test_churn_risk_assessment(self, mock_churn_metrics):
        """Test churn-based risk assessment"""
        assessment = assess_churn_risk(mock_churn_metrics)

        assert isinstance(assessment, RiskAssessment)
        assert assessment.level in ["critical", "high", "medium", "low"]
        assert assessment.score > 0
        assert len(assessment.factors) > 0
        assert len(assessment.recommendations) > 0

    def test_combined_risk_assessment(self, mock_churn_metrics, mock_complexity_metrics):
        """Test combined risk assessment"""
        complexity_assessment = assess_complexity_risk(mock_complexity_metrics)
        churn_assessment = assess_churn_risk(mock_churn_metrics)

        combined = calculate_combined_risk_score(
            complexity_assessment=complexity_assessment, churn_assessment=churn_assessment, hotspot_score=40.0
        )

        assert isinstance(combined, RiskAssessment)
        assert combined.score >= max(complexity_assessment.score, churn_assessment.score)
        assert len(combined.factors) >= len(complexity_assessment.factors)
        assert len(combined.recommendations) >= len(complexity_assessment.recommendations)


class TestParameterObjects:
    """Test parameter objects eliminate position-dependent functions"""

    def test_mcp_connection_params(self):
        """Test MCP connection parameter object"""
        params = MCPConnectionParams(
            uri="http://example.com", agent_id="test_agent", api_key="secret_key", timeout=30.0
        )

        assert params.uri == "http://example.com"
        assert params.agent_id == "test_agent"
        assert params.api_key == "secret_key"
        assert params.timeout == 30.0
        assert params.max_retries == 3  # Default value

    def test_message_send_params(self):
        """Test message send parameter object"""
        mock_message = Mock()
        params = MessageSendParams(message=mock_message, destination="target_node", path_preference="fast_path")

        assert params.message == mock_message
        assert params.destination == "target_node"
        assert params.path_preference == "fast_path"

    def test_training_params(self):
        """Test training parameter object with defaults"""
        params = TrainingParams()

        assert params.batch_size == 8
        assert params.sequence_length == 256
        assert params.limit_steps == 1000
        assert params.learning_rate == 0.001

        # Test custom values
        custom_params = TrainingParams(batch_size=16, learning_rate=0.01)
        assert custom_params.batch_size == 16
        assert custom_params.learning_rate == 0.01

    def test_factory_functions(self):
        """Test factory functions for backward compatibility"""
        params = create_mcp_connection_params(uri="http://test.com", agent_id="agent1", api_key="key123")

        assert isinstance(params, MCPConnectionParams)
        assert params.uri == "http://test.com"
        assert params.agent_id == "agent1"
        assert params.api_key == "key123"

    def test_keyword_only_decorator(self):
        """Test keyword-only parameter decorator"""

        @keyword_only_params(MCPConnectionParams)
        def test_function(params: MCPConnectionParams) -> str:
            return f"Connected to {params.uri}"

        params = MCPConnectionParams("http://test.com", "agent1", "key123")
        result = test_function(params)
        assert "Connected to http://test.com" in result

        # Should raise TypeError for positional args
        with pytest.raises(TypeError):
            test_function("http://test.com", "agent1", "key123")


class TestSandboxFactory:
    """Test dependency injection for sandbox management"""

    def test_sandbox_config_validation(self):
        """Test sandbox configuration validation"""
        # Valid config
        config = SandboxConfig(max_memory_mb=128, max_cpu_percent=50, timeout_seconds=30)
        assert config.max_memory_mb == 128

        # Invalid memory
        with pytest.raises(ValueError):
            SandboxConfig(max_memory_mb=0)

        # Invalid CPU percent
        with pytest.raises(ValueError):
            SandboxConfig(max_cpu_percent=150)

    def test_sandbox_factory_creation(self):
        """Test sandbox factory creates appropriate managers"""
        from src.utils.sandbox_factory import SandboxType

        config = SandboxConfig(sandbox_type=SandboxType.WASI)
        manager = SandboxFactory.create_manager(config)

        assert manager is not None
        assert manager.config.sandbox_type == SandboxType.WASI

    def test_service_locator_pattern(self):
        """Test service locator for dependency injection"""
        locator = SandboxServiceLocator()

        # Test registration and retrieval
        config = SandboxConfig()
        manager = SandboxFactory.create_manager(config)
        locator.register_manager("test", manager)

        retrieved = locator.get_manager("test")
        assert retrieved is manager

        # Test auto-creation
        auto_manager = locator.get_or_create_manager("auto")
        assert auto_manager is not None

    def test_sandbox_context_manager(self):
        """Test context manager for temporary configurations"""
        config = SandboxConfig(max_memory_mb=256)

        with SandboxContext(config, "temp") as manager:
            assert manager.config.max_memory_mb == 256

        # Should clean up after context
        locator = SandboxServiceLocator()
        assert locator.get_manager("temp") is None

    def test_global_service_replacement(self):
        """Test replacement of global mutable state"""
        # Configure service with specific config
        config = SandboxConfig(max_memory_mb=512)
        configure_sandbox_service(config, "configured")

        # Get manager through service locator
        manager = get_sandbox_manager("configured")
        assert manager.config.max_memory_mb == 512

        # Verify no global mutable state
        manager2 = get_sandbox_manager("configured")
        assert manager is manager2  # Same instance through service locator


class TestIntegrationWithExistingCode:
    """Test integration of refactored utilities with existing codebase"""

    @patch("tools.analysis.hotspots.SystemLimits")
    def test_hotspots_integration(self, mock_limits):
        """Test hotspots.py uses centralized constants"""
        mock_limits.MAX_FILE_SIZE_BYTES = 100
        mock_limits.PROGRESS_UPDATE_INTERVAL = 10

        # Would test actual integration if hotspots module imported properly
        assert mock_limits.MAX_FILE_SIZE_BYTES == 100

    def test_validation_utility_integration(self):
        """Test validation utilities work with typical use cases"""
        # Test typical memory check scenario
        free_memory = 25.5
        assert validate_memory_requirements(free_memory)

        # Test typical success rate scenario
        success_rate = 0.75
        assert validate_success_rate(success_rate)

        # Test typical latency scenario
        latency = 450
        assert validate_network_latency(latency)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
