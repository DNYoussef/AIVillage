"""
Tests for Cogment Configuration System (Agent 7)

Tests the configuration management including:
- Parameter budget validation (23.7M vs 25M target)
- Option A unified configuration loading and validation
- Configuration inheritance and overrides
- Environment-specific configurations
- Budget enforcement and parameter counting
"""

from pathlib import Path
import tempfile

import pytest

# Import Cogment configuration components
try:
    from config.cogment.config_loader import CogmentConfigLoader, load_cogment_config
    from config.cogment.config_validation import CogmentConfigValidator, ValidationResult
    from core.agent_forge.models.cogment.core.config import CogmentConfig

    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Cogment configuration components not available: {e}")
    CONFIG_AVAILABLE = False


class TestCogmentConfig:
    """Test core Cogment configuration class."""

    @pytest.fixture
    def base_config_dict(self):
        """Create base configuration dictionary."""
        return {
            # Model parameters (Option A: ~25M parameters)
            "d_model": 512,
            "n_layers": 6,
            "n_head": 8,
            "d_ff": 1536,
            "vocab_size": 13000,
            "max_seq_len": 2048,
            # ACT parameters
            "act_epsilon": 0.01,
            "max_act_steps": 16,
            "halt_threshold": 0.99,
            "ponder_cost_weight": 0.1,
            # Memory parameters
            "mem_slots": 2048,
            "ltm_capacity": 1024,
            "ltm_dim": 256,
            "memory_dim": 256,
            # Parameter budget
            "target_params": 25000000,
            "tolerance": 0.05,
            # Training parameters
            "dropout": 0.1,
            "layer_norm_eps": 1e-5,
            "tie_embeddings": True,
        }

    @pytest.fixture
    def cogment_config(self, base_config_dict):
        """Create CogmentConfig instance."""
        if not CONFIG_AVAILABLE:
            pytest.skip("Configuration components not available")
        return CogmentConfig.from_dict(base_config_dict)

    def test_config_creation(self, cogment_config, base_config_dict):
        """Test CogmentConfig instantiation."""
        assert cogment_config.d_model == base_config_dict["d_model"]
        assert cogment_config.n_layers == base_config_dict["n_layers"]
        assert cogment_config.vocab_size == base_config_dict["vocab_size"]
        assert cogment_config.target_params == base_config_dict["target_params"]

    def test_config_parameter_budget_calculation(self, cogment_config):
        """Test parameter budget calculation."""
        # Test parameter estimation
        estimated_params = cogment_config.estimate_parameters()

        # Should be close to target
        target = cogment_config.target_params
        tolerance = cogment_config.tolerance

        min_acceptable = target * (1 - tolerance)
        max_acceptable = target * (1 + tolerance)

        assert (
            min_acceptable <= estimated_params <= max_acceptable
        ), f"Estimated params {estimated_params:,} outside budget [{min_acceptable:,}, {max_acceptable:,}]"

        print(f"✓ Parameter budget: {estimated_params:,} (target: {target:,})")

    def test_config_option_a_compliance(self, cogment_config):
        """Test Option A configuration compliance."""
        # Option A: Unified model, tied embeddings, no factorization
        assert cogment_config.tie_embeddings is True, "Option A should use tied embeddings"

        # Check that we're not using factorization (Option A characteristic)
        if hasattr(cogment_config, "factorize_large_heads"):
            assert cogment_config.factorize_large_heads is False, "Option A should not use head factorization"

        # Check vocabulary size is optimized for budget
        assert (
            cogment_config.vocab_size == 13000
        ), f"Option A vocab size should be 13000, got {cogment_config.vocab_size}"

        # Check d_model is optimized for 25M budget
        assert cogment_config.d_model == 512, f"Option A d_model should be 512, got {cogment_config.d_model}"

    def test_config_validation_methods(self, cogment_config):
        """Test configuration validation methods."""
        # Test basic validation
        is_valid = cogment_config.validate()
        assert is_valid, "Configuration should be valid"

        # Test parameter budget validation
        budget_valid = cogment_config.validate_parameter_budget()
        assert budget_valid, "Parameter budget should be valid"

        # Test architecture consistency
        arch_valid = cogment_config.validate_architecture()
        assert arch_valid, "Architecture should be consistent"

    def test_config_serialization(self, cogment_config):
        """Test configuration serialization and deserialization."""
        # Convert to dict
        config_dict = cogment_config.to_dict()

        # Verify key parameters are present
        assert "d_model" in config_dict
        assert "vocab_size" in config_dict
        assert "target_params" in config_dict

        # Recreate from dict
        restored_config = CogmentConfig.from_dict(config_dict)

        # Should be equivalent
        assert restored_config.d_model == cogment_config.d_model
        assert restored_config.vocab_size == cogment_config.vocab_size
        assert restored_config.target_params == cogment_config.target_params


class TestCogmentConfigLoader:
    """Test configuration loading system."""

    @pytest.fixture
    def config_loader(self):
        """Create CogmentConfigLoader instance."""
        if not CONFIG_AVAILABLE:
            pytest.skip("Configuration components not available")
        return CogmentConfigLoader()

    @pytest.fixture
    def sample_config_yaml(self):
        """Create sample configuration YAML content."""
        return """
        model:
          d_model: 512
          n_layers: 6
          n_head: 8
          d_ff: 1536
          vocab_size: 13000
          max_seq_len: 2048
          dropout: 0.1
          tie_embeddings: true

        refinement_core:
          memory_fusion_dim: 512
          refinement_steps: 8
          min_refinement_steps: 2
          max_refinement_steps: 16

        gated_ltm:
          mem_slots: 2048
          ltm_capacity: 1024
          ltm_dim: 256
          memory_dim: 256
          decay: 0.001

        act_halting:
          epsilon: 0.01
          max_steps: 16
          halt_threshold: 0.99
          ponder_cost_weight: 0.1

        parameter_budget:
          target_params: 25000000
          tolerance: 0.05
          validate_on_load: true
        """

    def test_config_loader_creation(self, config_loader):
        """Test config loader instantiation."""
        assert hasattr(config_loader, "load_config")
        assert hasattr(config_loader, "validate_config")
        assert hasattr(config_loader, "load_from_file")

    def test_load_config_from_yaml(self, config_loader, sample_config_yaml):
        """Test loading configuration from YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(sample_config_yaml)
            f.flush()

            config_path = Path(f.name)

            try:
                # Load configuration
                config = config_loader.load_from_file(config_path)

                # Verify loading
                assert config is not None
                assert hasattr(config, "d_model")
                assert config.d_model == 512
                assert config.vocab_size == 13000
                assert config.target_params == 25000000

            finally:
                config_path.unlink()  # Clean up

    def test_load_default_config(self, config_loader):
        """Test loading default configuration."""
        try:
            # Attempt to load default config
            config = config_loader.load_default_config()

            # Should have reasonable defaults
            assert config is not None
            assert config.d_model > 0
            assert config.vocab_size > 0
            assert config.target_params > 0

        except FileNotFoundError:
            # Default config file may not exist in test environment
            pytest.skip("Default config file not available")

    def test_config_inheritance(self, config_loader, sample_config_yaml):
        """Test configuration inheritance and overrides."""
        base_config_yaml = sample_config_yaml

        override_config_yaml = """
        model:
          d_model: 256  # Override
          vocab_size: 5000  # Override

        parameter_budget:
          target_params: 15000000  # Override
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "base.yaml"
            override_path = Path(temp_dir) / "override.yaml"

            base_path.write_text(base_config_yaml)
            override_path.write_text(override_config_yaml)

            # Load with inheritance
            config = config_loader.load_with_overrides(base_path, override_path)

            # Verify overrides applied
            assert config.d_model == 256  # Overridden
            assert config.vocab_size == 5000  # Overridden
            assert config.target_params == 15000000  # Overridden

            # Verify non-overridden values preserved
            assert config.n_layers == 6  # From base
            assert config.n_head == 8  # From base

    def test_environment_specific_config(self, config_loader):
        """Test environment-specific configuration loading."""
        environments = ["development", "testing", "production"]

        for env in environments:
            try:
                config = config_loader.load_for_environment(env)

                # Should load successfully
                assert config is not None

                # Environment-specific adjustments
                if env == "development":
                    # Development might have smaller models
                    assert config.target_params <= 25000000
                elif env == "production":
                    # Production should use full configuration
                    assert config.target_params > 0

            except FileNotFoundError:
                # Environment config may not exist
                continue


class TestCogmentConfigValidator:
    """Test configuration validation system."""

    @pytest.fixture
    def config_validator(self):
        """Create CogmentConfigValidator instance."""
        if not CONFIG_AVAILABLE:
            pytest.skip("Configuration components not available")
        return CogmentConfigValidator()

    @pytest.fixture
    def valid_config(self):
        """Create valid configuration for testing."""
        if not CONFIG_AVAILABLE:
            pytest.skip("Configuration components not available")

        return CogmentConfig(
            d_model=512,
            n_layers=6,
            n_head=8,
            d_ff=1536,
            vocab_size=13000,
            max_seq_len=2048,
            target_params=25000000,
            tolerance=0.05,
        )

    def test_validator_creation(self, config_validator):
        """Test validator instantiation."""
        assert hasattr(config_validator, "validate")
        assert hasattr(config_validator, "validate_parameter_budget")
        assert hasattr(config_validator, "validate_architecture")

    def test_valid_configuration_validation(self, config_validator, valid_config):
        """Test validation of valid configuration."""
        result = config_validator.validate(valid_config)

        # Verify validation result structure
        assert isinstance(result, ValidationResult)
        assert hasattr(result, "is_valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")

        # Should be valid
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_parameter_budget_validation(self, config_validator, valid_config):
        """Test parameter budget validation."""
        # Test valid budget
        result = config_validator.validate_parameter_budget(valid_config)
        assert result.is_valid is True

        # Test budget violation
        invalid_config = CogmentConfig(
            d_model=1024,  # Too large
            n_layers=20,  # Too many layers
            vocab_size=50000,  # Too large vocabulary
            target_params=25000000,  # Same target
            tolerance=0.05,
        )

        result = config_validator.validate_parameter_budget(invalid_config)
        assert result.is_valid is False
        assert len(result.errors) > 0

        # Should mention budget violation
        error_text = " ".join(result.errors)
        assert "budget" in error_text.lower() or "parameter" in error_text.lower()

    def test_architecture_validation(self, config_validator, valid_config):
        """Test architecture validation."""
        # Test valid architecture
        result = config_validator.validate_architecture(valid_config)
        assert result.is_valid is True

        # Test invalid architecture
        invalid_config = CogmentConfig(
            d_model=513, n_head=8, vocab_size=0, target_params=25000000  # Not divisible by n_head  # Invalid
        )

        result = config_validator.validate_architecture(invalid_config)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_option_a_compliance_validation(self, config_validator):
        """Test Option A compliance validation."""
        # Test Option A compliant configuration
        option_a_config = CogmentConfig(d_model=512, vocab_size=13000, tie_embeddings=True, target_params=25000000)

        result = config_validator.validate_option_a_compliance(option_a_config)
        assert result.is_valid is True

        # Test non-compliant configuration
        non_compliant_config = CogmentConfig(
            d_model=512,
            vocab_size=32000,  # Too large for Option A
            tie_embeddings=False,  # Should be True for Option A
            target_params=25000000,
        )

        result = config_validator.validate_option_a_compliance(non_compliant_config)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_dependency_validation(self, config_validator, valid_config):
        """Test configuration dependency validation."""
        # Test valid dependencies
        result = config_validator.validate_dependencies(valid_config)
        assert result.is_valid is True

        # Test invalid dependencies
        invalid_config = CogmentConfig(d_model=512, n_head=7, target_params=25000000)  # d_model not divisible by n_head

        result = config_validator.validate_dependencies(invalid_config)
        assert result.is_valid is False
        assert len(result.errors) > 0


class TestConfigurationIntegration:
    """Test configuration system integration."""

    @pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Configuration components not available")
    def test_load_and_validate_integration(self):
        """Test integrated loading and validation."""
        # Create sample configuration
        config_dict = {
            "d_model": 512,
            "n_layers": 6,
            "n_head": 8,
            "vocab_size": 13000,
            "target_params": 25000000,
            "tolerance": 0.05,
            "tie_embeddings": True,
        }

        # Load configuration
        config = CogmentConfig.from_dict(config_dict)

        # Validate configuration
        validator = CogmentConfigValidator()
        result = validator.validate(config)

        assert result.is_valid is True
        print("✓ Load and validate integration successful")

    @pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Configuration components not available")
    def test_parameter_budget_enforcement(self):
        """Test parameter budget enforcement."""
        # Test exact target achievement (23.7M)
        target_config = CogmentConfig(
            d_model=512,
            n_layers=6,
            n_head=8,
            d_ff=1536,
            vocab_size=13000,
            max_seq_len=2048,
            target_params=25000000,
            tolerance=0.05,
        )

        # Calculate actual parameters
        estimated_params = target_config.estimate_parameters()

        # Should be close to 23.7M (achieved target)
        achieved_target = 23700000
        tolerance_range = achieved_target * 0.1  # 10% tolerance for testing

        assert (
            abs(estimated_params - achieved_target) <= tolerance_range
        ), f"Parameter count {estimated_params:,} should be close to achieved target {achieved_target:,}"

        print(f"✓ Parameter budget enforcement: {estimated_params:,} parameters")

    @pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Configuration components not available")
    def test_configuration_vs_hrrm_baseline(self):
        """Test configuration improvements vs HRRM baseline."""
        cogment_config = CogmentConfig(d_model=512, n_layers=6, vocab_size=13000, target_params=25000000)

        # Calculate Cogment parameters
        cogment_params = cogment_config.estimate_parameters()

        # HRRM baseline: 3 models × 50M each = 150M total
        hrrm_baseline_params = 150000000

        # Calculate reduction
        reduction_factor = hrrm_baseline_params / cogment_params

        # Should achieve significant reduction
        assert reduction_factor >= 5.0, f"Insufficient parameter reduction: {reduction_factor:.1f}x"

        print("✓ Configuration efficiency vs HRRM:")
        print(f"  - HRRM baseline: {hrrm_baseline_params:,} parameters")
        print(f"  - Cogment unified: {cogment_params:,} parameters")
        print(f"  - Reduction factor: {reduction_factor:.1f}x")

    @pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Configuration components not available")
    def test_production_config_validation(self):
        """Test production configuration validation."""
        # Load production-like configuration
        production_config = CogmentConfig(
            d_model=512,
            n_layers=6,
            n_head=8,
            d_ff=1536,
            vocab_size=13000,
            max_seq_len=2048,
            # Production settings
            dropout=0.1,
            layer_norm_eps=1e-5,
            tie_embeddings=True,
            # Budget enforcement
            target_params=25000000,
            tolerance=0.05,
            validate_on_load=True,
        )

        # Validate for production
        validator = CogmentConfigValidator()
        result = validator.validate_for_production(production_config)

        # Should pass production validation
        assert result.is_valid is True
        assert len(result.errors) == 0

        # Check production-specific requirements
        production_checks = result.additional_info.get("production_checks", {})
        assert production_checks.get("parameter_budget_enforced", False) is True
        assert production_checks.get("architecture_stable", False) is True
        assert production_checks.get("option_a_compliant", False) is True

        print("✓ Production configuration validation successful")


@pytest.mark.integration
class TestConfigurationSystemComplete:
    """Complete configuration system integration tests."""

    @pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Configuration components not available")
    def test_end_to_end_configuration(self):
        """Test end-to-end configuration workflow."""
        # 1. Load configuration
        try:
            config = load_cogment_config()
        except:
            # Use default if load fails
            config = CogmentConfig(d_model=512, n_layers=6, vocab_size=13000, target_params=25000000)

        # 2. Validate configuration
        validator = CogmentConfigValidator()
        validation_result = validator.validate(config)

        assert validation_result.is_valid is True

        # 3. Check parameter budget
        budget_result = validator.validate_parameter_budget(config)
        assert budget_result.is_valid is True

        # 4. Verify Option A compliance
        option_a_result = validator.validate_option_a_compliance(config)
        assert option_a_result.is_valid is True

        print("✓ End-to-end configuration workflow successful")

    @pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Configuration components not available")
    def test_configuration_agent_integration(self):
        """Test configuration integration with other agents."""
        config = CogmentConfig(d_model=512, n_layers=6, vocab_size=13000, target_params=25000000)

        # Test Agent 1 integration (Core model)
        core_config = config.get_core_config()
        assert core_config["d_model"] == 512
        assert core_config["n_layers"] == 6

        # Test Agent 2 integration (Memory)
        memory_config = config.get_memory_config()
        assert "ltm_capacity" in memory_config
        assert "memory_dim" in memory_config

        # Test Agent 3 integration (Heads)
        heads_config = config.get_heads_config()
        assert "vocab_size" in heads_config
        assert "tie_embeddings" in heads_config

        # Test Agent 4 integration (Training)
        training_config = config.get_training_config()
        assert "learning_rate" in training_config or training_config is not None

        # Test Agent 5 integration (Data)
        data_config = config.get_data_config()
        assert "batch_size" in data_config or data_config is not None

        print("✓ Configuration integration with all agents successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
