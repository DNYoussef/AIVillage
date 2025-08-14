"""Tests for RAG Offline Defaults - Prompt B

Comprehensive validation that RAG system works completely offline without
external dependencies, APIs, or network access.

Integration Point: Validates offline readiness for Phase 4 testing
"""

import json
import os

# Add src to path for imports
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from production.rag.rag_system.core.offline_defaults import (
    OfflineDefaultsManager,
    OfflineRAGConfig,
    auto_configure_for_environment,
    get_offline_rag_config,
    is_offline_environment,
    setup_offline_rag_environment,
    validate_offline_readiness,
)


class TestOfflineRAGConfig:
    """Test offline RAG configuration."""

    def test_offline_config_creation(self):
        """Test that offline config can be created with proper defaults."""
        config = OfflineRAGConfig()

        # Verify offline-first settings
        assert config.enable_internet_features is False
        assert config.enable_api_calls is False
        assert config.enable_cloud_sync is False
        assert config.cache_enabled is True
        assert config.cache_type == "local_disk"
        assert config.vector_store_type == "faiss_local"
        assert config.graph_store_type == "networkx_local"

    def test_offline_config_model_settings(self):
        """Test offline-compatible model settings."""
        config = OfflineRAGConfig()

        # Verify CPU-compatible models
        assert "sentence-transformers" in config.embedding_model
        assert config.model_name == "local_llm"
        assert config.temperature <= 0.5  # Conservative for offline
        assert config.max_tokens <= 1024  # Resource-conscious

    def test_offline_config_mobile_optimizations(self):
        """Test mobile/resource optimization settings."""
        config = OfflineRAGConfig()

        assert config.enable_mobile_optimizations is True
        assert config.low_memory_mode is True
        assert config.chunk_size <= 512  # Small chunks for mobile
        assert config.cache_max_size_mb <= 1024  # Reasonable cache size

    def test_config_customization(self):
        """Test that offline config can be customized."""
        custom_config = OfflineRAGConfig(
            chunk_size=128, cache_max_size_mb=256, strict_offline_mode=True
        )

        assert custom_config.chunk_size == 128
        assert custom_config.cache_max_size_mb == 256
        assert custom_config.strict_offline_mode is True


class TestOfflineDefaultsManager:
    """Test offline defaults manager functionality."""

    def test_manager_initialization(self):
        """Test manager initialization with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineDefaultsManager(temp_dir)

            assert manager.base_data_dir == Path(temp_dir)
            assert isinstance(manager.config, OfflineRAGConfig)

            # Check that directories were created
            expected_dirs = [
                "vector_store",
                "graph_store",
                "cache",
                "models",
                "logs",
                "documents",
                "embeddings",
            ]

            for dir_name in expected_dirs:
                assert (Path(temp_dir) / dir_name).exists()

    def test_get_offline_config_with_overrides(self):
        """Test getting config with overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineDefaultsManager(temp_dir)

            config = manager.get_offline_config(
                chunk_size=64, temperature=0.1, custom_param="test_value"
            )

            assert config.chunk_size == 64
            assert config.temperature == 0.1
            assert config.extra_params["custom_param"] == "test_value"

            # Verify paths are set correctly
            assert config.vector_store_path == str(Path(temp_dir) / "vector_store")
            assert config.graph_store_path == str(Path(temp_dir) / "graph_store")

    def test_validate_offline_readiness_empty_environment(self):
        """Test validation with empty environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineDefaultsManager(temp_dir)
            results = manager.validate_offline_readiness()

            # Should have warnings about missing models but directories should exist
            assert "ready_for_offline" in results
            assert "missing_components" in results
            assert "warnings" in results
            assert "recommendations" in results
            assert "storage_info" in results

            # Directory checks should pass (they were created)
            assert (
                len([w for w in results["warnings"] if "No local models found" in w])
                > 0
            )

    def test_validate_offline_readiness_with_disk_space_check(self):
        """Test validation includes disk space checking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineDefaultsManager(temp_dir)
            results = manager.validate_offline_readiness()

            # Should include disk space information
            assert "disk_space" in results["storage_info"]
            disk_info = results["storage_info"]["disk_space"]
            assert "total_gb" in disk_info
            assert "used_gb" in disk_info
            assert "free_gb" in disk_info
            assert all(isinstance(v, float) for v in disk_info.values())

    @patch("importlib.util.find_spec")
    def test_validate_offline_readiness_missing_dependencies(self, mock_find_spec):
        """Test validation detects missing dependencies."""
        # Mock missing dependencies
        mock_find_spec.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineDefaultsManager(temp_dir)
            results = manager.validate_offline_readiness()

            # Should detect missing dependencies
            assert results["ready_for_offline"] is False
            assert len(results["missing_components"]) > 0

    def test_setup_offline_environment(self):
        """Test complete offline environment setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineDefaultsManager(temp_dir)
            results = manager.setup_offline_environment()

            assert "success" in results
            assert "components_initialized" in results
            assert "errors" in results
            assert "warnings" in results

            # Check that configuration file was created
            config_file = Path(temp_dir) / "offline_config.json"
            assert config_file.exists()

            # Verify config file content
            with open(config_file) as f:
                config_data = json.load(f)
            assert config_data["enable_internet_features"] is False
            assert config_data["cache_enabled"] is True


class TestOfflineUtilities:
    """Test offline utility functions."""

    def test_get_offline_rag_config_function(self):
        """Test main config function."""
        config = get_offline_rag_config(chunk_size=256)

        assert isinstance(config, OfflineRAGConfig)
        assert config.chunk_size == 256
        assert config.enable_internet_features is False

    def test_validate_offline_readiness_function(self):
        """Test validation function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = validate_offline_readiness(temp_dir)

            assert isinstance(results, dict)
            assert "ready_for_offline" in results

    def test_setup_offline_rag_environment_function(self):
        """Test setup function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = setup_offline_rag_environment(temp_dir)

            assert isinstance(results, dict)
            assert "success" in results

    @patch("socket.create_connection")
    def test_is_offline_environment_no_network(self, mock_connection):
        """Test offline detection when network unavailable."""
        mock_connection.side_effect = OSError("Network unreachable")

        assert is_offline_environment() is True

    @patch("socket.create_connection")
    def test_is_offline_environment_with_network(self, mock_connection):
        """Test offline detection when network available."""
        mock_connection.return_value = MagicMock()

        assert is_offline_environment() is False

    def test_is_offline_environment_env_var(self):
        """Test offline detection via environment variable."""
        with patch.dict(os.environ, {"RAG_OFFLINE_MODE": "1"}):
            assert is_offline_environment() is True

        with patch.dict(os.environ, {"OFFLINE": "1"}):
            assert is_offline_environment() is True

    @patch("production.rag.rag_system.core.offline_defaults.is_offline_environment")
    def test_auto_configure_offline(self, mock_is_offline):
        """Test auto-configuration for offline environment."""
        mock_is_offline.return_value = True

        config = auto_configure_for_environment()
        assert isinstance(config, OfflineRAGConfig)
        assert config.strict_offline_mode is True
        assert config.enable_internet_features is False

    @patch("production.rag.rag_system.core.offline_defaults.is_offline_environment")
    def test_auto_configure_online(self, mock_is_offline):
        """Test auto-configuration for online environment."""
        mock_is_offline.return_value = False

        config = auto_configure_for_environment()
        assert isinstance(config, OfflineRAGConfig)
        assert config.strict_offline_mode is False
        assert config.enable_internet_features is True


class TestOfflineIntegration:
    """Test integration scenarios for offline RAG."""

    def test_complete_offline_setup_workflow(self):
        """Test complete workflow from setup to validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Setup environment
            setup_results = setup_offline_rag_environment(temp_dir)
            assert setup_results["success"] is True

            # Step 2: Validate readiness
            validation_results = validate_offline_readiness(temp_dir)

            # Step 3: Get configuration
            config = get_offline_rag_config()
            assert isinstance(config, OfflineRAGConfig)

            # Verify end-to-end consistency
            assert Path(temp_dir).exists()
            expected_files = [
                "offline_config.json",
            ]
            for filename in expected_files:
                assert (Path(temp_dir) / filename).exists()

    def test_offline_config_serialization(self):
        """Test that offline config can be serialized and deserialized."""
        original_config = get_offline_rag_config(
            chunk_size=128, temperature=0.2, custom_setting="test"
        )

        # Test dict conversion
        config_dict = original_config.dict()
        assert config_dict["chunk_size"] == 128
        assert config_dict["temperature"] == 0.2
        assert config_dict["extra_params"]["custom_setting"] == "test"

        # Test JSON serialization
        import json

        json_str = json.dumps(config_dict, default=str)
        loaded_dict = json.loads(json_str)

        # Verify critical fields preserved
        assert loaded_dict["enable_internet_features"] is False
        assert loaded_dict["cache_enabled"] is True
        assert loaded_dict["chunk_size"] == 128

    def test_offline_graceful_degradation(self):
        """Test graceful degradation when optional components missing."""
        config = get_offline_rag_config(
            enable_graceful_degradation=True, strict_offline_mode=False
        )

        # Should allow graceful degradation
        assert config.enable_graceful_degradation is True
        assert config.strict_offline_mode is False

        # But still be offline-first
        assert config.enable_api_calls is False
        assert config.cache_type == "local_disk"


if __name__ == "__main__":
    # Run offline defaults validation
    print("=== Testing RAG Offline Defaults ===")

    # Test basic configuration
    print("Testing offline configuration...")
    config = get_offline_rag_config()
    print(f"✓ Offline config created: cache_enabled={config.cache_enabled}")

    # Test environment detection
    print("Testing environment detection...")
    is_offline = is_offline_environment()
    print(f"✓ Environment detection: offline={is_offline}")

    # Test auto-configuration
    print("Testing auto-configuration...")
    auto_config = auto_configure_for_environment()
    print(f"✓ Auto-config: strict_offline={auto_config.strict_offline_mode}")

    print("=== All offline defaults tests completed ===")
