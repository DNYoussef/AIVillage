"""Tests for unified configuration system."""

import pytest
import os
from pathlib import Path
import yaml
from typing import Dict, Any

from config.unified_config import (
    UnifiedConfig,
    ModelConfig,
    AgentConfig,
    RAGConfig,
    DatabaseConfig,
    PerformanceConfig,
    ModelType,
    AgentType
)

@pytest.fixture
def config_dir(tmp_path):
    """Create temporary config directory with test files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create default.yaml
    default_config = {
        "environment": "development",
        "system": {
            "log_level": "INFO",
            "max_memory_usage": 0.9,
            "max_cpu_usage": 0.8
        },
        "rag_system": {
            "retrieval_depth": 3,
            "relevance_threshold": 0.7,
            "feedback_enabled": True
        },
        "database": {
            "path": "data/agent_data.db",
            "backup_interval": 24,
            "max_backup_count": 7
        },
        "performance": {
            "metrics": {
                "response_quality": 0.4,
                "task_completion": 0.3
            }
        }
    }
    
    with open(config_dir / "default.yaml", "w") as f:
        yaml.dump(default_config, f)
    
    # Create openrouter_agents.yaml
    agent_config = {
        "agents": {
            "king": {
                "frontier_model": "nvidia/llama-3.1-nemotron-70b-instruct",
                "local_model": "Qwen/Qwen2.5-3B-Instruct",
                "description": "Advanced instruction-following model",
                "settings": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            },
            "sage": {
                "frontier_model": "anthropic/claude-3.5-sonnet",
                "local_model": "deepseek-ai/Janus-1.3B",
                "description": "Research focused model",
                "settings": {
                    "temperature": 0.8,
                    "max_tokens": 2000
                }
            }
        }
    }
    
    with open(config_dir / "openrouter_agents.yaml", "w") as f:
        yaml.dump(agent_config, f)
    
    return config_dir

@pytest.fixture
def unified_config(config_dir, monkeypatch):
    """Create UnifiedConfig instance with test configuration."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
    return UnifiedConfig(str(config_dir))

def test_config_loading(unified_config):
    """Test basic configuration loading."""
    assert unified_config.config["environment"] == "development"
    assert unified_config.config["system"]["log_level"] == "INFO"
    assert len(unified_config.agents) == 2  # king and sage

def test_agent_config(unified_config):
    """Test agent configuration access."""
    king_config = unified_config.get_agent_config("king")
    assert isinstance(king_config, AgentConfig)
    assert king_config.type == AgentType.KING
    assert king_config.frontier_model.name == "nvidia/llama-3.1-nemotron-70b-instruct"
    assert king_config.local_model.name == "Qwen/Qwen2.5-3B-Instruct"

def test_model_config(unified_config):
    """Test model configuration handling."""
    king_config = unified_config.get_agent_config("king")
    
    # Test frontier model config
    frontier_model = king_config.frontier_model
    assert isinstance(frontier_model, ModelConfig)
    assert frontier_model.type == ModelType.FRONTIER
    assert frontier_model.temperature == 0.7
    assert frontier_model.max_tokens == 1000
    
    # Test local model config
    local_model = king_config.local_model
    assert isinstance(local_model, ModelConfig)
    assert local_model.type == ModelType.LOCAL

def test_rag_config(unified_config):
    """Test RAG system configuration."""
    rag_config = unified_config.get_rag_config()
    assert isinstance(rag_config, RAGConfig)
    assert rag_config.retrieval_depth == 3
    assert rag_config.relevance_threshold == 0.7
    assert rag_config.feedback_enabled is True

def test_database_config(unified_config):
    """Test database configuration."""
    db_config = unified_config.get_db_config()
    assert isinstance(db_config, DatabaseConfig)
    assert db_config.path == "data/agent_data.db"
    assert db_config.backup_interval == 24
    assert db_config.max_backup_count == 7

def test_performance_config(unified_config):
    """Test performance tracking configuration."""
    perf_config = unified_config.get_performance_config()
    assert isinstance(perf_config, PerformanceConfig)
    assert perf_config.metrics["response_quality"] == 0.4
    assert perf_config.metrics["task_completion"] == 0.3

def test_environment_variables(unified_config):
    """Test environment variable handling."""
    assert unified_config.get_api_key() == "test_key"
    
    # Test environment-specific settings
    assert unified_config.is_development() is True
    assert unified_config.is_production() is False

def test_config_validation(config_dir, monkeypatch):
    """Test configuration validation."""
    # Test missing API key
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Missing required configuration keys"):
        UnifiedConfig(str(config_dir))

def test_agent_config_update(unified_config):
    """Test agent configuration updates."""
    updates = {
        "temperature": 0.5,
        "max_tokens": 1500
    }
    
    unified_config.update_agent_config("king", updates)
    king_config = unified_config.get_agent_config("king")
    assert king_config.frontier_model.temperature == 0.5
    assert king_config.frontier_model.max_tokens == 1500

def test_invalid_agent_type(unified_config):
    """Test handling of invalid agent types."""
    with pytest.raises(ValueError, match="Invalid agent type"):
        unified_config.get_agent_config("invalid_agent")

def test_config_persistence(unified_config, config_dir):
    """Test configuration persistence."""
    # Update configuration
    updates = {"temperature": 0.5}
    unified_config.update_agent_config("king", updates)
    
    # Create new instance and verify updates persisted
    new_config = UnifiedConfig(str(config_dir))
    king_config = new_config.get_agent_config("king")
    assert king_config.frontier_model.temperature == 0.5

def test_environment_override(config_dir, monkeypatch):
    """Test environment variable override of configuration."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
    monkeypatch.setenv("AIVILLAGE_ENVIRONMENT", "production")
    
    config = UnifiedConfig(str(config_dir))
    assert config.get_environment() == "production"
    assert config.is_production() is True

def test_default_values(unified_config):
    """Test default configuration values."""
    rag_config = unified_config.get_rag_config()
    assert rag_config.max_context_length == 2000  # Default value
    
    db_config = unified_config.get_db_config()
    assert db_config.vacuum_threshold == 1000  # Default value

def test_config_type_safety(unified_config):
    """Test type safety of configuration values."""
    king_config = unified_config.get_agent_config("king")
    
    # Verify all fields have correct types
    assert isinstance(king_config.frontier_model.temperature, float)
    assert isinstance(king_config.frontier_model.max_tokens, int)
    assert isinstance(king_config.description, str)
    assert isinstance(king_config.capabilities, list)

if __name__ == "__main__":
    pytest.main([__file__])
