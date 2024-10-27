"""Core RAG system components."""

from .unified_config import UnifiedConfig, ConfigManager

# Create default configuration manager instance
config_manager = ConfigManager("config/rag_config.yaml")
unified_config = config_manager.get_config()

__all__ = ['UnifiedConfig', 'ConfigManager', 'unified_config', 'config_manager']
