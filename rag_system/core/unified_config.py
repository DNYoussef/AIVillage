"""Unified configuration for RAG system."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class UnifiedConfig:
    """Configuration for RAG system components."""
    
    # Core settings
    model_name: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-ada-002"
    
    # Retrieval settings
    retrieval_depth: int = 3
    relevance_threshold: float = 0.7
    feedback_enabled: bool = True
    max_context_length: int = 2000
    
    # Vector store settings
    vector_dimension: int = 768
    index_type: str = "flat"  # or "hnsw", "ivf", etc.
    
    # Graph store settings
    graph_type: str = "neo4j"
    max_neighbors: int = 5
    
    # Performance settings
    batch_size: int = 32
    max_concurrent_requests: int = 5
    request_timeout: int = 30
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval-specific configuration."""
        return {
            "depth": self.retrieval_depth,
            "threshold": self.relevance_threshold,
            "feedback_enabled": self.feedback_enabled,
            "max_context": self.max_context_length
        }
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration."""
        return {
            "dimension": self.vector_dimension,
            "index_type": self.index_type
        }
    
    def get_graph_store_config(self) -> Dict[str, Any]:
        """Get graph store configuration."""
        return {
            "type": self.graph_type,
            "max_neighbors": self.max_neighbors
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration."""
        return {
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent_requests,
            "timeout": self.request_timeout
        }

class ConfigManager:
    """Manager for RAG system configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or "config/rag_config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> UnifiedConfig:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            return UnifiedConfig(**config_data)
        except Exception as e:
            # Return default config if loading fails
            return UnifiedConfig()
    
    def save_config(self) -> None:
        """Save configuration to file."""
        config_dict = {
            field.name: getattr(self.config, field.name)
            for field in self.config.__dataclass_fields__.values()
        }
        
        # Ensure directory exists
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(config_dict, f)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save_config()

# Create default configuration instance
config_manager = ConfigManager()
unified_config = config_manager.config

__all__ = ['UnifiedConfig', 'ConfigManager', 'unified_config', 'config_manager']
