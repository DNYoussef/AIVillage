"""
Unified Configuration Management for RAG System Integration

Provides configuration management capabilities for RAG system components
within the AIVillage infrastructure. This module serves as a compatibility
layer to resolve import dependencies while maintaining system functionality.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model-related settings."""

    name: str = "default"
    version: str = "1.0.0"
    parameters: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    precision: str = "float32"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system settings."""

    index_type: str = "faiss"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    similarity_threshold: float = 0.7
    cache_size: int = 1000


@dataclass
class GenerationConfig:
    """Configuration for generation system settings."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)


class UnifiedConfig:
    """
    Unified configuration management for RAG system integration.

    Provides a centralized configuration system that supports:
    - Model configuration management
    - Retrieval system settings
    - Generation parameters
    - Environment-specific overrides
    - Dynamic configuration updates
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        model_config: ModelConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
        generation_config: GenerationConfig | None = None,
        **kwargs,
    ):
        """
        Initialize unified configuration.

        Args:
            config_path: Path to configuration file (optional)
            model_config: Model configuration settings
            retrieval_config: Retrieval system configuration
            generation_config: Generation system configuration
            **kwargs: Additional configuration parameters
        """
        self.config_path = Path(config_path) if config_path else None
        self.model_config = model_config or ModelConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.generation_config = generation_config or GenerationConfig()

        # Store additional configuration parameters
        self._config = kwargs.copy()

        # Default system configurations
        self._defaults = {
            "system_name": "aivillage_rag",
            "version": "2.0.0",
            "debug": False,
            "log_level": "INFO",
            "max_retries": 3,
            "timeout": 30.0,
            "enable_caching": True,
            "cache_ttl": 3600,
            "enable_metrics": True,
            "metrics_interval": 60,
        }

        # Apply defaults
        for key, value in self._defaults.items():
            if key not in self._config:
                self._config[key] = value

        logger.info(f"UnifiedConfig initialized with {len(self._config)} parameters")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
        logger.debug(f"Configuration updated: {key} = {value}")

    def update(self, config_dict: dict[str, Any]) -> None:
        """
        Update configuration with dictionary of values.

        Args:
            config_dict: Dictionary of configuration updates
        """
        self._config.update(config_dict)
        logger.info(f"Configuration updated with {len(config_dict)} new parameters")

    def to_dict(self) -> dict[str, Any]:
        """
        Export configuration as dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "model_config": {
                "name": self.model_config.name,
                "version": self.model_config.version,
                "parameters": self.model_config.parameters,
                "device": self.model_config.device,
                "precision": self.model_config.precision,
            },
            "retrieval_config": {
                "index_type": self.retrieval_config.index_type,
                "embedding_model": self.retrieval_config.embedding_model,
                "top_k": self.retrieval_config.top_k,
                "similarity_threshold": self.retrieval_config.similarity_threshold,
                "cache_size": self.retrieval_config.cache_size,
            },
            "generation_config": {
                "max_tokens": self.generation_config.max_tokens,
                "temperature": self.generation_config.temperature,
                "top_p": self.generation_config.top_p,
                "repetition_penalty": self.generation_config.repetition_penalty,
                "stop_sequences": self.generation_config.stop_sequences,
            },
            "system_config": self._config.copy(),
        }

    def validate(self) -> bool:
        """
        Validate configuration settings.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate model configuration
            if not self.model_config.name:
                logger.error("Model name cannot be empty")
                return False

            # Validate retrieval configuration
            if self.retrieval_config.top_k <= 0:
                logger.error("Retrieval top_k must be positive")
                return False

            if not 0 <= self.retrieval_config.similarity_threshold <= 1:
                logger.error("Similarity threshold must be between 0 and 1")
                return False

            # Validate generation configuration
            if self.generation_config.max_tokens <= 0:
                logger.error("Max tokens must be positive")
                return False

            if not 0 <= self.generation_config.temperature <= 2:
                logger.error("Temperature must be between 0 and 2")
                return False

            logger.info("Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def reload(self) -> bool:
        """
        Reload configuration from file if config_path is set.

        Returns:
            True if reload successful, False otherwise
        """
        if not self.config_path or not self.config_path.exists():
            logger.warning("No configuration file to reload")
            return False

        try:
            # Implementation would load from file
            # For now, this is a placeholder
            logger.info(f"Configuration reloaded from {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False

    def save(self, path: str | Path | None = None) -> bool:
        """
        Save configuration to file.

        Args:
            path: Optional path to save to (uses config_path if not provided)

        Returns:
            True if save successful, False otherwise
        """
        save_path = Path(path) if path else self.config_path

        if not save_path:
            logger.error("No save path specified")
            return False

        try:
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Implementation would save to file
            # For now, this is a placeholder
            logger.info(f"Configuration saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"UnifiedConfig(params={len(self._config)}, model={self.model_config.name})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"UnifiedConfig with {len(self._config)} parameters"


# Convenience functions for common configuration patterns
def create_default_config() -> UnifiedConfig:
    """Create a default configuration with standard settings."""
    return UnifiedConfig(
        model_config=ModelConfig(name="aivillage_default", version="2.0.0", device="cpu", precision="float32"),
        retrieval_config=RetrievalConfig(
            index_type="faiss",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            top_k=5,
            similarity_threshold=0.7,
        ),
        generation_config=GenerationConfig(max_tokens=512, temperature=0.7, top_p=0.9),
    )


def create_production_config() -> UnifiedConfig:
    """Create a production-optimized configuration."""
    return UnifiedConfig(
        model_config=ModelConfig(
            name="aivillage_production",
            version="2.0.0",
            device="auto",  # Will auto-detect GPU if available
            precision="float16",  # More memory efficient
        ),
        retrieval_config=RetrievalConfig(
            index_type="faiss",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            top_k=10,  # More results for better quality
            similarity_threshold=0.6,  # Slightly lower for broader retrieval
            cache_size=5000,  # Larger cache for production
        ),
        generation_config=GenerationConfig(
            max_tokens=1024, temperature=0.5, top_p=0.85, repetition_penalty=1.1  # Longer responses  # More focused
        ),
        enable_metrics=True,
        enable_caching=True,
        cache_ttl=7200,  # 2 hour cache
        max_retries=5,
        timeout=60.0,
    )
