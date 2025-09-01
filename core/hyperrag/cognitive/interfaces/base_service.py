"""
Base Service Interface for GraphFixer Service Decomposition

Defines the common interface and patterns for all GraphFixer services,
enabling clean architecture principles and dependency injection.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
from dataclasses import dataclass


@dataclass
class ServiceConfig:
    """Configuration for all services."""

    trust_graph: Any = None
    vector_engine: Any = None
    min_confidence_threshold: float = 0.3
    max_proposals_per_gap: int = 3
    cache_enabled: bool = True
    logging_level: str = "INFO"


class BaseService(ABC):
    """
    Base service interface for all GraphFixer services.

    Provides common functionality for logging, configuration,
    and dependency management.
    """

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, config.logging_level))
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the service with required dependencies."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources and connections."""
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if service is properly initialized."""
        return self._initialized

    def validate_dependencies(self, required_deps: List[str]) -> bool:
        """Validate that required dependencies are available."""
        for dep in required_deps:
            if not hasattr(self.config, dep) or getattr(self.config, dep) is None:
                self.logger.error(f"Missing required dependency: {dep}")
                return False
        return True


class AsyncServiceMixin:
    """Mixin for services that need async context management."""

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


class CacheableMixin:
    """Mixin for services that support caching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, Any] = {}

    def get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(key) if self.config.cache_enabled else None

    def set_cache(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if self.config.cache_enabled:
            self._cache[key] = value

    def clear_cache(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
