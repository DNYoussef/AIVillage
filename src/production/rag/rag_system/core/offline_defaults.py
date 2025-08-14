"""RAG Offline Defaults Configuration - Prompt B

Comprehensive offline-first configuration for RAG system enabling full functionality
without internet access, external APIs, or cloud dependencies.

Integration Point: Base configuration for Phase 4 integration testing
"""

import logging
import os
from pathlib import Path
from typing import Any

from .config import RAGConfig

logger = logging.getLogger(__name__)


class OfflineRAGConfig(RAGConfig):
    """Offline-first RAG configuration with all defaults pointing to local resources."""

    # Offline-first embedding models (CPU-compatible, downloadable)
    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"  # Small, fast, offline
    )
    fallback_embedding_model: str = "distilbert-base-uncased"  # Backup option

    # Local vector storage (no cloud dependencies)
    vector_store_type: str = "faiss_local"  # FAISS with local persistence
    vector_store_path: str = "./data/vector_store/"

    # Local graph storage
    graph_store_type: str = "networkx_local"  # NetworkX with local persistence
    graph_store_path: str = "./data/graph_store/"

    # Local cache storage
    cache_enabled: bool = True
    cache_type: str = "local_disk"  # Local disk cache, no Redis/external cache
    cache_path: str = "./data/cache/"
    cache_max_size_mb: int = 512  # 512MB cache limit
    cache_ttl_hours: int = 72  # 3 days TTL for offline scenarios

    # Offline-compatible models (no API calls)
    model_name: str = "local_llm"  # Placeholder for local LLM
    temperature: float = 0.3  # Conservative for consistent offline results
    max_tokens: int = 512  # Conservative for mobile/resource-constrained

    # Offline retrieval settings
    retriever_type: str = "hybrid_offline"  # Hybrid without external APIs
    max_results: int = 5  # Conservative for offline scenarios
    chunk_size: int = 256  # Smaller chunks for mobile devices
    chunk_overlap: int = 32  # Minimal overlap for efficiency

    # Offline-first processing
    enable_internet_features: bool = False  # Disable all internet-dependent features
    enable_api_calls: bool = False  # No external API calls
    enable_cloud_sync: bool = False  # No cloud synchronization

    # Battery/resource-aware settings
    enable_mobile_optimizations: bool = True
    low_memory_mode: bool = True  # Optimize for low memory devices
    batch_processing: bool = False  # Process individually to save memory

    # Local data directories
    local_data_dir: str = "./data/"
    local_models_dir: str = "./models/"
    local_logs_dir: str = "./logs/"

    # Connection timeouts (for when internet is available but slow)
    connection_timeout_seconds: int = 5  # Quick timeout for offline-first
    read_timeout_seconds: int = 10

    # Fallback behavior configuration
    enable_graceful_degradation: bool = (
        True  # Degrade gracefully when resources unavailable
    )
    strict_offline_mode: bool = (
        False  # Allow opportunistic online features if available
    )


class OfflineDefaultsManager:
    """Manager for offline-first RAG configuration and resource validation."""

    def __init__(self, base_data_dir: str | None = None):
        """Initialize offline defaults manager.

        Args:
            base_data_dir: Base directory for all local data (defaults to ./data/)
        """
        self.base_data_dir = Path(base_data_dir or "./data/")
        self.config = OfflineRAGConfig()

        # Ensure all local directories exist
        self._ensure_local_directories()

    def _ensure_local_directories(self) -> None:
        """Create all required local directories."""
        directories = [
            self.base_data_dir,
            self.base_data_dir / "vector_store",
            self.base_data_dir / "graph_store",
            self.base_data_dir / "cache",
            self.base_data_dir / "models",
            self.base_data_dir / "logs",
            self.base_data_dir / "documents",
            self.base_data_dir / "embeddings",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    def get_offline_config(self, **overrides) -> OfflineRAGConfig:
        """Get offline-first configuration with optional overrides.

        Args:
            **overrides: Configuration overrides

        Returns:
            OfflineRAGConfig configured for offline operation
        """
        config = OfflineRAGConfig()

        # Apply data directory paths
        config.vector_store_path = str(self.base_data_dir / "vector_store")
        config.graph_store_path = str(self.base_data_dir / "graph_store")
        config.cache_path = str(self.base_data_dir / "cache")
        config.local_data_dir = str(self.base_data_dir)
        config.local_models_dir = str(self.base_data_dir / "models")
        config.local_logs_dir = str(self.base_data_dir / "logs")

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                config.extra_params[key] = value

        return config

    def validate_offline_readiness(self) -> dict[str, Any]:
        """Validate system readiness for offline operation.

        Returns:
            Dict with validation results and recommendations
        """
        validation_results = {
            "ready_for_offline": True,
            "missing_components": [],
            "warnings": [],
            "recommendations": [],
            "storage_info": {},
        }

        # Check local directories
        required_dirs = [
            "vector_store",
            "graph_store",
            "cache",
            "models",
            "logs",
            "documents",
        ]

        for dir_name in required_dirs:
            dir_path = self.base_data_dir / dir_name
            if not dir_path.exists():
                validation_results["missing_components"].append(
                    f"Directory: {dir_path}"
                )
                validation_results["ready_for_offline"] = False
            else:
                # Check disk space
                try:
                    stat = dir_path.stat()
                    validation_results["storage_info"][dir_name] = {
                        "path": str(dir_path),
                        "exists": True,
                        "size_bytes": stat.st_size,
                    }
                except Exception as e:
                    validation_results["warnings"].append(
                        f"Could not stat {dir_path}: {e}"
                    )

        # Check for offline-compatible models
        models_dir = self.base_data_dir / "models"
        if not any(models_dir.glob("*")):
            validation_results["warnings"].append("No local models found")
            validation_results["recommendations"].append(
                "Download offline-compatible models: sentence-transformers/all-MiniLM-L6-v2"
            )

        # Check available disk space
        try:
            import shutil

            total, used, free = shutil.disk_usage(self.base_data_dir)
            free_gb = free / (1024**3)

            validation_results["storage_info"]["disk_space"] = {
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free_gb,
            }

            if free_gb < 1.0:  # Less than 1GB free
                validation_results["warnings"].append(
                    f"Low disk space: {free_gb:.1f}GB available"
                )
                validation_results["recommendations"].append(
                    "Free up disk space for optimal operation"
                )

        except Exception as e:
            validation_results["warnings"].append(f"Could not check disk space: {e}")

        # Check Python environment for offline dependencies
        offline_dependencies = [
            "sentence_transformers",
            "faiss",
            "networkx",
            "sqlite3",
            "transformers",
        ]

        missing_deps = []
        for dep in offline_dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            validation_results["missing_components"].extend(missing_deps)
            validation_results["ready_for_offline"] = False
            validation_results["recommendations"].append(
                f"Install missing offline dependencies: {', '.join(missing_deps)}"
            )

        # Overall readiness assessment
        if validation_results["missing_components"]:
            validation_results["ready_for_offline"] = False

        return validation_results

    def setup_offline_environment(self) -> dict[str, Any]:
        """Set up complete offline environment for RAG system.

        Returns:
            Dict with setup results and status
        """
        setup_results = {
            "success": True,
            "components_initialized": [],
            "errors": [],
            "warnings": [],
        }

        try:
            # Ensure directories
            self._ensure_local_directories()
            setup_results["components_initialized"].append("local_directories")

            # Initialize empty vector store
            vector_store_path = self.base_data_dir / "vector_store"
            if not (vector_store_path / "index.faiss").exists():
                try:
                    # Create minimal FAISS index
                    import faiss
                    import numpy as np

                    # Create empty 384-dimension index (all-MiniLM-L6-v2 size)
                    index = faiss.IndexFlatL2(384)
                    faiss.write_index(index, str(vector_store_path / "index.faiss"))

                    # Create metadata file
                    metadata = {
                        "dimension": 384,
                        "count": 0,
                        "created": "offline_setup",
                    }
                    import json

                    with open(vector_store_path / "metadata.json", "w") as f:
                        json.dump(metadata, f)

                    setup_results["components_initialized"].append("vector_store")

                except ImportError:
                    setup_results["warnings"].append(
                        "FAISS not available, vector store setup skipped"
                    )
                except Exception as e:
                    setup_results["errors"].append(f"Vector store setup failed: {e}")

            # Initialize graph store
            graph_store_path = self.base_data_dir / "graph_store"
            if not (graph_store_path / "graph.json").exists():
                try:
                    import json

                    import networkx as nx

                    # Create empty graph
                    G = nx.Graph()
                    data = nx.node_link_data(G)

                    with open(graph_store_path / "graph.json", "w") as f:
                        json.dump(data, f)

                    setup_results["components_initialized"].append("graph_store")

                except ImportError:
                    setup_results["warnings"].append(
                        "NetworkX not available, graph store setup skipped"
                    )
                except Exception as e:
                    setup_results["errors"].append(f"Graph store setup failed: {e}")

            # Create cache directories
            cache_path = self.base_data_dir / "cache"
            for cache_level in ["l1", "l2", "l3"]:
                (cache_path / cache_level).mkdir(exist_ok=True)
            setup_results["components_initialized"].append("cache_structure")

            # Create configuration file
            config_file = self.base_data_dir / "offline_config.json"
            config_data = self.get_offline_config().dict()

            import json

            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2, default=str)
            setup_results["components_initialized"].append("configuration_file")

        except Exception as e:
            setup_results["success"] = False
            setup_results["errors"].append(f"Setup failed: {e}")

        return setup_results


def get_offline_rag_config(**overrides) -> OfflineRAGConfig:
    """Get offline-first RAG configuration with optional overrides.

    This is the main entry point for getting offline-configured RAG settings.

    Args:
        **overrides: Configuration overrides

    Returns:
        OfflineRAGConfig ready for offline operation
    """
    manager = OfflineDefaultsManager()
    return manager.get_offline_config(**overrides)


def validate_offline_readiness(data_dir: str | None = None) -> dict[str, Any]:
    """Validate system readiness for offline RAG operation.

    Args:
        data_dir: Base data directory to validate

    Returns:
        Dict with validation results
    """
    manager = OfflineDefaultsManager(data_dir)
    return manager.validate_offline_readiness()


def setup_offline_rag_environment(data_dir: str | None = None) -> dict[str, Any]:
    """Set up complete offline RAG environment.

    Args:
        data_dir: Base data directory for setup

    Returns:
        Dict with setup results
    """
    manager = OfflineDefaultsManager(data_dir)
    return manager.setup_offline_environment()


# Environment detection utilities
def is_offline_environment() -> bool:
    """Detect if running in offline environment."""
    # Check environment variables
    if os.getenv("RAG_OFFLINE_MODE") == "1":
        return True
    if os.getenv("OFFLINE") == "1":
        return True

    # Check for network connectivity (basic test)
    try:
        import socket

        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return False  # Network available
    except OSError:
        return True  # Network unavailable, assume offline


def auto_configure_for_environment() -> OfflineRAGConfig:
    """Automatically configure RAG for current environment (online/offline).

    Returns:
        OfflineRAGConfig optimized for detected environment
    """
    if is_offline_environment():
        logger.info("Offline environment detected, using offline-first configuration")
        config = get_offline_rag_config()
        config.strict_offline_mode = True
    else:
        logger.info("Online environment detected, using offline-friendly configuration")
        config = get_offline_rag_config()
        config.strict_offline_mode = False
        config.enable_internet_features = True  # Allow opportunistic online features

    return config


# Default configuration instance
DEFAULT_OFFLINE_CONFIG = get_offline_rag_config()
