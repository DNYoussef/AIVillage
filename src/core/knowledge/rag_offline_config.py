"""
RAG Defaults Offline Validated Config - Prompt 6

Offline-validated RAG system configuration with verified defaults for reliable
knowledge retrieval and reasoning without dependency on external services.

Key Features:
- Pre-validated configuration templates for common RAG scenarios
- Offline embedding model specifications and fallbacks
- Resource-aware chunk sizing and retrieval parameters
- Mobile/constrained-device optimized configurations
- Validation framework for config integrity

Knowledge & Data Integration Point: Validated RAG system defaults
"""

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def resolve_model_path(
    model_name: str,
    device: str | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    """Resolve model paths using configuration or environment variables.

    Priority order:
    1. Explicit ``base_dir`` supplied by configuration.
    2. Environment variable ``RAG_<DEVICE>_MODEL_PATH`` pointing to a file or
       directory for the specific device.
    3. Environment variable ``RAG_MODEL_DIR`` as a base directory.
    4. Default to ``models/<device>/<model_name>`` relative to the working
       directory.
    """

    if base_dir:
        base = Path(base_dir)
        if base.is_dir():
            return base / model_name
        return base

    if device:
        env_specific = os.getenv(f"RAG_{device.upper()}_MODEL_PATH")
        if env_specific:
            specific_path = Path(env_specific)
            if specific_path.is_dir():
                return specific_path / model_name
            return specific_path

    env_base = os.getenv("RAG_MODEL_DIR")
    if env_base:
        base = Path(env_base)
        if device:
            base = base / device
        return base / model_name

    base = Path("models")
    if device:
        base = base / device
    return base / model_name


class RAGMode(Enum):
    """RAG operational modes for different scenarios."""

    OFFLINE_ONLY = "offline_only"  # No external API calls
    HYBRID_OFFLINE_FIRST = "hybrid_offline_first"  # Prefer offline, fallback to online
    BALANCED = "balanced"  # Balance offline/online based on quality
    PERFORMANCE_FIRST = "performance_first"  # Best quality regardless of mode
    MOBILE_OPTIMIZED = "mobile_optimized"  # Optimized for mobile constraints


class EmbeddingProvider(Enum):
    """Available embedding model providers."""

    SENTENCE_TRANSFORMERS = "sentence_transformers"  # Offline-capable
    TORCH_HUB = "torch_hub"  # Offline-capable
    SPACY = "spacy"  # Offline-capable
    OPENAI = "openai"  # Online only
    COHERE = "cohere"  # Online only
    LOCAL_ONNX = "local_onnx"  # Optimized offline


class VectorStoreType(Enum):
    """Vector storage backends."""

    FAISS = "faiss"  # Fast, memory-based
    CHROMA = "chroma"  # Persistent, versatile
    MEMORY = "memory"  # Simple in-memory
    SQLITE_VSS = "sqlite_vss"  # SQLite with vector search
    HNSWLIB = "hnswlib"  # Hierarchical NSW graphs


class RetrieverType(Enum):
    """Retrieval strategies."""

    SEMANTIC = "semantic"  # Pure vector similarity
    KEYWORD = "keyword"  # BM25/TF-IDF based
    HYBRID = "hybrid"  # Semantic + keyword fusion
    GRAPH_ENHANCED = "graph_enhanced"  # Knowledge graph integration
    CONTEXTUAL = "contextual"  # Context-aware retrieval


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""

    provider: EmbeddingProvider
    model_name: str
    model_path: str | None = None  # Local path for offline models
    dimensions: int = 384
    max_seq_length: int = 512
    batch_size: int = 32
    device: str = "cpu"  # cpu, cuda, mps

    # Validation metadata
    checksum: str | None = None
    validated_date: str | None = None
    performance_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    separator: str = "\n\n"
    keep_separator: bool = True

    # Advanced chunking options
    respect_sentence_boundaries: bool = True
    min_chunk_size: int = 50
    max_chunk_size: int = 2048

    # Mobile optimization
    mobile_chunk_size: int = 256
    mobile_overlap: int = 25

    def get_effective_chunk_size(self, is_mobile: bool = False) -> int:
        """Get effective chunk size based on device constraints."""
        return self.mobile_chunk_size if is_mobile else self.chunk_size

    def get_effective_overlap(self, is_mobile: bool = False) -> int:
        """Get effective overlap based on device constraints."""
        return self.mobile_overlap if is_mobile else self.chunk_overlap


@dataclass
class RetrievalConfig:
    """Configuration for retrieval parameters."""

    top_k: int = 5
    similarity_threshold: float = 0.7
    max_results: int = 20

    # Reranking configuration
    enable_reranking: bool = True
    rerank_top_k: int = 10
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Filtering and diversity
    enable_diversity: bool = True
    diversity_threshold: float = 0.8
    temporal_decay_factor: float = 0.1

    # Mobile optimization
    mobile_top_k: int = 3
    mobile_max_results: int = 10

    def get_effective_top_k(self, is_mobile: bool = False) -> int:
        """Get effective top_k based on device constraints."""
        return self.mobile_top_k if is_mobile else self.top_k

    def get_effective_max_results(self, is_mobile: bool = False) -> int:
        """Get effective max_results based on device constraints."""
        return self.mobile_max_results if is_mobile else self.max_results


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Offline generation fallbacks
    offline_model_path: str | None = None
    offline_model_type: str = "transformers"  # transformers, onnx, ggml

    # Context management
    max_context_length: int = 4096
    context_overlap: int = 512
    enable_context_compression: bool = True


@dataclass
class CacheConfig:
    """Configuration for caching systems."""

    enabled: bool = True
    cache_type: str = "memory"  # memory, redis, sqlite
    max_size_mb: int = 100
    ttl_seconds: int = 3600

    # Cache keys and similarity
    similarity_threshold: float = 0.95
    key_prefix: str = "rag_cache"
    compression_enabled: bool = True

    # Offline cache persistence
    persistent_cache_path: str | None = None
    enable_cache_warmup: bool = True


@dataclass
class OfflineRAGConfig:
    """Comprehensive offline-validated RAG configuration."""

    # Core configuration
    mode: RAGMode = RAGMode.OFFLINE_ONLY
    name: str = "default_offline_rag"
    description: str = "Default offline RAG configuration"
    version: str = "1.0.0"

    # Component configurations
    embedding: EmbeddingConfig = field(
        default_factory=lambda: EmbeddingConfig(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            dimensions=384,
        )
    )
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Vector store configuration
    vector_store_type: VectorStoreType = VectorStoreType.FAISS
    vector_store_config: dict[str, Any] = field(default_factory=dict)

    # Retriever configuration
    retriever_type: RetrieverType = RetrieverType.HYBRID
    retriever_config: dict[str, Any] = field(default_factory=dict)

    # Resource constraints
    max_memory_mb: int = 512
    max_cpu_threads: int = 2
    enable_gpu: bool = False

    # Mobile/device-specific settings
    is_mobile_optimized: bool = False
    device_type: str = "laptop"  # mobile, tablet, laptop, desktop, server

    # Validation metadata
    validated: bool = False
    validation_checksum: str | None = None
    validation_date: str | None = None
    validation_errors: list[str] = field(default_factory=list)

    def adapt_for_mobile(self) -> "OfflineRAGConfig":
        """Create mobile-optimized variant of this config."""
        base_dir = (
            Path(self.embedding.model_path).parent
            if self.embedding.model_path
            else None
        )

        mobile_config = OfflineRAGConfig(
            mode=RAGMode.MOBILE_OPTIMIZED,
            name=f"{self.name}_mobile",
            description=f"{self.description} (mobile optimized)",
            version=self.version,
            # Use mobile-optimized settings
            embedding=EmbeddingConfig(
                provider=EmbeddingProvider.LOCAL_ONNX,
                model_name="all-MiniLM-L6-v2-onnx",
                model_path=str(
                    resolve_model_path(
                        "all-MiniLM-L6-v2.onnx", device="mobile", base_dir=base_dir
                    )
                ),
                dimensions=384,
                batch_size=16,  # Smaller batches
                device="cpu",
            ),
            chunking=ChunkingConfig(
                chunk_size=256,
                chunk_overlap=25,
                mobile_chunk_size=128,
                mobile_overlap=12,  # Smaller chunks
            ),
            retrieval=RetrievalConfig(
                top_k=3,  # Fewer results
                max_results=10,
                enable_reranking=False,  # Disable expensive reranking
                mobile_top_k=2,
                mobile_max_results=5,
            ),
            generation=GenerationConfig(
                max_tokens=256,  # Shorter responses
                max_context_length=2048,  # Smaller context window
                enable_context_compression=True,
            ),
            cache=CacheConfig(
                max_size_mb=50,
                ttl_seconds=1800,
                compression_enabled=True,  # Smaller cache  # Shorter TTL
            ),
            # Mobile-specific settings
            vector_store_type=VectorStoreType.SQLITE_VSS,  # More efficient for mobile
            retriever_type=RetrieverType.SEMANTIC,  # Simpler retrieval
            max_memory_mb=256,
            max_cpu_threads=1,
            enable_gpu=False,
            is_mobile_optimized=True,
            device_type="mobile",
        )

        return mobile_config

    def validate_config(self) -> bool:
        """Validate configuration for consistency and feasibility."""
        errors = []

        # Validate embedding configuration
        if self.embedding.dimensions <= 0:
            errors.append("Embedding dimensions must be positive")

        if self.embedding.provider == EmbeddingProvider.LOCAL_ONNX:
            if not self.embedding.model_path:
                errors.append("Local ONNX provider requires model_path")
            else:
                model_path = Path(self.embedding.model_path)
                if not model_path.exists():
                    errors.append(
                        f"Local ONNX model path not found: {self.embedding.model_path}"
                    )

        # Validate chunking configuration
        if self.chunking.chunk_size <= self.chunking.chunk_overlap:
            errors.append("Chunk size must be larger than overlap")

        if self.chunking.min_chunk_size >= self.chunking.max_chunk_size:
            errors.append("Min chunk size must be less than max chunk size")

        # Validate retrieval configuration
        if self.retrieval.top_k > self.retrieval.max_results:
            errors.append("top_k cannot exceed max_results")

        if not 0.0 <= self.retrieval.similarity_threshold <= 1.0:
            errors.append("Similarity threshold must be between 0.0 and 1.0")

        # Validate resource constraints
        if self.max_memory_mb <= 0:
            errors.append("Max memory must be positive")

        if self.max_cpu_threads <= 0:
            errors.append("Max CPU threads must be positive")

        # Validate mobile optimization consistency
        if self.is_mobile_optimized:
            if self.max_memory_mb > 512:
                errors.append("Mobile config should have memory limit ≤ 512MB")

            if self.chunking.chunk_size > 512:
                errors.append("Mobile config should have chunk size ≤ 512")

        self.validation_errors = errors
        self.validated = len(errors) == 0

        if self.validated:
            # Calculate validation checksum
            config_str = json.dumps(asdict(self), sort_keys=True, default=str)
            self.validation_checksum = hashlib.sha256(config_str.encode()).hexdigest()[
                :16
            ]
            logger.info(f"Config validation passed: {self.name}")
        else:
            logger.warning(f"Config validation failed: {self.name}, errors: {errors}")

        return self.validated


class OfflineRAGConfigRegistry:
    """Registry for pre-validated offline RAG configurations."""

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path(__file__).parent / "configs" / "rag"
        self.configs: dict[str, OfflineRAGConfig] = {}
        self.validated_configs: dict[str, OfflineRAGConfig] = {}

        # Initialize default configurations
        self._initialize_default_configs()
        self._validate_all_configs()

        logger.info(
            f"OfflineRAGConfigRegistry initialized with {len(self.validated_configs)} validated configs"
        )

    def _initialize_default_configs(self):
        """Initialize default RAG configurations for common scenarios."""

        # 1. Standard Offline RAG
        standard_config = OfflineRAGConfig(
            name="standard_offline",
            description="Standard offline RAG configuration with balanced performance",
            mode=RAGMode.OFFLINE_ONLY,
            embedding=EmbeddingConfig(
                provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
                model_name="all-MiniLM-L6-v2",
                dimensions=384,
                batch_size=32,
            ),
            retrieval=RetrievalConfig(
                top_k=5, similarity_threshold=0.7, enable_reranking=True
            ),
            vector_store_type=VectorStoreType.FAISS,
            retriever_type=RetrieverType.HYBRID,
        )
        self.configs["standard_offline"] = standard_config

        # 2. High Performance Offline RAG
        performance_config = OfflineRAGConfig(
            name="performance_offline",
            description="High-performance offline RAG with larger models",
            mode=RAGMode.PERFORMANCE_FIRST,
            embedding=EmbeddingConfig(
                provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
                model_name="all-mpnet-base-v2",
                dimensions=768,
                batch_size=16,
            ),
            chunking=ChunkingConfig(chunk_size=1024, chunk_overlap=100),
            retrieval=RetrievalConfig(
                top_k=10,
                similarity_threshold=0.6,
                enable_reranking=True,
                rerank_top_k=20,
            ),
            vector_store_type=VectorStoreType.CHROMA,
            retriever_type=RetrieverType.GRAPH_ENHANCED,
            max_memory_mb=2048,
            max_cpu_threads=4,
        )
        self.configs["performance_offline"] = performance_config

        # 3. Memory-Efficient RAG
        memory_efficient_config = OfflineRAGConfig(
            name="memory_efficient",
            description="Memory-efficient RAG for resource-constrained environments",
            mode=RAGMode.OFFLINE_ONLY,
            embedding=EmbeddingConfig(
                provider=EmbeddingProvider.LOCAL_ONNX,
                model_name="all-MiniLM-L6-v2-onnx",
                model_path=str(resolve_model_path("all-MiniLM-L6-v2.onnx")),
                dimensions=384,
                batch_size=8,
            ),
            chunking=ChunkingConfig(chunk_size=256, chunk_overlap=25),
            retrieval=RetrievalConfig(top_k=3, max_results=10, enable_reranking=False),
            cache=CacheConfig(max_size_mb=25, compression_enabled=True),
            vector_store_type=VectorStoreType.MEMORY,
            retriever_type=RetrieverType.SEMANTIC,
            max_memory_mb=128,
            max_cpu_threads=1,
        )
        self.configs["memory_efficient"] = memory_efficient_config

        # 4. Mobile-Optimized RAG
        mobile_config = standard_config.adapt_for_mobile()
        self.configs["mobile_optimized"] = mobile_config

        # 5. Hybrid Online/Offline RAG
        hybrid_config = OfflineRAGConfig(
            name="hybrid_offline_first",
            description="Hybrid RAG that prefers offline but can fallback to online",
            mode=RAGMode.HYBRID_OFFLINE_FIRST,
            embedding=EmbeddingConfig(
                provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
                model_name="all-MiniLM-L6-v2",
                dimensions=384,
            ),
            retrieval=RetrievalConfig(
                top_k=7, similarity_threshold=0.75, enable_reranking=True
            ),
            vector_store_type=VectorStoreType.CHROMA,
            retriever_type=RetrieverType.CONTEXTUAL,
        )
        self.configs["hybrid_offline_first"] = hybrid_config

    def _validate_all_configs(self):
        """Validate all registered configurations."""
        for name, config in self.configs.items():
            if config.validate_config():
                self.validated_configs[name] = config
                logger.info(f"Config '{name}' validated successfully")
            else:
                logger.warning(
                    f"Config '{name}' failed validation: {config.validation_errors}"
                )

    def get_config(
        self, name: str, validated_only: bool = True
    ) -> OfflineRAGConfig | None:
        """Get configuration by name."""
        if validated_only:
            return self.validated_configs.get(name)
        else:
            return self.configs.get(name)

    def list_configs(self, validated_only: bool = True) -> list[str]:
        """List available configuration names."""
        if validated_only:
            return list(self.validated_configs.keys())
        else:
            return list(self.configs.keys())

    def get_config_for_device(
        self, device_type: str, memory_mb: int
    ) -> OfflineRAGConfig:
        """Get optimal configuration for device constraints."""
        if device_type in ["mobile", "tablet"] or memory_mb < 512:
            return self.validated_configs["mobile_optimized"]
        elif memory_mb < 1024:
            return self.validated_configs["memory_efficient"]
        elif memory_mb >= 2048:
            return self.validated_configs["performance_offline"]
        else:
            return self.validated_configs["standard_offline"]

    def export_config(self, name: str, output_path: Path, format: str = "yaml") -> bool:
        """Export configuration to file."""
        config = self.get_config(name)
        if not config:
            logger.error(f"Config '{name}' not found")
            return False

        try:
            config_dict = asdict(config)

            if format.lower() == "yaml":
                with open(output_path, "w") as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                with open(output_path, "w") as f:
                    json.dump(config_dict, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Exported config '{name}' to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export config '{name}': {e}")
            return False

    def import_config(self, file_path: Path, validate: bool = True) -> str | None:
        """Import configuration from file."""
        try:
            with open(file_path) as f:
                if (
                    file_path.suffix.lower() == ".yaml"
                    or file_path.suffix.lower() == ".yml"
                ):
                    config_dict = yaml.safe_load(f)
                elif file_path.suffix.lower() == ".json":
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")

            config = OfflineRAGConfig(**config_dict)

            if validate and not config.validate_config():
                logger.error(
                    f"Imported config failed validation: {config.validation_errors}"
                )
                return None

            # Use filename as config name if not specified
            config_name = config.name or file_path.stem

            self.configs[config_name] = config
            if config.validated:
                self.validated_configs[config_name] = config

            logger.info(f"Imported config '{config_name}' from {file_path}")
            return config_name

        except Exception as e:
            logger.error(f"Failed to import config from {file_path}: {e}")
            return None

    def get_registry_status(self) -> dict[str, Any]:
        """Get registry status information."""
        return {
            "total_configs": len(self.configs),
            "validated_configs": len(self.validated_configs),
            "available_configs": list(self.validated_configs.keys()),
            "config_modes": list(
                {config.mode.value for config in self.validated_configs.values()}
            ),
            "embedding_providers": list(
                {
                    config.embedding.provider.value
                    for config in self.validated_configs.values()
                }
            ),
            "vector_store_types": list(
                {
                    config.vector_store_type.value
                    for config in self.validated_configs.values()
                }
            ),
            "retriever_types": list(
                {
                    config.retriever_type.value
                    for config in self.validated_configs.values()
                }
            ),
        }


# Global registry instance
_global_rag_registry = None


def get_rag_config_registry() -> OfflineRAGConfigRegistry:
    """Get global RAG config registry instance."""
    global _global_rag_registry
    if _global_rag_registry is None:
        _global_rag_registry = OfflineRAGConfigRegistry()
    return _global_rag_registry


# Convenience functions
def get_offline_rag_config(name: str = "standard_offline") -> OfflineRAGConfig | None:
    """Get validated offline RAG configuration by name."""
    registry = get_rag_config_registry()
    return registry.get_config(name)


def get_mobile_rag_config() -> OfflineRAGConfig:
    """Get mobile-optimized RAG configuration."""
    registry = get_rag_config_registry()
    return registry.get_config("mobile_optimized")


def create_custom_rag_config(
    name: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 512,
    top_k: int = 5,
    **kwargs,
) -> OfflineRAGConfig:
    """Create custom RAG configuration with validated defaults."""
    config = OfflineRAGConfig(
        name=name,
        description=f"Custom RAG config: {name}",
        embedding=EmbeddingConfig(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMERS, model_name=embedding_model
        ),
        chunking=ChunkingConfig(chunk_size=chunk_size),
        retrieval=RetrievalConfig(top_k=top_k),
        **kwargs,
    )

    # Validate the custom config
    config.validate_config()

    return config
