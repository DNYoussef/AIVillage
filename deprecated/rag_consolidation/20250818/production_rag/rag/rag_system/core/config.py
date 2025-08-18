from datetime import timedelta
from typing import Any

from pydantic import BaseModel, Field

from common.config import load_config


class UnifiedConfig(BaseModel):
    # Add common configuration parameters here
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    embedding_model: str = "bert-base-uncased"
    vector_store_type: str = "faiss"
    graph_store_type: str = "networkx"
    retriever_type: str = "hybrid"
    reasoning_engine_type: str = "uncertainty_aware"

    # SageAgent configuration
    agent_name: str = "SageAgent"
    agent_description: str = (
        "A research and analysis agent equipped with advanced reasoning "
        "and NLP capabilities."
    )

    # Retrieval configuration
    MAX_RESULTS: int = 10
    FEEDBACK_ITERATIONS: int = 3
    TEMPORAL_GRANULARITY: timedelta = timedelta(days=1)

    # Additional configuration parameters
    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Cache configuration
    cache_enabled: bool = True
    cache_max_size: int = 1000
    cache_ttl_hours: int = 24
    cache_similarity: float = 0.95

    # Extensible configuration dictionary for additional parameters
    extra_params: dict[str, Any] = Field(default_factory=dict)

    def update(self, **kwargs) -> None:
        # Update configuration parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extra_params[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        # Get configuration parameter
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra_params.get(key, default)


class RAGConfig(UnifiedConfig):
    # Add RAG-specific configuration parameters here
    num_documents: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# You can add more specific config classes as needed


_DEFAULTS = RAGConfig().model_dump()


def load_rag_config(config_path: str | None = None) -> RAGConfig:
    """Load RAG configuration using the shared loader."""
    data = load_config(_DEFAULTS, config_path, env_prefix="RAG_")
    return RAGConfig(**data)
