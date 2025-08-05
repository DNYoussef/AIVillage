from datetime import timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
import yaml


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
    agent_description: str = "A research and analysis agent equipped with advanced reasoning and NLP capabilities."

    # Retrieval configuration
    MAX_RESULTS: int = 10
    FEEDBACK_ITERATIONS: int = 3
    TEMPORAL_GRANULARITY: timedelta = timedelta(days=1)

    # Additional configuration parameters
    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200

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


def load_from_yaml(config_path: str) -> RAGConfig:
    """Load configuration from a YAML file.

    Parameters
    ----------
    config_path: str
        Path to the YAML configuration file.

    Returns:
    -------
    RAGConfig
        Configuration populated with values from the YAML file.
    """
    path = Path(config_path)
    data = {}
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    return RAGConfig(**data)
