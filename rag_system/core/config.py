"""Configuration classes for RAG system."""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import timedelta

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
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    def update(self, **kwargs):
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
    vector_dimension: int = 768  # Added vector dimension
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

class OpenAIGPTConfig(BaseModel):
    """Configuration for OpenAI GPT models."""
    api_key: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: Optional[str] = None
    chat_model: Optional[str] = "gpt-4"
    
    def create(self):
        """Create a model instance with this configuration."""
        # This would typically create an OpenAI model instance
        # For now, return self as we're just using it for configuration
        return self

# You can add more specific config classes as needed
