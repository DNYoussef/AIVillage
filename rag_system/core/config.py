# rag_system/core/config.py

from datetime import timedelta
from pydantic import BaseSettings, Field

class RAGConfig(BaseSettings):
    VECTOR_SIZE: int = Field(768, description="Size of vector embeddings")
    QDRANT_URL: str = Field("http://localhost:6333", description="URL for Qdrant vector database")
    QDRANT_API_KEY: str = Field("your_api_key", description="API key for Qdrant")
    VECTOR_COLLECTION_NAME: str = Field("your_collection_name", description="Name of the vector collection in Qdrant")
    NEO4J_URI: str = Field("bolt://localhost:7687", description="URI for Neo4j graph database")
    NEO4J_USER: str = Field("neo4j", description="Username for Neo4j")
    NEO4J_PASSWORD: str = Field("password", description="Password for Neo4j")
    
    RETRIEVAL_TYPE: str = Field("hybrid", description="Type of retrieval to use (vector, graph, hybrid)")
    KNOWLEDGE_CONSTRUCTION_TYPE: str = Field("default", description="Type of knowledge construction to use")
    REASONING_TYPE: str = Field("default", description="Type of reasoning engine to use")
    
    MAX_RESULTS: int = Field(10, description="Maximum number of results to return")
    
    # New configuration options
    TEMPORAL_GRANULARITY: timedelta = Field(timedelta(hours=1), description="Granularity for temporal features")
    UNCERTAINTY_THRESHOLD: float = Field(0.1, description="Threshold for uncertainty handling")

    # Additional configuration for satisfactory results
    MIN_SATISFACTORY_RESULTS: int = Field(5, description="Minimum number of results to consider satisfactory")
    HIGH_SCORE_THRESHOLD: float = Field(0.8, description="Threshold for high-scoring results")
    MIN_HIGH_SCORE_RESULTS: int = Field(3, description="Minimum number of high-scoring results")
    MIN_DIVERSE_SOURCES: int = Field(2, description="Minimum number of diverse sources")

    # Configuration for vector and graph retrieval
    VECTOR_TOP_K: int = Field(20, description="Number of top results to retrieve from vector store")
    GRAPH_TOP_K: int = Field(20, description="Number of top results to retrieve from graph store")

class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class ActiveRAGConfig(RAGConfig):
    FEEDBACK_ITERATIONS: int = Field(3, description="Number of feedback iterations for active retrieval")

class PlanRAGConfig(RAGConfig):
    PLANNING_STEPS: int = Field(3, description="Number of planning steps for plan-aware retrieval")
