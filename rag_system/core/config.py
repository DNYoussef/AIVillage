# rag_system/core/config.py

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
    
class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class ActiveRAGConfig(RAGConfig):
    FEEDBACK_ITERATIONS: int = Field(3, description="Number of feedback iterations for active retrieval")

class PlanRAGConfig(RAGConfig):
    PLANNING_STEPS: int = Field(3, description="Number of planning steps for plan-aware retrieval")
