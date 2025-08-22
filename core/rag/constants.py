"""RAG (Retrieval-Augmented Generation) constants for AIVillage.

This module centralizes all RAG-related magic literals to eliminate
connascence of meaning and ensure consistent RAG configurations.
"""

from enum import Enum
from typing import Final

# Vector database configuration
DEFAULT_VECTOR_DIMENSION: Final[int] = 768
MAX_VECTOR_DIMENSION: Final[int] = 4096
DEFAULT_SIMILARITY_THRESHOLD: Final[float] = 0.7
MIN_SIMILARITY_THRESHOLD: Final[float] = 0.1
MAX_SIMILARITY_THRESHOLD: Final[float] = 1.0

# Retrieval parameters
DEFAULT_TOP_K: Final[int] = 10
MAX_TOP_K: Final[int] = 100
DEFAULT_CHUNK_SIZE: Final[int] = 512
MAX_CHUNK_SIZE: Final[int] = 2048
MIN_CHUNK_SIZE: Final[int] = 64
CHUNK_OVERLAP: Final[int] = 50

# Collection management
MAX_COLLECTIONS_PER_TENANT: Final[int] = 100
MAX_DOCUMENTS_PER_COLLECTION: Final[int] = 1_000_000
DEFAULT_BATCH_SIZE: Final[int] = 100
MAX_BATCH_SIZE: Final[int] = 1000

# Indexing and processing
INDEX_UPDATE_INTERVAL_SECONDS: Final[int] = 300  # 5 minutes
DOCUMENT_PROCESSING_TIMEOUT_SECONDS: Final[int] = 120
EMBEDDING_CACHE_TTL_SECONDS: Final[int] = 3600  # 1 hour
MAX_CONCURRENT_EMBEDDINGS: Final[int] = 10

# Text processing
MAX_TEXT_LENGTH_CHARS: Final[int] = 100_000
MIN_TEXT_LENGTH_CHARS: Final[int] = 10
DEFAULT_LANGUAGE: Final[str] = "en"
TEXT_ENCODING: Final[str] = "utf-8"

# Query optimization
QUERY_CACHE_SIZE: Final[int] = 1000
QUERY_TIMEOUT_SECONDS: Final[int] = 30
MAX_QUERY_LENGTH: Final[int] = 1000
RERANK_TOP_K: Final[int] = 50

# Knowledge graph integration
MAX_GRAPH_DEPTH: Final[int] = 5
GRAPH_TRAVERSAL_TIMEOUT_SECONDS: Final[int] = 10
ENTITY_EXTRACTION_BATCH_SIZE: Final[int] = 50
RELATION_CONFIDENCE_THRESHOLD: Final[float] = 0.8

# Hybrid search configuration
SEMANTIC_WEIGHT: Final[float] = 0.7
KEYWORD_WEIGHT: Final[float] = 0.3
FUSION_ALGORITHM: Final[str] = "reciprocal_rank_fusion"
RRF_CONSTANT: Final[int] = 60


class VectorDatabase(Enum):
    """Supported vector database backends."""

    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    FAISS = "faiss"


class EmbeddingModel(Enum):
    """Supported embedding models."""

    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    COHERE_EMBED = "embed-english-v3.0"
    HUGGINGFACE_BGE = "BAAI/bge-large-en-v1.5"


class ChunkingStrategy(Enum):
    """Document chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"
    PARAGRAPH_BASED = "paragraph_based"
    SEMANTIC_BASED = "semantic_based"
    RECURSIVE_CHARACTER = "recursive_character"


class RetrievalMode(Enum):
    """Retrieval modes for different use cases."""

    SEMANTIC_ONLY = "semantic_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    GRAPH_ENHANCED = "graph_enhanced"
    MULTI_MODAL = "multi_modal"


class DocumentStatus(Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


# File format support
SUPPORTED_TEXT_FORMATS: Final[tuple[str, ...]] = (".txt", ".md", ".json", ".yaml", ".yml", ".csv")

SUPPORTED_DOCUMENT_FORMATS: Final[tuple[str, ...]] = (".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls")

SUPPORTED_IMAGE_FORMATS: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")

# Performance tuning
EMBEDDING_BATCH_SIZE: Final[int] = 32
INDEX_REFRESH_THRESHOLD: Final[int] = 1000
MEMORY_USAGE_LIMIT_MB: Final[int] = 2048
DISK_CACHE_SIZE_GB: Final[int] = 10

# Quality metrics
RELEVANCE_SCORE_THRESHOLD: Final[float] = 0.5
DIVERSITY_PENALTY_FACTOR: Final[float] = 0.1
COVERAGE_TARGET_PERCENT: Final[int] = 80
PRECISION_TARGET_PERCENT: Final[int] = 90


class RAGMessages:
    """Standardized RAG system messages."""

    COLLECTION_CREATED: Final[str] = "Collection '{collection_name}' created successfully"
    DOCUMENT_INDEXED: Final[str] = "Document '{doc_id}' indexed in collection '{collection}'"
    QUERY_EXECUTED: Final[str] = "Query executed: {results_count} results in {duration_ms}ms"
    EMBEDDING_CACHED: Final[str] = "Embedding cached for text length {length}"
    INDEX_UPDATED: Final[str] = "Index updated: {documents_added} documents added"
    RETRIEVAL_FAILED: Final[str] = "Retrieval failed for query '{query}': {error}"
    COLLECTION_NOT_FOUND: Final[str] = "Collection '{collection_name}' not found"
    INSUFFICIENT_RESULTS: Final[str] = "Only {found} results found, requested {requested}"
    PROCESSING_COMPLETE: Final[str] = "Document processing complete: {success}/{total} successful"
    CACHE_HIT: Final[str] = "Cache hit for query: {query_hash}"
    KNOWLEDGE_GRAPH_UPDATED: Final[str] = "Knowledge graph updated: {entities} entities, {relations} relations"


# API configuration
RAG_API_VERSION: Final[str] = "v1"
MAX_CONCURRENT_REQUESTS: Final[int] = 100
REQUEST_TIMEOUT_SECONDS: Final[int] = 60
RATE_LIMIT_PER_MINUTE: Final[int] = 1000

# Storage configuration
DEFAULT_STORAGE_PATH: Final[str] = "data/rag"
INDEX_STORAGE_PATH: Final[str] = "data/rag/indexes"
CACHE_STORAGE_PATH: Final[str] = "data/rag/cache"
BACKUP_STORAGE_PATH: Final[str] = "data/rag/backups"

# Monitoring and logging
LOG_QUERY_PERFORMANCE: Final[bool] = True
LOG_EMBEDDING_STATS: Final[bool] = True
METRIC_COLLECTION_INTERVAL_SECONDS: Final[int] = 60
PERFORMANCE_ALERT_THRESHOLD_MS: Final[int] = 5000

# Security and privacy
ENABLE_QUERY_LOGGING: Final[bool] = True
ANONYMIZE_QUERIES: Final[bool] = False
ENCRYPT_EMBEDDINGS: Final[bool] = True
ACCESS_LOG_RETENTION_DAYS: Final[int] = 90

# Multi-tenancy
TENANT_ISOLATION_ENABLED: Final[bool] = True
SHARED_EMBEDDINGS_ENABLED: Final[bool] = False
TENANT_QUOTA_DOCUMENTS: Final[int] = 100_000
TENANT_QUOTA_STORAGE_GB: Final[int] = 50

# Advanced features
ENABLE_AUTO_RERANKING: Final[bool] = True
ENABLE_QUERY_EXPANSION: Final[bool] = True
ENABLE_CONTEXTUAL_COMPRESSION: Final[bool] = True
ENABLE_MULTI_HOP_REASONING: Final[bool] = False

# Model configuration
DEFAULT_MODEL_NAME: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_DIR: Final[str] = ".cache/rag_models"
MODEL_DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 300
MODEL_LOADING_TIMEOUT_SECONDS: Final[int] = 60
