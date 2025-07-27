"""Standardized RAG (Retrieval-Augmented Generation) Interface

This module defines the standard interface for RAG systems, query processing,
document management, and embedding operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from core import ErrorContext


class QueryType(Enum):
    """Standard query types for RAG systems."""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    CREATIVE = "creative"
    CONVERSATIONAL = "conversational"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


class DocumentType(Enum):
    """Standard document types."""

    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"


class RetrievalStrategy(Enum):
    """Standard retrieval strategies."""

    VECTOR_SIMILARITY = "vector_similarity"
    KEYWORD_SEARCH = "keyword_search"
    HYBRID = "hybrid"
    GRAPH_TRAVERSAL = "graph_traversal"
    SEMANTIC_SEARCH = "semantic_search"
    FUZZY_SEARCH = "fuzzy_search"


@dataclass
class DocumentMetadata:
    """Metadata for documents in RAG systems."""

    document_id: str
    title: str
    source: str
    document_type: DocumentType
    language: str = "en"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    author: str | None = None
    tags: list[str] = field(default_factory=list)
    category: str | None = None
    confidence_score: float = 1.0
    word_count: int = 0
    character_count: int = 0
    custom_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentInterface:
    """Standard document interface for RAG systems."""

    document_id: str
    content: str
    metadata: DocumentMetadata
    chunks: list[str] | None = None
    embeddings: list[list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            "document_id": self.document_id,
            "content": self.content,
            "metadata": {
                "document_id": self.metadata.document_id,
                "title": self.metadata.title,
                "source": self.metadata.source,
                "document_type": self.metadata.document_type.value,
                "language": self.metadata.language,
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "author": self.metadata.author,
                "tags": self.metadata.tags,
                "category": self.metadata.category,
                "confidence_score": self.metadata.confidence_score,
                "word_count": self.metadata.word_count,
                "character_count": self.metadata.character_count,
                "custom_metadata": self.metadata.custom_metadata,
            },
            "chunks": self.chunks,
            "embeddings": self.embeddings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentInterface":
        """Create document from dictionary representation."""
        metadata_dict = data["metadata"]
        metadata = DocumentMetadata(
            document_id=metadata_dict["document_id"],
            title=metadata_dict["title"],
            source=metadata_dict["source"],
            document_type=DocumentType(metadata_dict["document_type"]),
            language=metadata_dict.get("language", "en"),
            created_at=datetime.fromisoformat(metadata_dict["created_at"]),
            updated_at=datetime.fromisoformat(metadata_dict["updated_at"]),
            author=metadata_dict.get("author"),
            tags=metadata_dict.get("tags", []),
            category=metadata_dict.get("category"),
            confidence_score=metadata_dict.get("confidence_score", 1.0),
            word_count=metadata_dict.get("word_count", 0),
            character_count=metadata_dict.get("character_count", 0),
            custom_metadata=metadata_dict.get("custom_metadata", {}),
        )

        return cls(
            document_id=data["document_id"],
            content=data["content"],
            metadata=metadata,
            chunks=data.get("chunks"),
            embeddings=data.get("embeddings"),
        )


@dataclass
class QueryInterface:
    """Standard query interface for RAG systems."""

    query_id: str
    query_text: str
    query_type: QueryType
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    max_results: int = 10
    similarity_threshold: float = 0.7
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.VECTOR_SIMILARITY
    filters: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert query to dictionary representation."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "query_type": self.query_type.value,
            "context": self.context,
            "metadata": self.metadata,
            "max_results": self.max_results,
            "similarity_threshold": self.similarity_threshold,
            "retrieval_strategy": self.retrieval_strategy.value,
            "filters": self.filters,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class RetrievalResult:
    """Result from document retrieval."""

    document: DocumentInterface
    relevance_score: float
    chunk_index: int | None = None
    chunk_text: str | None = None
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Result from query processing."""

    query: QueryInterface
    retrieved_documents: list[RetrievalResult]
    generated_response: str
    confidence_score: float
    processing_time_ms: float
    tokens_used: int = 0
    reasoning_trace: str | None = None
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class EmbeddingInterface(ABC):
    """Standard interface for embedding models."""

    @abstractmethod
    async def encode_text(self, text: str) -> list[float]:
        """Encode text into embedding vector.

        Args:
            text: Text to encode

        Returns:
            Embedding vector as list of floats
        """

    @abstractmethod
    async def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode batch of texts into embedding vectors.

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors
        """

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embedding vectors."""

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the embedding model."""


class DocumentStore(ABC):
    """Standard interface for document storage."""

    @abstractmethod
    async def add_document(self, document: DocumentInterface) -> bool:
        """Add document to store.

        Args:
            document: Document to add

        Returns:
            bool: True if document added successfully
        """

    @abstractmethod
    async def get_document(self, document_id: str) -> DocumentInterface | None:
        """Get document by ID.

        Args:
            document_id: ID of document to retrieve

        Returns:
            Document if found, None otherwise
        """

    @abstractmethod
    async def update_document(self, document: DocumentInterface) -> bool:
        """Update existing document.

        Args:
            document: Updated document

        Returns:
            bool: True if document updated successfully
        """

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete document by ID.

        Args:
            document_id: ID of document to delete

        Returns:
            bool: True if document deleted successfully
        """

    @abstractmethod
    async def search_documents(self, query: QueryInterface) -> list[RetrievalResult]:
        """Search documents based on query.

        Args:
            query: Search query

        Returns:
            List of retrieval results
        """

    @abstractmethod
    async def get_document_count(self) -> int:
        """Get total number of documents in store."""

    @abstractmethod
    async def list_documents(
        self, offset: int = 0, limit: int = 100, filters: dict[str, Any] | None = None
    ) -> list[DocumentInterface]:
        """List documents with pagination and filtering.

        Args:
            offset: Number of documents to skip
            limit: Maximum number of documents to return
            filters: Optional filters to apply

        Returns:
            List of documents
        """


class RAGInterface(ABC):
    """Main interface for RAG (Retrieval-Augmented Generation) systems.

    This interface defines the standard API for RAG systems that combine
    document retrieval with text generation.
    """

    def __init__(self):
        self.document_store: DocumentStore | None = None
        self.embedding_model: EmbeddingInterface | None = None
        self.retrieval_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_response_time_ms": 0.0,
            "total_documents_retrieved": 0,
        }

    @abstractmethod
    async def initialize(
        self,
        document_store: DocumentStore,
        embedding_model: EmbeddingInterface,
        **config,
    ) -> bool:
        """Initialize RAG system with required components.

        Args:
            document_store: Document storage system
            embedding_model: Embedding model for vector operations
            **config: Additional configuration parameters

        Returns:
            bool: True if initialization successful
        """

    @abstractmethod
    async def process_query(self, query: QueryInterface) -> QueryResult:
        """Process a query and return augmented response.

        Args:
            query: Query to process

        Returns:
            Query result with retrieved documents and generated response
        """

    @abstractmethod
    async def add_documents(
        self, documents: list[DocumentInterface]
    ) -> dict[str, bool]:
        """Add multiple documents to the RAG system.

        Args:
            documents: List of documents to add

        Returns:
            Dictionary mapping document IDs to success status
        """

    @abstractmethod
    async def update_document(self, document: DocumentInterface) -> bool:
        """Update existing document in the RAG system.

        Args:
            document: Updated document

        Returns:
            bool: True if update successful
        """

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete document from RAG system.

        Args:
            document_id: ID of document to delete

        Returns:
            bool: True if deletion successful
        """

    @abstractmethod
    async def retrieve_documents(self, query: QueryInterface) -> list[RetrievalResult]:
        """Retrieve relevant documents for query without generation.

        Args:
            query: Query for document retrieval

        Returns:
            List of relevant documents with relevance scores
        """

    @abstractmethod
    async def generate_response(
        self, query: str, context_documents: list[DocumentInterface]
    ) -> str:
        """Generate response based on query and context documents.

        Args:
            query: Original query
            context_documents: Retrieved documents for context

        Returns:
            Generated response
        """

    # Optional advanced methods

    async def explain_retrieval(
        self, query: QueryInterface, result: RetrievalResult
    ) -> str:
        """Provide explanation for why a document was retrieved.

        Args:
            query: Original query
            result: Retrieval result to explain

        Returns:
            Human-readable explanation
        """
        return f"Document '{result.document.metadata.title}' retrieved with relevance score {result.relevance_score:.3f}"

    async def get_similar_documents(
        self, document_id: str, max_results: int = 10
    ) -> list[RetrievalResult]:
        """Find documents similar to the given document.

        Args:
            document_id: ID of reference document
            max_results: Maximum number of similar documents to return

        Returns:
            List of similar documents with similarity scores
        """
        # Default implementation - can be overridden
        document = await self.document_store.get_document(document_id)
        if not document:
            return []

        query = QueryInterface(
            query_id=f"similar-{document_id}",
            query_text=document.content[:500],  # Use first 500 chars as query
            query_type=QueryType.ANALYTICAL,
            max_results=max_results,
        )

        return await self.retrieve_documents(query)

    def get_statistics(self) -> dict[str, Any]:
        """Get RAG system statistics."""
        return self.retrieval_stats.copy()

    def create_error_context(self, operation: str, **metadata) -> ErrorContext:
        """Create error context for RAG operations."""
        from core import create_rag_context

        return create_rag_context(operation=operation, **metadata)


# Utility functions


def create_document(
    content: str,
    title: str,
    source: str,
    document_type: DocumentType = DocumentType.TEXT,
    **metadata_kwargs,
) -> DocumentInterface:
    """Create a document with auto-generated ID.

    Args:
        content: Document content
        title: Document title
        source: Document source
        document_type: Type of document
        **metadata_kwargs: Additional metadata

    Returns:
        DocumentInterface instance
    """
    import uuid

    doc_id = str(uuid.uuid4())

    metadata = DocumentMetadata(
        document_id=doc_id,
        title=title,
        source=source,
        document_type=document_type,
        word_count=len(content.split()),
        character_count=len(content),
        **metadata_kwargs,
    )

    return DocumentInterface(document_id=doc_id, content=content, metadata=metadata)


def create_query(
    query_text: str,
    query_type: QueryType = QueryType.FACTUAL,
    max_results: int = 10,
    **query_kwargs,
) -> QueryInterface:
    """Create a query with auto-generated ID.

    Args:
        query_text: Query text
        query_type: Type of query
        max_results: Maximum results to retrieve
        **query_kwargs: Additional query parameters

    Returns:
        QueryInterface instance
    """
    import uuid

    return QueryInterface(
        query_id=str(uuid.uuid4()),
        query_text=query_text,
        query_type=query_type,
        max_results=max_results,
        **query_kwargs,
    )


def validate_rag_interface(rag: Any) -> bool:
    """Validate that an object implements RAGInterface.

    Args:
        rag: Object to validate

    Returns:
        bool: True if object implements interface correctly
    """
    required_methods = [
        "initialize",
        "process_query",
        "add_documents",
        "update_document",
        "delete_document",
        "retrieve_documents",
        "generate_response",
    ]

    for method in required_methods:
        if not hasattr(rag, method) or not callable(getattr(rag, method)):
            return False

    return True


def validate_document_store_interface(store: Any) -> bool:
    """Validate that an object implements DocumentStore interface.

    Args:
        store: Object to validate

    Returns:
        bool: True if object implements interface correctly
    """
    required_methods = [
        "add_document",
        "get_document",
        "update_document",
        "delete_document",
        "search_documents",
        "get_document_count",
        "list_documents",
    ]

    for method in required_methods:
        if not hasattr(store, method) or not callable(getattr(store, method)):
            return False

    return True


def validate_embedding_interface(embedding: Any) -> bool:
    """Validate that an object implements EmbeddingInterface.

    Args:
        embedding: Object to validate

    Returns:
        bool: True if object implements interface correctly
    """
    required_methods = [
        "encode_text",
        "encode_batch",
        "get_embedding_dimension",
        "get_model_info",
    ]

    for method in required_methods:
        if not hasattr(embedding, method) or not callable(getattr(embedding, method)):
            return False

    return True
