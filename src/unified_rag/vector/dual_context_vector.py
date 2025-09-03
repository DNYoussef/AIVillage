"""
Dual Context Vector RAG - Advanced Vector Search with Hierarchical Context

High-performance vector similarity search with dual context tags,
contextual embeddings, and hierarchical retrieval using HuggingFace MCP.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ContextLevel(Enum):
    """Levels of contextual information."""
    
    DOCUMENT = "document"      # Document-level context
    CHAPTER = "chapter"        # Chapter/section context  
    PARAGRAPH = "paragraph"    # Paragraph-level context
    SENTENCE = "sentence"      # Sentence-level context


class SearchMode(Enum):
    """Vector search modes."""
    
    SEMANTIC = "semantic"      # Pure semantic similarity
    CONTEXTUAL = "contextual"  # Context-aware search
    HIERARCHICAL = "hierarchical" # Multi-level hierarchical
    HYBRID = "hybrid"          # Combined approaches


class ChunkingStrategy(Enum):
    """Strategies for document chunking."""

    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    HIERARCHICAL = "hierarchical"


class SimilarityMetric(Enum):
    """Similarity metrics for vector comparison."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


@dataclass
class ContextTag:
    """Hierarchical context tag."""
    
    level: ContextLevel
    content: str
    confidence: float = 1.0
    
    # Hierarchical relationships
    parent_context: Optional['ContextTag'] = None
    child_contexts: List['ContextTag'] = field(default_factory=list)
    
    # Vector representation
    embedding: Optional[np.ndarray] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorDocument:
    """Document with dual context vector representations."""
    
    doc_id: str
    content: str
    title: str = ""
    
    # Dual context system
    primary_context: Optional[ContextTag] = None    # Book/document level
    secondary_context: Optional[ContextTag] = None  # Chapter/section level
    local_contexts: List[ContextTag] = field(default_factory=list)  # Paragraph/sentence level
    
    # Multiple embedding representations
    content_embedding: Optional[np.ndarray] = None
    title_embedding: Optional[np.ndarray] = None
    context_embedding: Optional[np.ndarray] = None
    combined_embedding: Optional[np.ndarray] = None
    
    # Document metadata
    chunk_index: int = 0
    total_chunks: int = 1
    parent_document: str = ""
    
    # Quality scores
    relevance_score: float = 0.0
    context_match_score: float = 0.0
    semantic_score: float = 0.0
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class VectorSearchResult:
    """Result from dual context vector search."""
    
    documents: List[VectorDocument] = field(default_factory=list)
    search_time_ms: float = 0.0
    total_candidates: int = 0
    
    # Search metadata
    search_mode: SearchMode = SearchMode.SEMANTIC
    context_used: bool = False
    reranked: bool = False
    
    # Quality metrics
    avg_relevance: float = 0.0
    context_match_rate: float = 0.0
    semantic_diversity: float = 0.0
    
    # Performance
    embedding_time_ms: float = 0.0
    search_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class DualContextVectorRAG:
    """
    Dual Context Vector RAG System
    
    Advanced vector similarity search with hierarchical context awareness,
    multiple embedding strategies, and intelligent search mode selection.
    Uses HuggingFace MCP for embedding generation and model coordination.
    
    Features:
    - Dual context tagging (book/chapter hierarchy)
    - Multiple embedding representations per document
    - Context-aware similarity scoring
    - Hierarchical search strategies
    - Intelligent search mode selection
    - HuggingFace MCP integration for embeddings
    """
    
    def __init__(
        self,
        mcp_coordinator=None,
        embedding_dim: int = 768,
        similarity_threshold: float = 0.7,
        max_documents: int = 50000,
        enable_hierarchical_search: bool = True
    ):
        self.mcp_coordinator = mcp_coordinator
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.max_documents = max_documents
        self.enable_hierarchical_search = enable_hierarchical_search
        
        # Document storage
        self.documents: Dict[str, VectorDocument] = {}
        self.vector_index: Optional[np.ndarray] = None
        self.doc_id_mapping: List[str] = []
        
        # Context hierarchies
        self.context_hierarchies: Dict[str, List[ContextTag]] = {}
        self.context_embeddings: Dict[str, np.ndarray] = {}
        
        # Search components
        self.embedding_engine = None
        self.hierarchical_engine = None
        
        # Caching
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.search_cache: Dict[str, VectorSearchResult] = {}
        
        # Statistics
        self.stats = {
            "documents_indexed": 0,
            "searches_performed": 0,
            "embeddings_generated": 0,
            "context_matches": 0,
            "hierarchical_searches": 0,
            "mcp_calls": 0
        }
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the dual context vector RAG system."""
        try:
            logger.info("🔍 Initializing Dual Context Vector RAG...")
            
            # Initialize embedding engine
            from .contextual_embeddings import ContextualEmbeddingEngine
            self.embedding_engine = ContextualEmbeddingEngine(
                mcp_coordinator=self.mcp_coordinator,
                embedding_dim=self.embedding_dim
            )
            await self.embedding_engine.initialize()
            
            # Initialize hierarchical search if enabled
            if self.enable_hierarchical_search:
                from .hierarchical_search import HierarchicalSearchEngine
                self.hierarchical_engine = HierarchicalSearchEngine(
                    similarity_threshold=self.similarity_threshold
                )
                await self.hierarchical_engine.initialize()
            
            # Initialize vector index
            self.vector_index = np.empty((0, self.embedding_dim), dtype=np.float32)
            
            self.initialized = True
            logger.info("✅ Dual Context Vector RAG initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dual Context Vector RAG initialization failed: {e}")
            return False
    
    async def index_document(
        self,
        content: str,
        doc_id: str,
        title: str = "",
        primary_context: Optional[str] = None,
        secondary_context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Index document with dual context processing."""
        try:
            logger.debug(f"Indexing document: {doc_id}")
            
            # Create vector document
            doc = VectorDocument(
                doc_id=doc_id,
                content=content,
                title=title,
                metadata=metadata or {}
            )
            
            # Generate context tags
            if primary_context:
                doc.primary_context = ContextTag(
                    level=ContextLevel.DOCUMENT,
                    content=primary_context,
                    confidence=0.9
                )
            
            if secondary_context:
                doc.secondary_context = ContextTag(
                    level=ContextLevel.CHAPTER,
                    content=secondary_context,
                    confidence=0.8,
                    parent_context=doc.primary_context
                )
            
            # Extract local contexts
            doc.local_contexts = await self._extract_local_contexts(content)
            
            # Generate embeddings using MCP
            await self._generate_document_embeddings(doc)
            
            # Store document
            self.documents[doc_id] = doc
            
            # Update vector index
            await self._update_vector_index(doc)
            
            # Update context hierarchies
            await self._update_context_hierarchies(doc)
            
            self.stats["documents_indexed"] += 1
            logger.debug(f"Successfully indexed document {doc_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Document indexing failed for {doc_id}: {e}")
            return False
    
    async def search(
        self,
        query: str,
        k: int = 10,
        search_mode: SearchMode = SearchMode.HYBRID,
        context: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> List[VectorDocument]:
        """Perform dual context vector search."""
        start_time = time.time()
        
        try:
            logger.debug(f"Searching for: '{query[:50]}...' (mode: {search_mode.value})")
            
            # Check cache
            cache_key = f"{hash(query)}_{k}_{search_mode.value}_{hash(str(context))}"
            if cache_key in self.search_cache:
                cached_result = self.search_cache[cache_key]
                return cached_result.documents
            
            # Generate query embedding
            query_embedding = await self._generate_query_embedding(query, context)
            
            # Perform search based on mode
            if search_mode == SearchMode.SEMANTIC:
                results = await self._semantic_search(query_embedding, k)
            elif search_mode == SearchMode.CONTEXTUAL:
                results = await self._contextual_search(query_embedding, query, context, k)
            elif search_mode == SearchMode.HIERARCHICAL:
                results = await self._hierarchical_search(query_embedding, query, context, k)
            else:  # HYBRID
                results = await self._hybrid_search(query_embedding, query, context, k)
            
            # Post-process results
            results = await self._post_process_results(results, query, context, user_preferences)
            
            # Create and cache search result
            search_time = (time.time() - start_time) * 1000
            search_result = VectorSearchResult(
                documents=results,
                search_time_ms=search_time,
                total_candidates=len(results),
                search_mode=search_mode,
                context_used=context is not None,
                avg_relevance=np.mean([doc.relevance_score for doc in results]) if results else 0.0
            )
            
            self.search_cache[cache_key] = search_result
            self.stats["searches_performed"] += 1
            
            logger.debug(f"Search completed: {len(results)} results in {search_time:.1f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _generate_document_embeddings(self, doc: VectorDocument):
        """Generate all embedding representations for document."""
        try:
            # Content embedding
            doc.content_embedding = await self._get_embedding(doc.content, "content")
            
            # Title embedding
            if doc.title:
                doc.title_embedding = await self._get_embedding(doc.title, "title")
            
            # Context embedding (combined contexts)
            context_text = self._build_context_text(doc)
            if context_text:
                doc.context_embedding = await self._get_embedding(context_text, "context")
            
            # Combined embedding (weighted combination)
            doc.combined_embedding = await self._create_combined_embedding(doc)
            
            # Generate context embeddings
            await self._generate_context_embeddings(doc)
            
            self.stats["embeddings_generated"] += 1
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
    
    async def _get_embedding(self, text: str, embedding_type: str = "content") -> np.ndarray:
        """Get embedding using MCP coordinator or fallback."""
        # Check cache
        cache_key = f"{embedding_type}_{hash(text)}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Use MCP coordinator for embeddings
            if self.mcp_coordinator:
                embeddings = await self.mcp_coordinator.generate_embeddings([text])
                if embeddings is not None and len(embeddings) > 0:
                    embedding = embeddings[0]
                    self.embedding_cache[cache_key] = embedding
                    self.stats["mcp_calls"] += 1
                    return embedding
            
            # Fallback embedding
            return await self._create_fallback_embedding(text)
            
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return await self._create_fallback_embedding(text)
    
    async def _create_fallback_embedding(self, text: str) -> np.ndarray:
        """Create deterministic fallback embedding."""
        import hashlib
        
        text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
        seed = int(text_hash[:8], 16)
        
        np.random.seed(seed)
        embedding = np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def _build_context_text(self, doc: VectorDocument) -> str:
        """Build combined context text from all contexts."""
        context_parts = []
        
        if doc.primary_context:
            context_parts.append(f"[DOCUMENT] {doc.primary_context.content}")
        
        if doc.secondary_context:
            context_parts.append(f"[CHAPTER] {doc.secondary_context.content}")
        
        for local_context in doc.local_contexts:
            context_parts.append(f"[{local_context.level.value.upper()}] {local_context.content}")
        
        return " | ".join(context_parts)
    
    async def _create_combined_embedding(self, doc: VectorDocument) -> np.ndarray:
        """Create weighted combination of all embeddings."""
        embeddings = []
        weights = []
        
        if doc.content_embedding is not None:
            embeddings.append(doc.content_embedding)
            weights.append(0.6)  # Primary weight for content
        
        if doc.title_embedding is not None:
            embeddings.append(doc.title_embedding)
            weights.append(0.2)  # Secondary weight for title
        
        if doc.context_embedding is not None:
            embeddings.append(doc.context_embedding)
            weights.append(0.2)  # Secondary weight for context
        
        if not embeddings:
            return await self._create_fallback_embedding(doc.content)
        
        # Weighted combination
        combined = np.zeros(self.embedding_dim, dtype=np.float32)
        total_weight = sum(weights)
        
        for embedding, weight in zip(embeddings, weights):
            combined += embedding * (weight / total_weight)
        
        # Normalize
        combined = combined / (np.linalg.norm(combined) + 1e-8)
        return combined
    
    async def _generate_context_embeddings(self, doc: VectorDocument):
        """Generate embeddings for all context tags."""
        # Primary context embedding
        if doc.primary_context and not doc.primary_context.embedding is not None:
            doc.primary_context.embedding = await self._get_embedding(
                doc.primary_context.content, "primary_context"
            )
        
        # Secondary context embedding
        if doc.secondary_context and not doc.secondary_context.embedding is not None:
            doc.secondary_context.embedding = await self._get_embedding(
                doc.secondary_context.content, "secondary_context"
            )
        
        # Local context embeddings
        for context in doc.local_contexts:
            if context.embedding is None:
                context.embedding = await self._get_embedding(
                    context.content, f"local_{context.level.value}"
                )
    
    async def _extract_local_contexts(self, content: str) -> List[ContextTag]:
        """Extract local context tags from content."""
        contexts = []
        
        # Extract paragraph contexts
        paragraphs = content.split('\n\n')
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph) > 50:  # Minimum paragraph length
                context = ContextTag(
                    level=ContextLevel.PARAGRAPH,
                    content=paragraph[:200] + "..." if len(paragraph) > 200 else paragraph,
                    confidence=0.7,
                    metadata={"paragraph_index": i}
                )
                contexts.append(context)
        
        return contexts[:5]  # Limit to top 5 paragraphs
    
    async def _update_vector_index(self, doc: VectorDocument):
        """Update vector index with new document."""
        if doc.combined_embedding is None:
            return
        
        # Add to vector index
        if self.vector_index.shape[0] == 0:
            self.vector_index = doc.combined_embedding.reshape(1, -1)
        else:
            self.vector_index = np.vstack([self.vector_index, doc.combined_embedding])
        
        # Update document ID mapping
        self.doc_id_mapping.append(doc.doc_id)
    
    async def _update_context_hierarchies(self, doc: VectorDocument):
        """Update context hierarchy index."""
        # Index primary context
        if doc.primary_context:
            primary_key = f"document_{hash(doc.primary_context.content) % 10000}"
            if primary_key not in self.context_hierarchies:
                self.context_hierarchies[primary_key] = []
            self.context_hierarchies[primary_key].append(doc.primary_context)
        
        # Index secondary context
        if doc.secondary_context:
            secondary_key = f"chapter_{hash(doc.secondary_context.content) % 10000}"
            if secondary_key not in self.context_hierarchies:
                self.context_hierarchies[secondary_key] = []
            self.context_hierarchies[secondary_key].append(doc.secondary_context)
    
    async def _generate_query_embedding(self, query: str, context: Optional[str] = None) -> np.ndarray:
        """Generate query embedding with optional context."""
        if context:
            # Combine query with context for enhanced embedding
            enhanced_query = f"{query} [CONTEXT: {context}]"
            return await self._get_embedding(enhanced_query, "query_with_context")
        else:
            return await self._get_embedding(query, "query")
    
    async def _semantic_search(self, query_embedding: np.ndarray, k: int) -> List[VectorDocument]:
        """Pure semantic similarity search."""
        if self.vector_index.shape[0] == 0:
            return []
        
        # Calculate similarities
        similarities = np.dot(self.vector_index, query_embedding)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Filter by threshold
        results = []
        for idx in top_indices:
            if similarities[idx] >= self.similarity_threshold:
                doc_id = self.doc_id_mapping[idx]
                doc = self.documents[doc_id]
                doc.semantic_score = float(similarities[idx])
                doc.relevance_score = doc.semantic_score
                results.append(doc)
        
        return results
    
    async def _contextual_search(
        self, 
        query_embedding: np.ndarray, 
        query: str, 
        context: Optional[str], 
        k: int
    ) -> List[VectorDocument]:
        """Context-aware search with context matching."""
        # Start with semantic search
        semantic_results = await self._semantic_search(query_embedding, k * 2)
        
        if not context:
            return semantic_results[:k]
        
        # Enhance with context matching
        context_enhanced_results = []
        for doc in semantic_results:
            # Calculate context match score
            context_score = await self._calculate_context_match(doc, context)
            
            # Combined score
            doc.context_match_score = context_score
            doc.relevance_score = (doc.semantic_score * 0.7) + (context_score * 0.3)
            
            context_enhanced_results.append(doc)
        
        # Re-sort by combined score
        context_enhanced_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        self.stats["context_matches"] += len([d for d in context_enhanced_results if d.context_match_score > 0.5])
        
        return context_enhanced_results[:k]
    
    async def _hierarchical_search(
        self,
        query_embedding: np.ndarray,
        query: str,
        context: Optional[str],
        k: int
    ) -> List[VectorDocument]:
        """Hierarchical search using context hierarchies."""
        if not self.hierarchical_engine:
            return await self._contextual_search(query_embedding, query, context, k)
        
        results = await self.hierarchical_engine.search(
            query_embedding=query_embedding,
            query_text=query,
            context=context,
            documents=list(self.documents.values()),
            k=k
        )
        
        self.stats["hierarchical_searches"] += 1
        return results
    
    async def _hybrid_search(
        self,
        query_embedding: np.ndarray,
        query: str,
        context: Optional[str],
        k: int
    ) -> List[VectorDocument]:
        """Hybrid search combining multiple strategies."""
        # Get results from multiple strategies
        semantic_results = await self._semantic_search(query_embedding, k)
        contextual_results = await self._contextual_search(query_embedding, query, context, k)
        
        # Combine and deduplicate
        combined_results = {}
        
        # Add semantic results with weight
        for doc in semantic_results:
            doc_copy = self._copy_document(doc)
            doc_copy.relevance_score = doc.semantic_score * 0.5
            combined_results[doc.doc_id] = doc_copy
        
        # Add contextual results with weight
        for doc in contextual_results:
            if doc.doc_id in combined_results:
                # Combine scores
                combined_results[doc.doc_id].relevance_score += doc.relevance_score * 0.5
                combined_results[doc.doc_id].context_match_score = doc.context_match_score
            else:
                doc_copy = self._copy_document(doc)
                doc_copy.relevance_score = doc.relevance_score * 0.5
                combined_results[doc.doc_id] = doc_copy
        
        # Sort by combined relevance
        final_results = sorted(
            combined_results.values(),
            key=lambda x: x.relevance_score,
            reverse=True
        )
        
        return final_results[:k]
    
    def _copy_document(self, doc: VectorDocument) -> VectorDocument:
        """Create a copy of document for result processing."""
        # Simple copy - in production would use proper deep copy
        copy_doc = VectorDocument(
            doc_id=doc.doc_id,
            content=doc.content,
            title=doc.title,
            primary_context=doc.primary_context,
            secondary_context=doc.secondary_context,
            local_contexts=doc.local_contexts.copy(),
            metadata=doc.metadata.copy()
        )
        
        # Copy embeddings
        copy_doc.content_embedding = doc.content_embedding
        copy_doc.title_embedding = doc.title_embedding
        copy_doc.context_embedding = doc.context_embedding
        copy_doc.combined_embedding = doc.combined_embedding
        
        # Copy scores
        copy_doc.relevance_score = doc.relevance_score
        copy_doc.context_match_score = doc.context_match_score
        copy_doc.semantic_score = doc.semantic_score
        
        return copy_doc
    
    async def _calculate_context_match(self, doc: VectorDocument, context: str) -> float:
        """Calculate how well document context matches query context."""
        context_lower = context.lower()
        match_score = 0.0
        
        # Check primary context
        if doc.primary_context:
            primary_words = set(doc.primary_context.content.lower().split())
            context_words = set(context_lower.split())
            if primary_words and context_words:
                overlap = len(primary_words.intersection(context_words))
                match_score += (overlap / len(primary_words.union(context_words))) * 0.5
        
        # Check secondary context
        if doc.secondary_context:
            secondary_words = set(doc.secondary_context.content.lower().split())
            context_words = set(context_lower.split())
            if secondary_words and context_words:
                overlap = len(secondary_words.intersection(context_words))
                match_score += (overlap / len(secondary_words.union(context_words))) * 0.3
        
        # Check local contexts
        for local_context in doc.local_contexts:
            local_words = set(local_context.content.lower().split())
            context_words = set(context_lower.split())
            if local_words and context_words:
                overlap = len(local_words.intersection(context_words))
                match_score += (overlap / len(local_words.union(context_words))) * 0.2 / len(doc.local_contexts)
        
        return min(1.0, match_score)
    
    async def _post_process_results(
        self,
        results: List[VectorDocument],
        query: str,
        context: Optional[str],
        user_preferences: Optional[Dict[str, Any]]
    ) -> List[VectorDocument]:
        """Post-process search results for quality and personalization."""
        if not results:
            return results
        
        # Apply user preferences if provided
        if user_preferences:
            for doc in results:
                preference_boost = 0.0
                for pref_key, pref_value in user_preferences.items():
                    if pref_key in doc.metadata and doc.metadata[pref_key] == pref_value:
                        preference_boost += 0.1
                
                doc.relevance_score = min(1.0, doc.relevance_score + preference_boost)
        
        # Re-sort after preference adjustment
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Calculate diversity (simplified)
        if len(results) > 1:
            diversity_scores = []
            for i, doc in enumerate(results):
                # Simple diversity based on content differences
                other_docs = results[:i] + results[i+1:]
                avg_similarity = 0.0
                
                if other_docs and doc.combined_embedding is not None:
                    similarities = []
                    for other_doc in other_docs:
                        if other_doc.combined_embedding is not None:
                            sim = np.dot(doc.combined_embedding, other_doc.combined_embedding)
                            similarities.append(sim)
                    
                    if similarities:
                        avg_similarity = np.mean(similarities)
                
                diversity = 1.0 - avg_similarity
                diversity_scores.append(diversity)
            
            # Apply diversity bonus (small)
            for doc, diversity in zip(results, diversity_scores):
                doc.relevance_score = doc.relevance_score + (diversity * 0.05)
        
        return results
    
    async def get_status(self) -> Dict[str, Any]:
        """Get dual context vector RAG status."""
        return {
            "initialized": self.initialized,
            "document_count": len(self.documents),
            "vector_index_size": self.vector_index.shape[0] if self.vector_index is not None else 0,
            "context_hierarchies": len(self.context_hierarchies),
            "cache_sizes": {
                "embeddings": len(self.embedding_cache),
                "searches": len(self.search_cache)
            },
            "configuration": {
                "embedding_dim": self.embedding_dim,
                "similarity_threshold": self.similarity_threshold,
                "hierarchical_search_enabled": self.enable_hierarchical_search,
                "mcp_integration": self.mcp_coordinator is not None
            },
            "statistics": self.stats.copy()
        }
    
    async def close(self):
        """Close the dual context vector RAG system."""
        logger.info("Shutting down Dual Context Vector RAG...")
        
        # Clear data structures
        self.documents.clear()
        self.context_hierarchies.clear()
        self.context_embeddings.clear()
        self.embedding_cache.clear()
        self.search_cache.clear()
        
        # Clear vector index
        self.vector_index = None
        self.doc_id_mapping.clear()
        
        # Close components
        if self.embedding_engine:
            await self.embedding_engine.close()
        
        if self.hierarchical_engine:
            await self.hierarchical_engine.close()

        self.initialized = False
        logger.info("Dual Context Vector RAG shutdown complete")


def create_context_tag(tag_type: str, content: str, level: int = 0, confidence: float = 1.0) -> ContextTag:
    """Create a context tag for dual context system."""
    level_map = {
        "book": ContextLevel.DOCUMENT,
        "chapter": ContextLevel.CHAPTER,
        "paragraph": ContextLevel.PARAGRAPH,
        "sentence": ContextLevel.SENTENCE,
    }
    context_level = level_map.get(tag_type, ContextLevel.DOCUMENT)
    return ContextTag(level=context_level, content=content, confidence=confidence)


def create_book_chapter_contexts(
    book_title: str, book_summary: str, chapter_title: str, chapter_summary: str
) -> tuple[ContextTag, ContextTag]:
    """Create book and chapter context tags."""
    book_context = create_context_tag("book", f"{book_title}: {book_summary}", level=0, confidence=1.0)
    chapter_context = create_context_tag(
        "chapter", f"{chapter_title}: {chapter_summary}", level=1, confidence=1.0
    )
    return book_context, chapter_context


# Backwards compatibility alias
ContextualVectorEngine = DualContextVectorRAG