"""Contextual RAG Pipeline with Two-Level Contextual Tagging.

Integrates the Enhanced CODEX RAG Pipeline with comprehensive contextual tagging
for precise retrieval with rich bilateral context (document ↔ chunk).

Features:
- Two-level contextual metadata (document + chunk)
- Context inheritance and chain preservation
- Enhanced retrieval with contextual filtering
- Rich metadata for improved question answering
- Rhetorical structure preservation
"""

import asyncio
import logging
import time
from typing import Any

from src.production.rag.rag_system.core.codex_rag_integration import (
    Chunk,
    Document,
    RetrievalResult,
)
from src.production.rag.rag_system.core.contextual_tagging import (
    ChunkType,
    ContentDomain,
    ContextualTagger,
    DocumentContext,
    DocumentType,
    ReadingLevel,
)
from src.production.rag.rag_system.core.enhanced_codex_rag import (
    EnhancedCODEXRAGPipeline,
)

logger = logging.getLogger(__name__)


class ContextualRAGPipeline(EnhancedCODEXRAGPipeline):
    """Enhanced RAG Pipeline with comprehensive contextual tagging.

    Extends the intelligent chunking pipeline with:
    - Two-level contextual metadata extraction
    - Context inheritance and chain preservation
    - Rich bilateral context for precise retrieval
    - Advanced filtering based on contextual features
    """

    def __init__(
        self,
        enable_intelligent_chunking: bool = True,
        enable_contextual_tagging: bool = True,
        chunking_window_size: int = 3,
        chunking_min_sentences: int = 2,
        chunking_max_sentences: int = 15,
        chunking_context_overlap: int = 1,
        similarity_thresholds: dict | None = None,
        contextual_embedding_model: str = "paraphrase-MiniLM-L3-v2",
    ) -> None:
        """Initialize contextual RAG pipeline."""
        # Initialize parent class
        super().__init__(
            enable_intelligent_chunking=enable_intelligent_chunking,
            chunking_window_size=chunking_window_size,
            chunking_min_sentences=chunking_min_sentences,
            chunking_max_sentences=chunking_max_sentences,
            chunking_context_overlap=chunking_context_overlap,
            similarity_thresholds=similarity_thresholds,
        )

        # Initialize contextual tagger
        self.enable_contextual_tagging = enable_contextual_tagging
        if self.enable_contextual_tagging:
            logger.info("Initializing contextual tagger...")
            self.contextual_tagger = ContextualTagger(
                embedding_model=contextual_embedding_model,
                enable_spacy=False,  # Disable spaCy due to compatibility issues
            )
            logger.info("Contextual tagger initialized successfully")
        else:
            self.contextual_tagger = None

        # Storage for document contexts
        self.document_contexts = {}

        # Enhanced statistics
        self.contextual_stats = {
            "documents_with_context": 0,
            "chunks_with_context": 0,
            "avg_context_richness": 0.0,
            "context_inheritance_chains": 0,
            "domain_classifications": {},
            "reading_level_distribution": {},
            "chunk_type_distribution": {},
        }

    def chunk_document_with_context(self, document: Document, force_traditional: bool = False) -> list[Chunk]:
        """Chunk document with full contextual tagging.

        Args:
            document: Document to chunk with context
            force_traditional: Force traditional chunking

        Returns:
            List of chunks with rich contextual metadata
        """
        if not self.enable_contextual_tagging:
            # Fall back to intelligent chunking only
            return self.chunk_document_intelligently(document, force_traditional)

        try:
            logger.info(f"Processing document with contextual tagging: {document.id}")

            # Step 1: Extract document-level context
            document_context = self.contextual_tagger.extract_document_context(
                document_id=document.id,
                title=document.title,
                content=document.content,
                metadata=document.metadata,
            )

            # Store document context
            self.document_contexts[document.id] = document_context

            # Step 2: Perform intelligent chunking
            base_chunks = self.chunk_document_intelligently(document, force_traditional)

            # Step 3: Add contextual metadata to chunks
            contextual_chunks = []
            previous_chunk_context = None

            for i, base_chunk in enumerate(base_chunks):
                # Create contextual chunk with bilateral context
                contextual_chunk_data = self.contextual_tagger.create_contextual_chunk(
                    chunk_id=base_chunk.id,
                    chunk_text=base_chunk.text,
                    chunk_position=i,
                    start_char=base_chunk.start_idx,
                    end_char=base_chunk.end_idx,
                    document_context=document_context,
                    full_document_text=document.content,
                    previous_chunk_context=previous_chunk_context,
                )

                # Enhance chunk metadata with contextual information
                enhanced_metadata = base_chunk.metadata.copy() if base_chunk.metadata else {}

                # Add Level 1 context (Document)
                enhanced_metadata.update(
                    {
                        "document_context": contextual_chunk_data["document_context"],
                        "document_domain": document_context.domain.value,
                        "document_type": document_context.document_type.value,
                        "reading_level": document_context.reading_level.value,
                        "document_themes": document_context.key_themes,
                        "document_concepts": document_context.key_concepts,
                        "credibility_score": document_context.source_credibility_score,
                        "publication_info": {
                            "author": document_context.author,
                            "publication_date": document_context.publication_date,
                            "publisher": document_context.publisher,
                        },
                    }
                )

                # Add Level 2 context (Chunk)
                enhanced_metadata.update(
                    {
                        "chunk_context": contextual_chunk_data["chunk_context"],
                        "local_summary": contextual_chunk_data["chunk_context"]["local_summary"],
                        "chunk_type": contextual_chunk_data["chunk_context"]["chunk_type"],
                        "section_hierarchy": contextual_chunk_data["chunk_context"]["section_hierarchy"],
                        "chapter_info": {
                            "number": contextual_chunk_data["chunk_context"]["chapter_number"],
                            "title": contextual_chunk_data["chunk_context"]["chapter_title"],
                        },
                        "contextual_entities": contextual_chunk_data["chunk_context"]["key_entities"],
                        "local_keywords": contextual_chunk_data["chunk_context"]["local_keywords"],
                        "discourse_markers": contextual_chunk_data["chunk_context"]["discourse_markers"],
                    }
                )

                # Add quality and relationship metrics
                enhanced_metadata.update(
                    {
                        "quality_metrics": contextual_chunk_data["quality_metrics"],
                        "relationships": contextual_chunk_data["relationships"],
                        "context_inheritance": contextual_chunk_data["context_inheritance"],
                        "context_richness_score": self._calculate_context_richness(contextual_chunk_data),
                    }
                )

                # Create enhanced chunk
                contextual_chunk = Chunk(
                    id=base_chunk.id,
                    document_id=base_chunk.document_id,
                    text=base_chunk.text,
                    position=base_chunk.position,
                    start_idx=base_chunk.start_idx,
                    end_idx=base_chunk.end_idx,
                    metadata=enhanced_metadata,
                )

                contextual_chunks.append(contextual_chunk)

                # Update previous chunk context for next iteration
                previous_chunk_context = self.contextual_tagger.extract_chunk_context(
                    chunk_id=base_chunk.id,
                    chunk_text=base_chunk.text,
                    chunk_position=i,
                    start_char=base_chunk.start_idx,
                    end_char=base_chunk.end_idx,
                    document_context=document_context,
                    full_document_text=document.content,
                    previous_chunk_context=previous_chunk_context,
                )

            # Update statistics
            self._update_contextual_stats(document_context, contextual_chunks)

            logger.info(f"Created {len(contextual_chunks)} contextually-enhanced chunks for {document.id}")

            return contextual_chunks

        except Exception as e:
            logger.warning(f"Contextual chunking failed for document {document.id}: {e}")
            logger.info("Falling back to intelligent chunking")

            # Fall back to intelligent chunking
            return self.chunk_document_intelligently(document, force_traditional)

    def chunk_document(
        self,
        document: Document,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[Chunk]:
        """Enhanced chunk_document that uses contextual chunking by default.

        This overrides the parent method to provide contextual chunking
        while maintaining backward compatibility.
        """
        if self.enable_contextual_tagging:
            return self.chunk_document_with_context(document)
        # Use parent implementation
        return super().chunk_document(document, chunk_size, chunk_overlap)

    async def retrieve_with_contextual_analysis(
        self,
        query: str,
        k: int | None = None,
        use_cache: bool = True,
        domain_filter: ContentDomain | None = None,
        reading_level_filter: ReadingLevel | None = None,
        document_type_filter: DocumentType | None = None,
        chunk_type_filter: ChunkType | None = None,
        min_credibility: float = 0.0,
        min_quality: float = 0.0,
        require_entities: bool = False,
        context_similarity_boost: float = 0.1,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        """Enhanced retrieval with comprehensive contextual filtering and analysis.

        Args:
            query: Search query
            k: Number of results
            use_cache: Whether to use cache
            domain_filter: Filter by content domain
            reading_level_filter: Filter by reading level
            document_type_filter: Filter by document type
            chunk_type_filter: Filter by chunk type
            min_credibility: Minimum credibility score
            min_quality: Minimum quality score
            require_entities: Require chunks with entities
            context_similarity_boost: Boost for contextually similar content

        Returns:
            Enhanced retrieval results with contextual analysis
        """
        # Perform base retrieval
        results, metrics = await super().retrieve_with_content_analysis(
            query=query,
            k=k * 2 if self.enable_contextual_tagging else k,  # Get more for filtering
            use_cache=use_cache,
            include_entities=True,
        )

        if not self.enable_contextual_tagging or not results:
            return results[:k] if k else results, metrics

        # Apply contextual filtering
        contextual_results = []

        for result in results:
            metadata = result.metadata or {}

            # Domain filtering
            if domain_filter is not None:
                doc_domain = metadata.get("document_domain")
                if doc_domain != domain_filter.value:
                    continue

            # Reading level filtering
            if reading_level_filter is not None:
                reading_level = metadata.get("reading_level")
                if reading_level != reading_level_filter.value:
                    continue

            # Document type filtering
            if document_type_filter is not None:
                doc_type = metadata.get("document_type")
                if doc_type != document_type_filter.value:
                    continue

            # Chunk type filtering
            if chunk_type_filter is not None:
                chunk_type = metadata.get("chunk_context", {}).get("chunk_type")
                if chunk_type != chunk_type_filter.value:
                    continue

            # Credibility filtering
            credibility = metadata.get("credibility_score", 0.7)
            if credibility < min_credibility:
                continue

            # Quality filtering
            quality_score = metadata.get("quality_metrics", {}).get("overall_quality", 0.7)
            if quality_score < min_quality:
                continue

            # Entity requirement filtering
            if require_entities:
                entities = metadata.get("contextual_entities", [])
                if not entities:
                    continue

            # Calculate contextual similarity boost
            if context_similarity_boost > 0:
                contextual_boost = self._calculate_contextual_boost(query, metadata, context_similarity_boost)
                result.score += contextual_boost

            contextual_results.append(result)

        # Re-sort by enhanced scores
        contextual_results.sort(key=lambda x: x.score, reverse=True)

        # Limit to requested number
        if k:
            contextual_results = contextual_results[:k]

        # Update metrics
        metrics.update(
            {
                "contextual_filtering": True,
                "domain_filter": domain_filter.value if domain_filter else None,
                "reading_level_filter": (reading_level_filter.value if reading_level_filter else None),
                "document_type_filter": (document_type_filter.value if document_type_filter else None),
                "chunk_type_filter": (chunk_type_filter.value if chunk_type_filter else None),
                "min_credibility": min_credibility,
                "min_quality": min_quality,
                "results_after_contextual_filtering": len(contextual_results),
                "contextual_boost_applied": context_similarity_boost > 0,
            }
        )

        return contextual_results, metrics

    def _calculate_context_richness(self, contextual_chunk_data: dict[str, Any]) -> float:
        """Calculate richness score of contextual metadata."""
        richness_score = 0.0
        max_score = 0.0

        # Document context richness
        doc_context = contextual_chunk_data.get("document_context", {})
        if doc_context.get("executive_summary"):
            richness_score += 0.15
        if doc_context.get("key_themes"):
            richness_score += 0.10
        if doc_context.get("key_concepts"):
            richness_score += 0.10
        max_score += 0.35

        # Chunk context richness
        chunk_context = contextual_chunk_data.get("chunk_context", {})
        if chunk_context.get("local_summary"):
            richness_score += 0.15
        if chunk_context.get("key_entities"):
            richness_score += 0.10
        if chunk_context.get("section_hierarchy"):
            richness_score += 0.05
        if chunk_context.get("discourse_markers"):
            richness_score += 0.05
        max_score += 0.35

        # Quality metrics
        quality_metrics = contextual_chunk_data.get("quality_metrics", {})
        overall_quality = quality_metrics.get("overall_quality", 0.0)
        richness_score += overall_quality * 0.30
        max_score += 0.30

        return richness_score / max_score if max_score > 0 else 0.0

    def _calculate_contextual_boost(self, query: str, metadata: dict[str, Any], boost_factor: float) -> float:
        """Calculate contextual similarity boost for results."""
        boost = 0.0

        # Check query relevance to document themes
        doc_themes = metadata.get("document_themes", [])
        query_lower = query.lower()
        theme_matches = sum(1 for theme in doc_themes if theme.lower() in query_lower)
        if theme_matches > 0:
            boost += boost_factor * theme_matches

        # Check query relevance to local keywords
        local_keywords = metadata.get("local_keywords", [])
        keyword_matches = sum(1 for keyword in local_keywords if keyword.lower() in query_lower)
        if keyword_matches > 0:
            boost += boost_factor * 0.5 * keyword_matches

        # Check entity relevance
        entities = metadata.get("contextual_entities", [])
        entity_matches = sum(1 for entity in entities if entity.get("text", "").lower() in query_lower)
        if entity_matches > 0:
            boost += boost_factor * 0.3 * entity_matches

        return min(boost, boost_factor * 2)  # Cap the boost

    def _update_contextual_stats(self, document_context: DocumentContext, contextual_chunks: list[Chunk]) -> None:
        """Update contextual processing statistics."""
        self.contextual_stats["documents_with_context"] += 1
        self.contextual_stats["chunks_with_context"] += len(contextual_chunks)

        # Update domain distribution
        domain = document_context.domain.value
        self.contextual_stats["domain_classifications"][domain] = (
            self.contextual_stats["domain_classifications"].get(domain, 0) + 1
        )

        # Update reading level distribution
        reading_level = document_context.reading_level.value
        self.contextual_stats["reading_level_distribution"][reading_level] = (
            self.contextual_stats["reading_level_distribution"].get(reading_level, 0) + 1
        )

        # Update chunk type distribution
        for chunk in contextual_chunks:
            chunk_type = chunk.metadata.get("chunk_context", {}).get("chunk_type", "unknown")
            self.contextual_stats["chunk_type_distribution"][chunk_type] = (
                self.contextual_stats["chunk_type_distribution"].get(chunk_type, 0) + 1
            )

        # Calculate average context richness
        if contextual_chunks:
            total_richness = sum(chunk.metadata.get("context_richness_score", 0.0) for chunk in contextual_chunks)
            avg_richness = total_richness / len(contextual_chunks)

            total_docs = self.contextual_stats["documents_with_context"]
            current_avg = self.contextual_stats["avg_context_richness"]
            self.contextual_stats["avg_context_richness"] = (current_avg * (total_docs - 1) + avg_richness) / total_docs

    def get_contextual_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics including contextual features."""
        base_metrics = super().get_enhanced_performance_metrics()

        # Add contextual metrics
        contextual_metrics = {
            "contextual_tagging_enabled": self.enable_contextual_tagging,
            "contextual_statistics": self.contextual_stats.copy(),
            # Contextual quality metrics
            "contextual_quality": {
                "avg_context_richness": self.contextual_stats["avg_context_richness"],
                "documents_with_context": self.contextual_stats["documents_with_context"],
                "chunks_with_context": self.contextual_stats["chunks_with_context"],
                "context_coverage": (
                    self.contextual_stats["chunks_with_context"]
                    / max(
                        1,
                        self.chunking_stats["intelligent_chunks_created"]
                        + self.chunking_stats["traditional_chunks_created"],
                    )
                ),
            },
            # Content distribution
            "content_distribution": {
                "domains": self.contextual_stats["domain_classifications"],
                "reading_levels": self.contextual_stats["reading_level_distribution"],
                "chunk_types": self.contextual_stats["chunk_type_distribution"],
            },
        }

        return {**base_metrics, **contextual_metrics}

    def analyze_document_contextual_features(self, document_id: str) -> dict[str, Any]:
        """Analyze contextual features of a processed document."""
        if document_id not in self.document_contexts:
            return {"error": f"Document {document_id} not found or not processed with context"}

        document_context = self.document_contexts[document_id]

        return {
            "document_id": document_id,
            "contextual_analysis": {
                "document_type": document_context.document_type.value,
                "domain": document_context.domain.value,
                "reading_level": document_context.reading_level.value,
                "estimated_reading_time": document_context.estimated_reading_time,
                "credibility_score": document_context.source_credibility_score,
                "content_richness": {
                    "key_themes": document_context.key_themes,
                    "key_concepts": document_context.key_concepts,
                    "document_entities": len(document_context.document_entities),
                    "quality_indicators": document_context.quality_indicators,
                },
                "structure_analysis": {
                    "total_length": document_context.total_length,
                    "chapter_count": document_context.chapter_count,
                    "section_count": document_context.section_count,
                },
                "metadata": {
                    "author": document_context.author,
                    "publication_date": document_context.publication_date,
                    "target_audience": document_context.target_audience,
                    "language": document_context.language,
                },
            },
        }


# Test function
async def test_contextual_rag_pipeline() -> bool:
    """Test the contextual RAG pipeline with comprehensive documents."""
    print("Testing Contextual RAG Pipeline")
    print("=" * 60)

    # Initialize contextual RAG pipeline
    pipeline = ContextualRAGPipeline(
        enable_intelligent_chunking=True,
        enable_contextual_tagging=True,
        chunking_window_size=3,
        chunking_min_sentences=2,
        chunking_max_sentences=12,
        chunking_context_overlap=1,
    )

    # Create test documents with rich content
    test_documents = [
        Document(
            id="academic_ai_paper",
            title="Advances in Deep Learning Architectures: A Comprehensive Survey",
            content="""
            # Abstract

            This comprehensive survey examines recent advances in deep learning architectures, focusing on transformer models, convolutional neural networks, and recurrent architectures. Our analysis covers both theoretical foundations and practical applications across computer vision, natural language processing, and reinforcement learning domains.

            # Introduction

            Deep learning has revolutionized artificial intelligence research over the past decade. The introduction of transformer architectures by Vaswani et al. in 2017 marked a paradigm shift in sequence modeling, leading to breakthrough applications in language understanding and generation.

            ## Transformer Architectures

            The transformer architecture relies on self-attention mechanisms to capture long-range dependencies in sequential data. Key innovations include multi-head attention, positional encoding, and layer normalization techniques that enable effective training of very large models.

            For example, GPT-3 demonstrates remarkable few-shot learning capabilities across diverse language tasks, showcasing the power of scale in transformer models.

            ## Convolutional Neural Networks

            Despite the rise of transformers, CNNs remain crucial for computer vision applications. Recent advances include EfficientNet architectures that optimize for both accuracy and computational efficiency through compound scaling methods.

            # Conclusion

            The future of deep learning lies in hybrid architectures that combine the strengths of different neural network paradigms while addressing current limitations in computational efficiency and interpretability.
            """,
            source_type="research_paper",
            metadata={
                "author": "Dr. Sarah Chen",
                "publication_date": "2024-03-15",
                "journal": "Journal of AI Research",
                "credibility_score": 0.95,
                "target_audience": "academic",
            },
        ),
        Document(
            id="technical_guide",
            title="Building Scalable Web Applications: A Developer's Guide",
            content="""
            # Chapter 1: Architecture Fundamentals

            Building scalable web applications requires careful consideration of architectural patterns and design principles. This guide covers microservices, database design, and performance optimization strategies.

            ## Microservices Architecture

            Microservices decompose applications into small, independently deployable services. Each service handles a specific business function and communicates through well-defined APIs.

            ### Benefits and Challenges

            The primary benefits include scalability, technology diversity, and team autonomy. However, microservices introduce complexity in service coordination, data consistency, and monitoring.

            ## Database Design Patterns

            Database selection depends on application requirements. SQL databases provide ACID guarantees, while NoSQL solutions offer horizontal scalability and flexible schemas.

            ### Sharding Strategies

            Horizontal partitioning (sharding) distributes data across multiple database instances. Common strategies include range-based, hash-based, and directory-based sharding.

            # Chapter 2: Performance Optimization

            Performance optimization involves multiple layers: application code, database queries, caching strategies, and infrastructure scaling.

            ## Caching Strategies

            Implementing effective caching reduces database load and improves response times. Popular approaches include Redis for in-memory caching and CDNs for static content delivery.
            """,
            source_type="technical_doc",
            metadata={
                "author": "Engineering Team",
                "publication_date": "2024-01-20",
                "publisher": "TechBooks Publishing",
                "credibility_score": 0.85,
                "target_audience": "developers",
            },
        ),
    ]

    print(f"[PROCESS] Processing {len(test_documents)} documents with contextual tagging...")

    # Index documents with contextual tagging
    start_time = time.perf_counter()
    stats = pipeline.index_documents(test_documents)
    indexing_time = time.perf_counter() - start_time

    print(f"[SUCCESS] Indexing completed in {indexing_time:.2f}s:")
    print(f"  - Documents: {stats['documents_processed']}")
    print(f"  - Chunks: {stats['chunks_created']}")
    print(f"  - Contextual enhancement: {stats.get('contextual_tagging_enabled', False)}")

    # Analyze document contextual features
    print("\n[ANALYZE] Document Contextual Features:")
    for doc in test_documents:
        analysis = pipeline.analyze_document_contextual_features(doc.id)
        if "error" not in analysis:
            ctx = analysis["contextual_analysis"]
            print(f"\nDocument: {doc.id}")
            print(f"  Type: {ctx['document_type']}, Domain: {ctx['domain']}")
            print(f"  Reading Level: {ctx['reading_level']}")
            print(f"  Themes: {ctx['content_richness']['key_themes']}")
            print(f"  Concepts: {len(ctx['content_richness']['key_concepts'])} identified")
            print(f"  Quality Score: {ctx['credibility_score']:.2f}")

    # Test contextual retrieval
    test_queries = [
        {
            "query": "What are transformer architectures and how do they work?",
            "filters": {
                "domain_filter": ContentDomain.SCIENCE,
                "reading_level_filter": ReadingLevel.GRADUATE,
            },
        },
        {
            "query": "How do you implement microservices architecture?",
            "filters": {
                "domain_filter": ContentDomain.TECHNOLOGY,
                "min_credibility": 0.8,
            },
        },
        {
            "query": "What are the benefits of caching in web applications?",
            "filters": {"chunk_type_filter": ChunkType.BODY, "require_entities": False},
        },
        {
            "query": "How does sharding work in database design?",
            "filters": {"document_type_filter": DocumentType.TECHNICAL_DOC},
        },
    ]

    print("\n[TEST] Testing Contextual Retrieval:")
    print("-" * 60)

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        filters = test_case["filters"]

        print(f"\nQuery {i}: {query}")

        # Perform contextual retrieval
        start_time = time.perf_counter()
        results, metrics = await pipeline.retrieve_with_contextual_analysis(query=query, k=3, **filters)
        retrieval_time = (time.perf_counter() - start_time) * 1000

        print(f"  Latency: {retrieval_time:.1f}ms")
        print(f"  Results: {len(results)}")
        print(f"  Contextual Filtering: {metrics.get('contextual_filtering', False)}")
        print(f"  Applied Filters: {[k for k, v in filters.items() if v is not None]}")

        if results:
            best_result = results[0]
            metadata = best_result.metadata or {}

            print("  Best Match:")
            print(f"    Document: {best_result.document_id}")
            print(f"    Domain: {metadata.get('document_domain', 'unknown')}")
            print(f"    Reading Level: {metadata.get('reading_level', 'unknown')}")
            print(f"    Chunk Type: {metadata.get('chunk_context', {}).get('chunk_type', 'unknown')}")
            print(f"    Quality: {metadata.get('quality_metrics', {}).get('overall_quality', 0):.3f}")
            print(f"    Context Richness: {metadata.get('context_richness_score', 0):.3f}")
            print(f"    Text: {best_result.text[:150]}...")

    # Get comprehensive performance metrics
    perf_metrics = pipeline.get_contextual_performance_metrics()

    print(f"\n{'=' * 60}")
    print("Contextual Pipeline Performance Metrics")
    print("=" * 60)

    print("Contextual Features:")
    contextual_quality = perf_metrics["contextual_quality"]
    print(f"  - Context Richness: {contextual_quality['avg_context_richness']:.3f}")
    print(f"  - Context Coverage: {contextual_quality['context_coverage']:.2%}")
    print(f"  - Documents with Context: {contextual_quality['documents_with_context']}")

    print("\nContent Distribution:")
    content_dist = perf_metrics["content_distribution"]
    print(f"  - Domains: {content_dist['domains']}")
    print(f"  - Reading Levels: {content_dist['reading_levels']}")
    print(f"  - Chunk Types: {list(content_dist['chunk_types'].keys())}")

    print("\nPerformance:")
    print(f"  - Average Latency: {perf_metrics['avg_latency_ms']:.1f}ms")
    print(f"  - Cache Hit Rate: {perf_metrics['cache_metrics']['hit_rate']:.2%}")
    print(f"  - Index Size: {perf_metrics['index_size']} vectors")

    print("\n[ASSESSMENT] Overall Assessment:")
    if (
        perf_metrics["meets_target"]
        and contextual_quality["avg_context_richness"] > 0.6
        and contextual_quality["context_coverage"] > 0.8
    ):
        print("EXCELLENT: Contextual RAG pipeline fully operational!")
        print("  - Rich bilateral context (document <-> chunk)")
        print("  - Comprehensive metadata inheritance")
        print("  - Advanced contextual filtering")
        print("  - High-quality context extraction")
    else:
        print("GOOD: Contextual RAG pipeline functional with room for optimization")

    return True


if __name__ == "__main__":
    asyncio.run(test_contextual_rag_pipeline())
