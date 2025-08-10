"""
Enhanced CODEX RAG Pipeline with Intelligent Chunking.

Extends the base CODEX RAG pipeline with:
- Intelligent sliding window similarity-based chunking
- Idea-aware chunk boundaries with context preservation
- Content type detection and specialized handling
- Enhanced metadata and semantic coherence tracking
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from codex_rag_integration import CODEXRAGPipeline, Document, Chunk, RetrievalResult
from intelligent_chunking_simple import (
    SimpleIntelligentChunker as IntelligentChunker, 
    IntelligentChunk, 
    DocumentType, 
    ContentType
)

logger = logging.getLogger(__name__)


class EnhancedCODEXRAGPipeline(CODEXRAGPipeline):
    """
    Enhanced CODEX RAG pipeline with intelligent chunking capabilities.
    
    Maintains full CODEX compliance while adding:
    - Sliding window similarity analysis for chunking
    - Idea boundary detection
    - Content-aware processing
    - Enhanced semantic coherence
    """
    
    def __init__(
        self,
        enable_intelligent_chunking: bool = True,
        chunking_window_size: int = 3,
        chunking_min_sentences: int = 2,
        chunking_max_sentences: int = 15,
        chunking_context_overlap: int = 1,
        similarity_thresholds: Optional[Dict[DocumentType, float]] = None
    ):
        """
        Initialize enhanced pipeline.
        
        Args:
            enable_intelligent_chunking: Whether to use intelligent chunking
            chunking_window_size: Window size for similarity analysis
            chunking_min_sentences: Minimum sentences per chunk
            chunking_max_sentences: Maximum sentences per chunk
            chunking_context_overlap: Sentence overlap between chunks
            similarity_thresholds: Custom thresholds by document type
        """
        # Initialize base CODEX pipeline
        super().__init__()
        
        # Initialize intelligent chunker
        self.enable_intelligent_chunking = enable_intelligent_chunking
        
        if self.enable_intelligent_chunking:
            logger.info("Initializing intelligent chunker...")
            
            self.intelligent_chunker = IntelligentChunker(
                embedding_model="paraphrase-MiniLM-L3-v2",  # Use same model as RAG
                window_size=chunking_window_size,
                min_chunk_sentences=chunking_min_sentences,
                max_chunk_sentences=chunking_max_sentences,
                context_overlap=chunking_context_overlap,
                similarity_threshold=similarity_thresholds,
                # Note: Using simple chunker without spaCy dependency
            )
            
            logger.info("Intelligent chunker initialized successfully")
        else:
            self.intelligent_chunker = None
            logger.info("Using traditional word-based chunking")
            
        # Enhanced statistics
        self.chunking_stats = {
            "intelligent_chunks_created": 0,
            "traditional_chunks_created": 0,
            "avg_chunk_coherence": 0.0,
            "total_entities_extracted": 0,
            "content_type_distribution": {}
        }

    def chunk_document_intelligently(
        self, 
        document: Document,
        force_traditional: bool = False
    ) -> List[Chunk]:
        """
        Chunk document using intelligent or traditional method.
        
        Args:
            document: Document to chunk
            force_traditional: Force traditional chunking even if intelligent enabled
            
        Returns:
            List of chunks (CODEX-compliant format)
        """
        if not self.enable_intelligent_chunking or force_traditional:
            # Use traditional word-based chunking from parent class
            return self.chunk_document(document)
            
        try:
            # Infer document type for optimal chunking
            doc_type = self.intelligent_chunker.infer_document_type(document.content)
            logger.debug(f"Document type inferred as: {doc_type.value}")
            
            # Use intelligent chunking
            intelligent_chunks = self.intelligent_chunker.chunk_document(
                text=document.content,
                document_id=document.id,
                doc_type=doc_type
            )
            
            # Convert intelligent chunks to CODEX-compliant format
            codex_chunks = []
            
            for i, ichunk in enumerate(intelligent_chunks):
                # Create enhanced metadata combining original and intelligent chunking data
                enhanced_metadata = document.metadata.copy() if document.metadata else {}
                
                # Add intelligent chunking metadata
                enhanced_metadata.update({
                    # Intelligent chunking specific
                    "chunking_method": "intelligent_sliding_window",
                    "chunk_type": ichunk.content_type.value,
                    "topic_coherence": ichunk.topic_coherence,
                    "sentence_range": f"{ichunk.start_sentence_idx}-{ichunk.end_sentence_idx}",
                    "context_overlap": ichunk.context_overlap,
                    
                    # Content analysis
                    "entities": ichunk.entities or [],
                    "entity_count": len(ichunk.entities or []),
                    "summary": ichunk.summary,
                    "word_count": ichunk.word_count,
                    "sentence_count": len(ichunk.sentences),
                    
                    # Document context
                    "document_type": doc_type.value,
                    "chunk_position_ratio": i / len(intelligent_chunks) if len(intelligent_chunks) > 1 else 0.0
                })
                
                # Create CODEX-compliant chunk
                codex_chunk = Chunk(
                    id=ichunk.id,
                    document_id=document.id,
                    text=ichunk.text,
                    position=i,
                    start_idx=ichunk.start_sentence_idx,
                    end_idx=ichunk.end_sentence_idx,
                    metadata=enhanced_metadata
                )
                
                codex_chunks.append(codex_chunk)
                
            # Update statistics
            self.chunking_stats["intelligent_chunks_created"] += len(codex_chunks)
            
            if intelligent_chunks:
                avg_coherence = np.mean([ic.topic_coherence for ic in intelligent_chunks])
                total_entities = sum(len(ic.entities or []) for ic in intelligent_chunks)
                
                # Update running averages
                total_chunks = (self.chunking_stats["intelligent_chunks_created"] + 
                               self.chunking_stats["traditional_chunks_created"])
                
                if total_chunks > 0:
                    self.chunking_stats["avg_chunk_coherence"] = (
                        (self.chunking_stats["avg_chunk_coherence"] * (total_chunks - len(codex_chunks)) +
                         avg_coherence * len(codex_chunks)) / total_chunks
                    )
                    
                self.chunking_stats["total_entities_extracted"] += total_entities
                
                # Update content type distribution
                for ichunk in intelligent_chunks:
                    content_type = ichunk.content_type.value
                    self.chunking_stats["content_type_distribution"][content_type] = (
                        self.chunking_stats["content_type_distribution"].get(content_type, 0) + 1
                    )
                    
            logger.info(f"Created {len(codex_chunks)} intelligent chunks for document {document.id}")
            logger.debug(f"Average coherence: {avg_coherence:.3f}, Total entities: {total_entities}")
            
            return codex_chunks
            
        except Exception as e:
            logger.warning(f"Intelligent chunking failed for document {document.id}: {e}")
            logger.info("Falling back to traditional chunking")
            
            # Fall back to traditional chunking
            traditional_chunks = self.chunk_document(document)
            self.chunking_stats["traditional_chunks_created"] += len(traditional_chunks)
            
            return traditional_chunks

    def chunk_document(
        self,
        document: Document,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[Chunk]:
        """
        Enhanced chunk_document that uses intelligent chunking by default.
        
        This overrides the parent method to provide intelligent chunking
        while maintaining backward compatibility.
        """
        if self.enable_intelligent_chunking:
            return self.chunk_document_intelligently(document)
        else:
            # Use parent implementation for traditional chunking
            return super().chunk_document(document, chunk_size, chunk_overlap)

    def index_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Enhanced document indexing with intelligent chunking statistics.
        
        Args:
            documents: List of documents to index
            
        Returns:
            Enhanced indexing statistics
        """
        start_time = time.perf_counter()
        
        # Reset chunking stats for this indexing operation
        operation_stats = {
            "intelligent_chunks_created": 0,
            "traditional_chunks_created": 0,
            "content_types_found": set(),
            "avg_coherence_this_batch": 0.0,
            "entities_extracted_this_batch": 0
        }
        
        # Index documents using parent method (which will call our enhanced chunk_document)
        base_stats = super().index_documents(documents)
        
        # Calculate enhanced statistics
        total_chunks_this_batch = (
            self.chunking_stats["intelligent_chunks_created"] - 
            operation_stats["intelligent_chunks_created"] +
            self.chunking_stats["traditional_chunks_created"] -
            operation_stats["traditional_chunks_created"]
        )
        
        # Enhanced statistics
        enhanced_stats = {
            **base_stats,  # Include all base stats
            
            # Intelligent chunking stats
            "intelligent_chunking_enabled": self.enable_intelligent_chunking,
            "intelligent_chunks_created": self.chunking_stats["intelligent_chunks_created"],
            "traditional_chunks_created": self.chunking_stats["traditional_chunks_created"],
            "total_chunks_all_time": (
                self.chunking_stats["intelligent_chunks_created"] + 
                self.chunking_stats["traditional_chunks_created"]
            ),
            
            # Quality metrics
            "avg_chunk_coherence": self.chunking_stats["avg_chunk_coherence"],
            "total_entities_extracted": self.chunking_stats["total_entities_extracted"],
            "content_type_distribution": dict(self.chunking_stats["content_type_distribution"]),
            
            # This batch specific
            "chunks_this_batch": total_chunks_this_batch,
            "chunking_method": "intelligent" if self.enable_intelligent_chunking else "traditional"
        }
        
        logger.info(f"Enhanced indexing complete: {enhanced_stats}")
        
        return enhanced_stats

    async def retrieve_with_content_analysis(
        self,
        query: str,
        k: int = None,
        use_cache: bool = True,
        content_type_filter: Optional[ContentType] = None,
        min_coherence: float = 0.0,
        include_entities: bool = False
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Enhanced retrieval with content analysis and filtering.
        
        Args:
            query: Search query
            k: Number of results
            use_cache: Whether to use cache
            content_type_filter: Filter by content type
            min_coherence: Minimum topic coherence threshold
            include_entities: Whether to include entity information
            
        Returns:
            Enhanced retrieval results with content analysis
        """
        # Perform base retrieval
        results, metrics = await super().retrieve(query, k, use_cache)
        
        # Apply intelligent chunking filters if enabled
        if self.enable_intelligent_chunking and results:
            filtered_results = []
            
            for result in results:
                # Apply content type filter
                if content_type_filter is not None:
                    chunk_content_type = result.metadata.get("chunk_type", "text")
                    if chunk_content_type != content_type_filter.value:
                        continue
                        
                # Apply coherence filter
                coherence = result.metadata.get("topic_coherence", 1.0)
                if coherence < min_coherence:
                    continue
                    
                # Enhance result with entity information if requested
                if include_entities and "entities" in result.metadata:
                    result.metadata["entity_types"] = result.metadata["entities"]
                    result.metadata["entity_count"] = len(result.metadata.get("entities", []))
                    
                filtered_results.append(result)
                
            results = filtered_results
            
            # Update metrics
            metrics.update({
                "content_filtered": True,
                "content_type_filter": content_type_filter.value if content_type_filter else None,
                "min_coherence_filter": min_coherence,
                "results_after_filtering": len(results)
            })
            
        return results, metrics

    def get_enhanced_performance_metrics(self) -> Dict[str, Any]:
        """
        Get enhanced performance metrics including chunking statistics.
        
        Returns:
            Enhanced performance metrics
        """
        base_metrics = super().get_performance_metrics()
        
        # Add chunking-specific metrics
        chunking_metrics = {
            "chunking_method": "intelligent" if self.enable_intelligent_chunking else "traditional",
            "chunking_statistics": self.chunking_stats.copy(),
            
            # Chunking quality metrics
            "chunking_quality": {
                "avg_coherence": self.chunking_stats["avg_chunk_coherence"],
                "total_entities": self.chunking_stats["total_entities_extracted"],
                "content_diversity": len(self.chunking_stats["content_type_distribution"])
            }
        }
        
        return {
            **base_metrics,
            **chunking_metrics
        }

    def analyze_document_structure(self, document: Document) -> Dict[str, Any]:
        """
        Analyze document structure and chunking characteristics.
        
        Args:
            document: Document to analyze
            
        Returns:
            Structure analysis results
        """
        if not self.enable_intelligent_chunking:
            return {"error": "Intelligent chunking not enabled"}
            
        try:
            # Extract sentences for analysis
            sentences = self.intelligent_chunker.extract_sentences(document.content)
            
            # Infer document type
            doc_type = self.intelligent_chunker.infer_document_type(document.content)
            
            # Create sliding windows for boundary detection
            windows = self.intelligent_chunker.create_sliding_windows(sentences)
            similarities = self.intelligent_chunker.calculate_similarity_scores(windows)
            boundaries = self.intelligent_chunker.detect_idea_boundaries(similarities, doc_type)
            
            # Content type analysis
            content_types = {}
            for sentence in sentences[:10]:  # Sample first 10 sentences
                content_type = self.intelligent_chunker.detect_content_type(sentence)
                content_types[content_type.value] = content_types.get(content_type.value, 0) + 1
                
            return {
                "document_id": document.id,
                "document_type": doc_type.value,
                "total_sentences": len(sentences),
                "sliding_windows": len(windows),
                "detected_boundaries": len(boundaries),
                "boundary_positions": [b.sentence_idx for b in boundaries],
                "boundary_confidences": [b.confidence for b in boundaries],
                "avg_similarity": np.mean(similarities) if similarities else 0.0,
                "min_similarity": np.min(similarities) if similarities else 0.0,
                "content_type_distribution": content_types,
                "estimated_chunks": len(boundaries) + 1,
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Document structure analysis failed: {e}")
            return {"error": str(e)}


async def test_enhanced_rag():
    """Test the enhanced RAG pipeline with intelligent chunking."""
    
    print("Testing Enhanced CODEX RAG Pipeline with Intelligent Chunking")
    print("=" * 70)
    
    # Initialize enhanced pipeline
    pipeline = EnhancedCODEXRAGPipeline(
        enable_intelligent_chunking=True,
        chunking_window_size=3,
        chunking_min_sentences=2,
        chunking_max_sentences=12,
        chunking_context_overlap=1
    )
    
    # Test document with multiple topics
    test_document = Document(
        id="multi_topic_doc",
        title="Technology and Environment Overview",
        content="""
        Artificial intelligence has transformed many industries in recent years. Machine learning algorithms can now process vast amounts of data to identify complex patterns. Deep learning has been particularly successful in image recognition and natural language processing tasks. These technologies are being applied in healthcare, finance, and autonomous vehicles.
        
        Climate change represents a significant global challenge that requires immediate action. Rising temperatures are causing sea levels to rise and weather patterns to become more extreme. The primary cause is greenhouse gas emissions from human activities, particularly the burning of fossil fuels for energy production.
        
        Renewable energy technologies offer promising solutions to reduce carbon emissions. Solar power has become increasingly cost-effective, with efficiency improvements making it competitive with traditional energy sources. Wind energy is another rapidly growing sector that can provide clean electricity at scale.
        
        Quantum computing represents the next frontier in computational power. Unlike classical computers that use binary bits, quantum computers use qubits that can exist in multiple states simultaneously. This quantum superposition allows for exponentially faster processing of certain types of problems.
        
        The intersection of AI and quantum computing could revolutionize scientific research. Quantum machine learning algorithms might solve optimization problems that are intractable for classical computers. This could accelerate drug discovery, materials science, and cryptography research.
        """,
        source_type="educational",
        metadata={
            "author": "Technology Researcher",
            "publication_date": "2024-01-01",
            "categories": ["AI", "Climate", "Quantum"]
        }
    )
    
    print("Analyzing document structure...")
    structure_analysis = pipeline.analyze_document_structure(test_document)
    
    print(f"Document Type: {structure_analysis['document_type']}")
    print(f"Total Sentences: {structure_analysis['total_sentences']}")
    print(f"Detected Boundaries: {structure_analysis['detected_boundaries']}")
    print(f"Boundary Positions: {structure_analysis['boundary_positions']}")
    print(f"Estimated Chunks: {structure_analysis['estimated_chunks']}")
    print(f"Content Types: {structure_analysis['content_type_distribution']}")
    
    print(f"\nIndexing document with intelligent chunking...")
    
    # Index document
    stats = pipeline.index_documents([test_document])
    
    print(f"Indexing Results:")
    print(f"- Documents processed: {stats['documents_processed']}")
    print(f"- Chunks created: {stats['chunks_created']}")
    print(f"- Intelligent chunks: {stats['intelligent_chunks_created']}")
    print(f"- Average coherence: {stats['avg_chunk_coherence']:.3f}")
    print(f"- Entities extracted: {stats['total_entities_extracted']}")
    print(f"- Content types: {stats['content_type_distribution']}")
    
    # Test queries
    test_queries = [
        "What are the applications of artificial intelligence?",
        "How does climate change affect the environment?",
        "What are the benefits of renewable energy?",
        "How do quantum computers work?",
        "What is the relationship between AI and quantum computing?"
    ]
    
    print(f"\nTesting retrieval with content analysis...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 50)
        
        # Enhanced retrieval
        results, metrics = await pipeline.retrieve_with_content_analysis(
            query=query,
            k=3,
            include_entities=True,
            min_coherence=0.0
        )
        
        print(f"Results: {len(results)}, Latency: {metrics['latency_ms']:.1f}ms")
        
        if results:
            best_result = results[0]
            print(f"Best match:")
            print(f"- Chunk ID: {best_result.chunk_id}")
            print(f"- Content Type: {best_result.metadata.get('chunk_type', 'unknown')}")
            print(f"- Coherence: {best_result.metadata.get('topic_coherence', 0):.3f}")
            print(f"- Entities: {best_result.metadata.get('entities', [])}")
            print(f"- Summary: {best_result.metadata.get('summary', 'N/A')}")
            print(f"- Text: {best_result.text[:150]}...")
            
    # Get enhanced performance metrics
    perf_metrics = pipeline.get_enhanced_performance_metrics()
    
    print(f"\n{'='*70}")
    print("Enhanced Performance Metrics")
    print("=" * 70)
    
    print(f"Chunking Method: {perf_metrics['chunking_method']}")
    print(f"Average Latency: {perf_metrics['avg_latency_ms']:.1f}ms")
    print(f"Cache Hit Rate: {perf_metrics['cache_metrics']['hit_rate']:.2%}")
    print(f"Index Size: {perf_metrics['index_size']} vectors")
    
    chunking_quality = perf_metrics['chunking_quality']
    print(f"\nChunking Quality:")
    print(f"- Average Coherence: {chunking_quality['avg_coherence']:.3f}")
    print(f"- Total Entities: {chunking_quality['total_entities']}")
    print(f"- Content Diversity: {chunking_quality['content_diversity']} types")
    
    print(f"\nOverall Assessment:")
    if perf_metrics['meets_target'] and stats['avg_chunk_coherence'] > 0.6:
        print("✅ Enhanced RAG pipeline performing excellently!")
        print("- Intelligent chunking creates coherent semantic boundaries")
        print("- Performance meets <100ms latency targets") 
        print("- Content analysis provides rich metadata")
    else:
        print("⚠️ Enhanced RAG pipeline needs optimization")
        
    return True


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_enhanced_rag())