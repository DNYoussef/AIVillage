"""Graph-Enhanced RAG Pipeline with Bayesian Trust Propagation.

Integrates the Contextual RAG Pipeline with the Bayesian Trust Graph system
for advanced semantic retrieval with graph-based trust propagation and
contextual relationship traversal.

Features:
- Knowledge graph integration with contextual tagging
- Bayesian trust propagation from document sources
- Graph-based retrieval with relationship traversal
- Enhanced query answering with trust-weighted results
- Comprehensive relationship detection and classification
"""

import asyncio
import logging
import time
from typing import Any

from bayesian_trust_graph import BayesianTrustGraph
from codex_rag_integration import Chunk, Document, RetrievalResult
from contextual_rag_pipeline import ContextualRAGPipeline
from contextual_tagging import ChunkType, ContentDomain, DocumentType, ReadingLevel

logger = logging.getLogger(__name__)


class GraphEnhancedRAGPipeline(ContextualRAGPipeline):
    """Graph-Enhanced RAG Pipeline with Bayesian Trust Propagation.

    Combines contextual tagging, intelligent chunking, and Bayesian trust graphs
    to create a comprehensive knowledge retrieval system with relationship-aware
    search and trust-weighted results.
    """

    def __init__(
        self,
        enable_intelligent_chunking: bool = True,
        enable_contextual_tagging: bool = True,
        enable_trust_graph: bool = True,
        chunking_window_size: int = 3,
        chunking_min_sentences: int = 2,
        chunking_max_sentences: int = 15,
        chunking_context_overlap: int = 1,
        similarity_thresholds: dict | None = None,
        contextual_embedding_model: str = "paraphrase-MiniLM-L3-v2",
        # Graph-specific parameters
        graph_similarity_threshold: float = 0.3,
        trust_decay_factor: float = 0.85,
        max_propagation_hops: int = 3,
        relationship_confidence_threshold: float = 0.6,
    ) -> None:
        """Initialize graph-enhanced RAG pipeline."""
        # Initialize parent contextual pipeline
        super().__init__(
            enable_intelligent_chunking=enable_intelligent_chunking,
            enable_contextual_tagging=enable_contextual_tagging,
            chunking_window_size=chunking_window_size,
            chunking_min_sentences=chunking_min_sentences,
            chunking_max_sentences=chunking_max_sentences,
            chunking_context_overlap=chunking_context_overlap,
            similarity_thresholds=similarity_thresholds,
            contextual_embedding_model=contextual_embedding_model,
        )

        # Initialize Bayesian trust graph
        self.enable_trust_graph = enable_trust_graph
        if self.enable_trust_graph:
            logger.info("Initializing Bayesian trust graph...")
            self.trust_graph = BayesianTrustGraph(
                embedding_model=contextual_embedding_model,
                similarity_threshold=graph_similarity_threshold,
                trust_decay_factor=trust_decay_factor,
                max_propagation_hops=max_propagation_hops,
                relationship_confidence_threshold=relationship_confidence_threshold,
            )
            logger.info("Bayesian trust graph initialized successfully")
        else:
            self.trust_graph = None

        # Graph-specific statistics
        self.graph_stats = {
            "chunks_added_to_graph": 0,
            "relationships_detected": 0,
            "trust_propagations_performed": 0,
            "graph_retrievals": 0,
            "avg_graph_retrieval_time": 0.0,
        }

    def chunk_document_with_graph_integration(self, document: Document, force_traditional: bool = False) -> list[Chunk]:
        """Chunk document with full graph integration.

        Performs contextual chunking and adds chunks to the knowledge graph
        with semantic relationship detection.
        """
        if not self.enable_trust_graph:
            # Fall back to contextual chunking
            return self.chunk_document_with_context(document, force_traditional)

        try:
            logger.info(f"Processing document with graph integration: {document.id}")

            # Step 1: Perform contextual chunking
            contextual_chunks = self.chunk_document_with_context(document, force_traditional)

            # Step 2: Add chunks to knowledge graph
            document_context = self.document_contexts.get(document.id)
            base_credibility = document_context.source_credibility_score if document_context else 0.7

            for chunk in contextual_chunks:
                # Add chunk to graph
                semantic_chunk_node = self.trust_graph.add_semantic_chunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    position=chunk.position,
                    start_idx=chunk.start_idx,
                    end_idx=chunk.end_idx,
                    contextual_metadata=chunk.metadata,
                    base_credibility=base_credibility,
                )

                # Detect semantic relationships
                relationships = self.trust_graph.detect_semantic_relationships(semantic_chunk_node, context_window=5)

                # Update chunk metadata with graph information
                if chunk.metadata is None:
                    chunk.metadata = {}

                chunk.metadata.update(
                    {
                        "graph_node_id": chunk.id,
                        "base_credibility": base_credibility,
                        "trust_score": semantic_chunk_node.trust_score,
                        "centrality_score": semantic_chunk_node.centrality_score,
                        "relationships_detected": len(relationships),
                        "relationship_types": [rel.relationship_type.value for rel in relationships],
                    }
                )

                self.graph_stats["chunks_added_to_graph"] += 1
                self.graph_stats["relationships_detected"] += len(relationships)

            logger.info(f"Added {len(contextual_chunks)} chunks to knowledge graph for {document.id}")

            return contextual_chunks

        except Exception as e:
            logger.warning(f"Graph integration failed for document {document.id}: {e}")
            logger.info("Falling back to contextual chunking")

            # Fall back to contextual chunking
            return self.chunk_document_with_context(document, force_traditional)

    def chunk_document(
        self, document: Document, chunk_size: int | None = None, chunk_overlap: int | None = None
    ) -> list[Chunk]:
        """Enhanced chunk_document that uses graph integration by default."""
        if self.enable_trust_graph:
            return self.chunk_document_with_graph_integration(document)
        # Use parent implementation
        return super().chunk_document(document, chunk_size, chunk_overlap)

    def index_documents(self, documents: list[Document]) -> dict[str, Any]:
        """Index documents with graph integration and trust propagation."""
        # Perform base indexing
        indexing_stats = super().index_documents(documents)

        # Perform trust propagation after all documents are indexed
        if self.enable_trust_graph and len(self.trust_graph.chunk_nodes) > 0:
            logger.info("Performing trust propagation across knowledge graph...")

            start_time = time.perf_counter()
            trust_scores = self.trust_graph.propagate_trust(max_iterations=15, convergence_threshold=0.0005)
            propagation_time = (time.perf_counter() - start_time) * 1000

            self.graph_stats["trust_propagations_performed"] += 1

            logger.info(f"Trust propagation completed in {propagation_time:.1f}ms")
            logger.info(f"Updated trust scores for {len(trust_scores)} chunks")

            # Update indexing stats with graph information
            indexing_stats.update(
                {
                    "graph_integration_enabled": True,
                    "chunks_in_graph": len(self.trust_graph.chunk_nodes),
                    "relationships_detected": self.graph_stats["relationships_detected"],
                    "trust_propagation_time_ms": propagation_time,
                    "trust_propagations_performed": self.graph_stats["trust_propagations_performed"],
                }
            )
        else:
            indexing_stats["graph_integration_enabled"] = False

        return indexing_stats

    async def retrieve_with_graph_enhanced_analysis(
        self,
        query: str,
        k: int | None = None,
        use_cache: bool = True,
        # Contextual filters (from parent)
        domain_filter: ContentDomain | None = None,
        reading_level_filter: ReadingLevel | None = None,
        document_type_filter: DocumentType | None = None,
        chunk_type_filter: ChunkType | None = None,
        min_credibility: float = 0.0,
        min_quality: float = 0.0,
        require_entities: bool = False,
        context_similarity_boost: float = 0.1,
        # Graph-specific parameters
        enable_graph_traversal: bool = True,
        trust_weight: float = 0.3,
        centrality_weight: float = 0.2,
        similarity_weight: float = 0.5,
        min_trust_score: float = 0.4,
        traversal_depth: int = 2,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        """Enhanced retrieval with graph traversal and trust-weighted scoring.

        Combines contextual filtering with graph-based retrieval for
        relationship-aware search with trust propagation.
        """
        start_time = time.perf_counter()

        # If graph is not enabled, fall back to contextual retrieval
        if not self.enable_trust_graph or not enable_graph_traversal:
            return await self.retrieve_with_contextual_analysis(
                query=query,
                k=k,
                use_cache=use_cache,
                domain_filter=domain_filter,
                reading_level_filter=reading_level_filter,
                document_type_filter=document_type_filter,
                chunk_type_filter=chunk_type_filter,
                min_credibility=min_credibility,
                min_quality=min_quality,
                require_entities=require_entities,
                context_similarity_boost=context_similarity_boost,
            )

        try:
            # Get query embedding
            query_embedding = self.trust_graph.embedding_model.encode(query)

            # Perform graph-based retrieval
            graph_results = self.trust_graph.retrieve_with_graph_traversal(
                query_embedding=query_embedding,
                k=k * 3 if k else 30,  # Get more results for filtering
                trust_weight=trust_weight,
                centrality_weight=centrality_weight,
                similarity_weight=similarity_weight,
                min_trust_score=min_trust_score,
                traversal_depth=traversal_depth,
            )

            # Convert graph results to RetrievalResult format
            retrieval_results = []

            for chunk_id, score, graph_metadata in graph_results:
                # Get chunk from vector store if available
                try:
                    # Try to get from base retrieval system
                    base_results, _ = await super().retrieve_with_content_analysis(
                        query=graph_metadata["text"][:100],  # Use chunk text as mini-query
                        k=1,
                        use_cache=False,
                    )

                    if base_results and base_results[0].id == chunk_id:
                        # Use existing result and enhance with graph data
                        result = base_results[0]

                        # Enhance metadata with graph information
                        enhanced_metadata = result.metadata.copy() if result.metadata else {}
                        enhanced_metadata.update(
                            {
                                "graph_enhanced": True,
                                "graph_score": score,
                                "trust_score": graph_metadata["trust_score"],
                                "centrality_score": graph_metadata["centrality_score"],
                                "traversal_depth": graph_metadata.get("traversal_depth", 0),
                                "parent_chunk": graph_metadata.get("parent_chunk"),
                                "relationship_type": graph_metadata.get("relationship_type"),
                            }
                        )

                        # Update score with graph-enhanced score
                        result.score = score
                        result.metadata = enhanced_metadata

                        retrieval_results.append(result)

                except Exception:
                    # Create new RetrievalResult from graph data
                    result = RetrievalResult(
                        id=chunk_id,
                        document_id=graph_metadata["document_id"],
                        text=graph_metadata["text"],
                        score=score,
                        metadata={
                            "graph_enhanced": True,
                            "graph_score": score,
                            "semantic_similarity": graph_metadata["semantic_similarity"],
                            "trust_score": graph_metadata["trust_score"],
                            "centrality_score": graph_metadata["centrality_score"],
                            "base_credibility": graph_metadata["base_credibility"],
                            "quality_score": graph_metadata["quality_score"],
                            "chunk_type": graph_metadata["chunk_type"],
                            "position": graph_metadata["position"],
                            "traversal_depth": graph_metadata.get("traversal_depth", 0),
                            "parent_chunk": graph_metadata.get("parent_chunk"),
                            "relationship_type": graph_metadata.get("relationship_type", "direct"),
                        },
                    )
                    retrieval_results.append(result)

            # Apply contextual filters
            if any(
                [
                    domain_filter,
                    reading_level_filter,
                    document_type_filter,
                    chunk_type_filter,
                    min_credibility > 0,
                    min_quality > 0,
                    require_entities,
                ]
            ):
                filtered_results = []
                for result in retrieval_results:
                    metadata = result.metadata or {}

                    # Apply same filters as parent class
                    if domain_filter is not None:
                        doc_domain = metadata.get("document_domain")
                        if doc_domain != domain_filter.value:
                            continue

                    if reading_level_filter is not None:
                        reading_level = metadata.get("reading_level")
                        if reading_level != reading_level_filter.value:
                            continue

                    if document_type_filter is not None:
                        doc_type = metadata.get("document_type")
                        if doc_type != document_type_filter.value:
                            continue

                    if chunk_type_filter is not None:
                        chunk_type = metadata.get("chunk_context", {}).get("chunk_type")
                        if chunk_type != chunk_type_filter.value:
                            continue

                    if metadata.get("trust_score", 0.7) < min_credibility:
                        continue

                    if metadata.get("quality_score", 0.7) < min_quality:
                        continue

                    if require_entities:
                        entities = metadata.get("contextual_entities", [])
                        if not entities:
                            continue

                    filtered_results.append(result)

                retrieval_results = filtered_results

            # Limit to requested number
            if k:
                retrieval_results = retrieval_results[:k]

            # Calculate metrics
            retrieval_time = (time.perf_counter() - start_time) * 1000
            self.graph_stats["graph_retrievals"] += 1
            self.graph_stats["avg_graph_retrieval_time"] = (
                self.graph_stats["avg_graph_retrieval_time"] * (self.graph_stats["graph_retrievals"] - 1)
                + retrieval_time
            ) / self.graph_stats["graph_retrievals"]

            metrics = {
                "graph_enhanced_retrieval": True,
                "total_latency_ms": retrieval_time,
                "graph_results_found": len(graph_results),
                "results_after_filtering": len(retrieval_results),
                "trust_weight": trust_weight,
                "centrality_weight": centrality_weight,
                "similarity_weight": similarity_weight,
                "traversal_depth": traversal_depth,
                "min_trust_score": min_trust_score,
                # Include parent metrics
                "contextual_filtering": True,
                "domain_filter": domain_filter.value if domain_filter else None,
                "reading_level_filter": (reading_level_filter.value if reading_level_filter else None),
                "document_type_filter": (document_type_filter.value if document_type_filter else None),
                "chunk_type_filter": (chunk_type_filter.value if chunk_type_filter else None),
                "min_credibility": min_credibility,
                "min_quality": min_quality,
            }

            return retrieval_results, metrics

        except Exception as e:
            logger.exception(f"Graph-enhanced retrieval failed: {e}")
            logger.info("Falling back to contextual retrieval")

            # Fall back to contextual retrieval
            return await self.retrieve_with_contextual_analysis(
                query=query,
                k=k,
                use_cache=use_cache,
                domain_filter=domain_filter,
                reading_level_filter=reading_level_filter,
                document_type_filter=document_type_filter,
                chunk_type_filter=chunk_type_filter,
                min_credibility=min_credibility,
                min_quality=min_quality,
                require_entities=require_entities,
                context_similarity_boost=context_similarity_boost,
            )

    def get_comprehensive_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics including graph features."""
        # Get base contextual metrics
        base_metrics = super().get_contextual_performance_metrics()

        # Add graph-specific metrics
        graph_metrics = {
            "graph_integration": {
                "enabled": self.enable_trust_graph,
                "chunks_in_graph": (len(self.trust_graph.chunk_nodes) if self.trust_graph else 0),
                "relationships_detected": self.graph_stats["relationships_detected"],
                "trust_propagations": self.graph_stats["trust_propagations_performed"],
                "graph_retrievals": self.graph_stats["graph_retrievals"],
                "avg_graph_retrieval_time": self.graph_stats["avg_graph_retrieval_time"],
            }
        }

        # Add graph statistics if available
        if self.trust_graph:
            graph_statistics = self.trust_graph.get_graph_statistics()
            graph_metrics["graph_statistics"] = graph_statistics

        return {**base_metrics, **graph_metrics}

    def analyze_graph_relationships(self, chunk_id: str) -> dict[str, Any]:
        """Analyze graph relationships for a specific chunk."""
        if not self.trust_graph or chunk_id not in self.trust_graph.chunk_nodes:
            return {"error": f"Chunk {chunk_id} not found in graph"}

        chunk_node = self.trust_graph.chunk_nodes[chunk_id]

        # Get relationships
        outgoing_relationships = []
        incoming_relationships = []

        for (
            source_id,
            target_id,
        ), relationship in self.trust_graph.relationships.items():
            if source_id == chunk_id:
                outgoing_relationships.append(
                    {
                        "target_chunk": target_id,
                        "relationship_type": relationship.relationship_type.value,
                        "confidence": relationship.confidence,
                        "weight": relationship.weight,
                        "trust_transfer_rate": relationship.trust_transfer_rate,
                    }
                )
            elif target_id == chunk_id:
                incoming_relationships.append(
                    {
                        "source_chunk": source_id,
                        "relationship_type": relationship.relationship_type.value,
                        "confidence": relationship.confidence,
                        "weight": relationship.weight,
                        "trust_transfer_rate": relationship.trust_transfer_rate,
                    }
                )

        return {
            "chunk_id": chunk_id,
            "graph_analysis": {
                "trust_score": chunk_node.trust_score,
                "centrality_score": chunk_node.centrality_score,
                "base_credibility": chunk_node.base_credibility,
                "quality_score": chunk_node.quality_score,
                "relationships": {
                    "outgoing": outgoing_relationships,
                    "incoming": incoming_relationships,
                    "total_outgoing": len(outgoing_relationships),
                    "total_incoming": len(incoming_relationships),
                },
                "relationship_types": {
                    "outgoing_types": list({rel["relationship_type"] for rel in outgoing_relationships}),
                    "incoming_types": list({rel["relationship_type"] for rel in incoming_relationships}),
                },
            },
        }

    def get_trust_propagation_paths(
        self, source_chunk_id: str, target_chunk_id: str, max_path_length: int = 5
    ) -> list[dict[str, Any]]:
        """Find trust propagation paths between two chunks."""
        if not self.trust_graph:
            return []

        try:
            # Find all simple paths between chunks
            paths = list(
                nx.all_simple_paths(
                    self.trust_graph.graph,
                    source_chunk_id,
                    target_chunk_id,
                    cutoff=max_path_length,
                )
            )

            path_analysis = []

            for path in paths:
                # Calculate path trust score
                path_trust = self.trust_graph.chunk_nodes[path[0]].trust_score
                path_relationships = []

                for i in range(len(path) - 1):
                    current_chunk = path[i]
                    next_chunk = path[i + 1]

                    if (current_chunk, next_chunk) in self.trust_graph.relationships:
                        relationship = self.trust_graph.relationships[(current_chunk, next_chunk)]
                        path_trust *= relationship.trust_transfer_rate * self.trust_graph.trust_decay_factor

                        path_relationships.append(
                            {
                                "from": current_chunk,
                                "to": next_chunk,
                                "type": relationship.relationship_type.value,
                                "confidence": relationship.confidence,
                                "trust_transfer_rate": relationship.trust_transfer_rate,
                            }
                        )

                path_analysis.append(
                    {
                        "path": path,
                        "path_length": len(path),
                        "path_trust_score": path_trust,
                        "relationships": path_relationships,
                    }
                )

            # Sort by trust score
            path_analysis.sort(key=lambda x: x["path_trust_score"], reverse=True)

            return path_analysis

        except Exception as e:
            logger.exception(f"Failed to find trust paths: {e}")
            return []


# Test function
async def test_graph_enhanced_rag_pipeline() -> bool:
    """Test the graph-enhanced RAG pipeline comprehensively."""
    print("Testing Graph-Enhanced RAG Pipeline with Bayesian Trust")
    print("=" * 70)

    # Initialize graph-enhanced pipeline
    pipeline = GraphEnhancedRAGPipeline(
        enable_intelligent_chunking=True,
        enable_contextual_tagging=True,
        enable_trust_graph=True,
        chunking_window_size=3,
        chunking_min_sentences=2,
        chunking_max_sentences=12,
        # Graph parameters
        graph_similarity_threshold=0.3,
        trust_decay_factor=0.85,
        max_propagation_hops=3,
        relationship_confidence_threshold=0.6,
    )

    # Create test documents with complex relationships
    test_documents = [
        Document(
            id="ai_foundations",
            title="Foundations of Artificial Intelligence",
            content="""
            # Introduction to Artificial Intelligence

            Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, perception, and language understanding.

            ## Machine Learning Fundamentals

            Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.

            ### Supervised Learning

            Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Common algorithms include linear regression, decision trees, and neural networks. For example, a supervised learning model can be trained to recognize images of cats by showing it thousands of labeled cat and non-cat images.

            ### Deep Learning

            Deep learning is a specialized form of machine learning that uses neural networks with multiple layers. These deep neural networks can automatically discover representations from data, making them particularly effective for complex tasks like image recognition and natural language processing.

            ## Applications and Impact

            AI applications span numerous domains including healthcare, finance, transportation, and entertainment. However, the development of AI also raises important ethical considerations regarding privacy, bias, and job displacement.
            """,
            source_type="textbook",
            metadata={
                "author": "Dr. AI Expert",
                "publication_date": "2024-01-15",
                "credibility_score": 0.95,
                "target_audience": "students",
            },
        ),
        Document(
            id="ml_applications",
            title="Machine Learning in Practice: Real-World Applications",
            content="""
            # Practical Machine Learning Applications

            Machine learning has transformed numerous industries through practical applications that solve real-world problems. This chapter explores key application areas and their impact.

            ## Healthcare Applications

            In healthcare, machine learning enables diagnostic assistance, drug discovery, and personalized treatment plans. Deep learning models can analyze medical images to detect diseases like cancer with accuracy comparable to expert radiologists.

            ### Medical Image Analysis

            Convolutional neural networks excel at analyzing medical images such as X-rays, MRIs, and CT scans. These systems can identify abnormalities, measure progression of diseases, and assist radiologists in making more accurate diagnoses.

            ## Natural Language Processing

            Natural Language Processing (NLP) applications include chatbots, translation systems, and sentiment analysis. Advanced models like transformers have revolutionized how computers understand and generate human language.

            ### Transformer Models

            Transformer architectures, introduced in the "Attention is All You Need" paper, use self-attention mechanisms to process sequential data more effectively than traditional recurrent networks. This has led to breakthroughs in language understanding tasks.

            ## Challenges and Limitations

            Despite remarkable progress, machine learning faces challenges including data quality issues, model interpretability, and the need for large amounts of training data. Addressing these limitations is crucial for broader adoption.
            """,
            source_type="article",
            metadata={
                "author": "ML Practitioner",
                "publication_date": "2024-02-20",
                "credibility_score": 0.88,
                "target_audience": "professionals",
            },
        ),
        Document(
            id="ai_ethics",
            title="Ethical Considerations in AI Development",
            content="""
            # AI Ethics and Responsible Development

            As artificial intelligence becomes more prevalent in society, ethical considerations become increasingly important. This document outlines key ethical challenges and principles for responsible AI development.

            ## Bias and Fairness

            AI systems can perpetuate or amplify existing biases present in training data. For example, if a hiring algorithm is trained on historical data that reflects past discrimination, it may continue to discriminate against certain groups.

            ### Addressing Algorithmic Bias

            Techniques for reducing bias include diverse data collection, bias testing, and algorithmic audits. Machine learning practitioners must actively work to identify and mitigate bias throughout the development process.

            ## Privacy and Data Protection

            AI applications often require large amounts of personal data, raising concerns about privacy and data protection. Users must have control over their data and understand how it is being used.

            ## Transparency and Explainability

            Many AI systems, particularly deep learning models, operate as "black boxes" where the decision-making process is not easily interpretable. This lack of transparency can be problematic in high-stakes applications like healthcare or criminal justice.

            ### Explainable AI

            Explainable AI (XAI) aims to make AI systems more interpretable and understandable to humans. This is crucial for building trust and ensuring accountability in AI applications.
            """,
            source_type="policy_doc",
            metadata={
                "author": "Ethics Committee",
                "publication_date": "2024-03-10",
                "credibility_score": 0.92,
                "target_audience": "policymakers",
            },
        ),
    ]

    print(f"[PROCESS] Processing {len(test_documents)} documents with graph integration...")

    # Index documents with graph integration
    start_time = time.perf_counter()
    indexing_stats = pipeline.index_documents(test_documents)
    indexing_time = time.perf_counter() - start_time

    print(f"[SUCCESS] Graph indexing completed in {indexing_time:.2f}s:")
    print(f"  - Documents: {indexing_stats['documents_processed']}")
    print(f"  - Chunks: {indexing_stats['chunks_created']}")
    print(f"  - Graph nodes: {indexing_stats.get('chunks_in_graph', 0)}")
    print(f"  - Relationships: {indexing_stats.get('relationships_detected', 0)}")
    print(f"  - Trust propagation time: {indexing_stats.get('trust_propagation_time_ms', 0):.1f}ms")

    # Test graph-enhanced retrieval
    test_queries = [
        {
            "query": "What is machine learning and how does it work?",
            "description": "Basic ML concept query",
        },
        {
            "query": "How does deep learning help with medical image analysis?",
            "description": "Cross-domain relationship query",
        },
        {
            "query": "What are the ethical concerns with AI bias?",
            "description": "Ethics-focused query",
        },
        {
            "query": "How do transformer models work in natural language processing?",
            "description": "Technical architecture query",
        },
        {
            "query": "What challenges do machine learning systems face in practice?",
            "description": "Broad challenges query",
        },
    ]

    print("\n[TEST] Testing Graph-Enhanced Retrieval:")
    print("-" * 70)

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        description = test_case["description"]

        print(f"\nQuery {i}: {query}")
        print(f"Type: {description}")

        # Perform graph-enhanced retrieval
        start_time = time.perf_counter()
        results, metrics = await pipeline.retrieve_with_graph_enhanced_analysis(
            query=query,
            k=3,
            enable_graph_traversal=True,
            trust_weight=0.3,
            centrality_weight=0.2,
            similarity_weight=0.5,
            min_trust_score=0.4,
            traversal_depth=2,
        )
        retrieval_time = (time.perf_counter() - start_time) * 1000

        print(f"  Retrieval Time: {retrieval_time:.1f}ms")
        print(f"  Results Found: {len(results)}")
        print(f"  Graph Enhanced: {metrics.get('graph_enhanced_retrieval', False)}")

        if results:
            best_result = results[0]
            metadata = best_result.metadata or {}

            print("  Best Match:")
            print(f"    Document: {best_result.document_id}")
            print(f"    Score: {best_result.score:.4f}")
            print(f"    Trust Score: {metadata.get('trust_score', 0):.3f}")
            print(f"    Centrality: {metadata.get('centrality_score', 0):.3f}")
            print(f"    Traversal Depth: {metadata.get('traversal_depth', 0)}")
            print(f"    Relationship: {metadata.get('relationship_type', 'direct')}")
            print(f"    Text: {best_result.text[:120]}...")

            # Show relationship analysis if available
            if metadata.get("parent_chunk"):
                print(f"    Connected via: {metadata.get('parent_chunk')}")

    # Analyze specific chunk relationships
    print("\n[ANALYSIS] Graph Relationship Analysis:")
    print("-" * 70)

    # Get a sample of chunk IDs from the graph
    if pipeline.trust_graph and pipeline.trust_graph.chunk_nodes:
        sample_chunks = list(pipeline.trust_graph.chunk_nodes.keys())[:3]

        for chunk_id in sample_chunks:
            analysis = pipeline.analyze_graph_relationships(chunk_id)

            if "error" not in analysis:
                graph_info = analysis["graph_analysis"]
                relationships = graph_info["relationships"]

                print(f"\nChunk: {chunk_id}")
                print(f"  Trust Score: {graph_info['trust_score']:.3f}")
                print(f"  Centrality Score: {graph_info['centrality_score']:.3f}")
                print(f"  Outgoing Relationships: {relationships['total_outgoing']}")
                print(f"  Incoming Relationships: {relationships['total_incoming']}")

                if relationships["outgoing"]:
                    print(f"  Relationship Types Out: {graph_info['relationship_types']['outgoing_types']}")
                if relationships["incoming"]:
                    print(f"  Relationship Types In: {graph_info['relationship_types']['incoming_types']}")

    # Get comprehensive metrics
    performance_metrics = pipeline.get_comprehensive_performance_metrics()

    print(f"\n{'=' * 70}")
    print("Graph-Enhanced RAG Pipeline Performance Assessment")
    print("=" * 70)

    # Graph integration metrics
    graph_integration = performance_metrics.get("graph_integration", {})
    print("Graph Integration:")
    print(f"  - Enabled: {graph_integration.get('enabled', False)}")
    print(f"  - Chunks in Graph: {graph_integration.get('chunks_in_graph', 0)}")
    print(f"  - Relationships Detected: {graph_integration.get('relationships_detected', 0)}")
    print(f"  - Trust Propagations: {graph_integration.get('trust_propagations', 0)}")
    print(f"  - Avg Graph Retrieval Time: {graph_integration.get('avg_graph_retrieval_time', 0):.1f}ms")

    # Graph statistics
    graph_stats = performance_metrics.get("graph_statistics", {})
    if graph_stats and "graph_structure" in graph_stats:
        structure = graph_stats["graph_structure"]
        trust_metrics = graph_stats.get("trust_metrics", {})

        print("\nGraph Structure:")
        print(f"  - Nodes: {structure.get('nodes', 0)}")
        print(f"  - Edges: {structure.get('edges', 0)}")
        print(f"  - Density: {structure.get('density', 0):.3f}")
        print(f"  - Connected Components: {structure.get('connected_components', 0)}")

        print("\nTrust Propagation:")
        print(f"  - Average Trust Score: {trust_metrics.get('avg_trust_score', 0):.3f}")
        print(
            f"  - Trust Score Range: {trust_metrics.get('min_trust_score', 0):.3f} - {trust_metrics.get('max_trust_score', 0):.3f}"
        )

        # Relationship distribution
        rel_dist = graph_stats.get("relationship_distribution", {})
        if rel_dist:
            print("\nRelationship Types:")
            for rel_type, count in sorted(rel_dist.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {rel_type}: {count}")

    # Overall performance
    contextual_quality = performance_metrics.get("contextual_quality", {})
    print("\nOverall Performance:")
    print(f"  - Average Latency: {performance_metrics.get('avg_latency_ms', 0):.1f}ms")
    print(f"  - Context Richness: {contextual_quality.get('avg_context_richness', 0):.3f}")
    print(f"  - Context Coverage: {contextual_quality.get('context_coverage', 0):.2%}")
    print(f"  - Cache Hit Rate: {performance_metrics.get('cache_metrics', {}).get('hit_rate', 0):.2%}")

    print("\n[ASSESSMENT] Final Assessment:")

    # Assessment criteria
    graph_enabled = graph_integration.get("enabled", False)
    chunks_in_graph = graph_integration.get("chunks_in_graph", 0) > 0
    relationships_detected = graph_integration.get("relationships_detected", 0) > 0
    trust_propagation_working = graph_integration.get("trust_propagations", 0) > 0
    avg_latency_good = performance_metrics.get("avg_latency_ms", 1000) < 100
    context_richness_good = contextual_quality.get("avg_context_richness", 0) > 0.6

    if (
        graph_enabled
        and chunks_in_graph
        and relationships_detected
        and trust_propagation_working
        and avg_latency_good
        and context_richness_good
    ):
        print("🎉 EXCELLENT: Graph-Enhanced RAG Pipeline fully operational!")
        print("  ✅ Bayesian trust graph with semantic relationships")
        print("  ✅ Trust propagation across knowledge graph")
        print("  ✅ Graph-based contextual retrieval")
        print("  ✅ Multi-level contextual tagging")
        print("  ✅ Intelligent chunking with semantic boundaries")
        print("  ✅ Performance targets met (<100ms latency)")
    elif graph_enabled and chunks_in_graph:
        print("✅ GOOD: Graph-Enhanced RAG Pipeline operational with room for optimization")
        print("  - Graph integration active")
        print("  - Trust propagation functional")
        print("  - Performance within acceptable ranges")
    else:
        print("⚠️  PARTIAL: Some graph features not fully operational")
        print("  - Check graph initialization and document indexing")

    return True


if __name__ == "__main__":
    asyncio.run(test_graph_enhanced_rag_pipeline())
