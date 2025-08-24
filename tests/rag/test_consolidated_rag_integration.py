"""
Comprehensive Integration Tests for Consolidated RAG System

Tests the unified HyperRAG system including:
- HippoIndex episodic memory with time-based decay
- BayesianTrustGraph with probabilistic reasoning
- ContextualVectorEngine with dual context tags
- CognitiveNexus analysis and reasoning
- GraphFixer gap detection and node proposals
- CreativityEngine non-obvious path discovery
- Integration with edge devices, P2P networks, and fog computing
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Import the unified RAG system
import sys
import tempfile
import time

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))

from rag import HyperRAG, MemoryType, QueryMode
from rag.graph.bayesian_trust_graph import Relationship, RelationshipType, create_graph_node
from rag.memory.hippo_index import create_episodic_document, create_hippo_node
from rag.vector.contextual_vector_engine import VectorDocument


class TestConsolidatedRAGSystem:
    """Test suite for the consolidated RAG system."""

    @pytest.fixture
    async def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    async def hyper_rag(self, temp_db_path):
        """Create HyperRAG instance for testing."""
        config = {
            "database_path": temp_db_path,
            "enable_edge_device_integration": True,
            "enable_p2p_integration": True,
            "enable_fog_computing": True,
            "hippo_max_nodes": 1000,
            "vector_max_documents": 5000,
            "graph_max_nodes": 2000,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        }

        rag = HyperRAG(config)
        await rag.initialize()
        yield rag
        await rag.close()

    @pytest.fixture
    async def sample_documents(self):
        """Create sample documents for testing."""
        return [
            {
                "id": "doc1",
                "content": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes that process information.",
                "title": "Introduction to Neural Networks",
                "metadata": {"chapter": "1", "book": "Deep Learning Fundamentals", "topic": "neural_networks"},
            },
            {
                "id": "doc2",
                "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers. It has revolutionized fields like computer vision and natural language processing.",
                "title": "Deep Learning Overview",
                "metadata": {"chapter": "2", "book": "Deep Learning Fundamentals", "topic": "deep_learning"},
            },
            {
                "id": "doc3",
                "content": "Transformers are a type of neural network architecture that relies entirely on attention mechanisms. They have become the foundation for large language models.",
                "title": "Transformer Architecture",
                "metadata": {"chapter": "3", "book": "Modern AI Architectures", "topic": "transformers"},
            },
            {
                "id": "doc4",
                "content": "Reinforcement learning is an area of machine learning where agents learn to make decisions by interacting with an environment and receiving rewards or penalties.",
                "title": "Reinforcement Learning Basics",
                "metadata": {"chapter": "4", "book": "AI Learning Methods", "topic": "reinforcement_learning"},
            },
            {
                "id": "doc5",
                "content": "Graph neural networks extend traditional neural networks to work with graph-structured data. They are particularly useful for social networks and molecular analysis.",
                "title": "Graph Neural Networks",
                "metadata": {"chapter": "5", "book": "Advanced Neural Architectures", "topic": "graph_networks"},
            },
        ]


class TestHyperRAGCore:
    """Test core HyperRAG functionality."""

    async def test_hyperrag_initialization(self, hyper_rag):
        """Test HyperRAG system initialization."""
        assert hyper_rag.initialized
        assert hyper_rag.hippo_index is not None
        assert hyper_rag.graph_system is not None
        assert hyper_rag.vector_engine is not None
        assert hyper_rag.cognitive_nexus is not None
        assert hyper_rag.graph_fixer is not None
        assert hyper_rag.creativity_engine is not None

        # Test integration bridges
        assert hyper_rag.edge_device_bridge is not None
        assert hyper_rag.p2p_bridge is not None
        assert hyper_rag.fog_bridge is not None

    async def test_document_ingestion(self, hyper_rag, sample_documents):
        """Test document ingestion into unified system."""
        results = []

        for doc in sample_documents:
            result = await hyper_rag.ingest_document(
                content=doc["content"],
                metadata=doc["metadata"],
                document_id=doc["id"],
                memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.VECTOR],
            )
            results.append(result)

        # Verify all ingestions succeeded
        for result in results:
            assert result["success"]
            assert "hippo_node_id" in result
            assert "graph_node_id" in result
            assert "vector_document_id" in result
            assert len(result["chunks_created"]) > 0

        # Verify counts
        stats = await hyper_rag.get_system_statistics()
        assert stats["hippo_nodes"] == len(sample_documents)
        assert stats["graph_nodes"] >= len(sample_documents)  # May have additional relationship nodes
        assert stats["vector_documents"] == len(sample_documents)

    async def test_query_modes(self, hyper_rag, sample_documents):
        """Test different query modes."""
        # Ingest documents first
        for doc in sample_documents:
            await hyper_rag.ingest_document(content=doc["content"], metadata=doc["metadata"], document_id=doc["id"])

        query = "neural networks and deep learning"

        # Test Fast mode
        fast_result = await hyper_rag.query(query, mode=QueryMode.FAST)
        assert fast_result.success
        assert len(fast_result.results) > 0
        assert fast_result.execution_time_ms < 2000  # Should be fast

        # Test Balanced mode
        balanced_result = await hyper_rag.query(query, mode=QueryMode.BALANCED)
        assert balanced_result.success
        assert len(balanced_result.results) >= len(fast_result.results)

        # Test Comprehensive mode
        comprehensive_result = await hyper_rag.query(query, mode=QueryMode.COMPREHENSIVE)
        assert comprehensive_result.success
        assert len(comprehensive_result.results) >= len(balanced_result.results)
        assert comprehensive_result.analysis_results is not None
        assert comprehensive_result.reasoning_chain is not None

        # Test Creative mode
        creative_result = await hyper_rag.query(query, mode=QueryMode.CREATIVE)
        assert creative_result.success
        assert creative_result.insights is not None
        assert len(creative_result.insights.get("non_obvious_connections", [])) > 0

        # Test Analytical mode
        analytical_result = await hyper_rag.query(query, mode=QueryMode.ANALYTICAL)
        assert analytical_result.success
        assert analytical_result.analysis_results is not None
        assert len(analytical_result.analysis_results) > 0


class TestHippoIndexIntegration:
    """Test HippoIndex episodic memory integration."""

    async def test_hippo_time_based_decay(self, hyper_rag, sample_documents):
        """Test time-based decay in HippoIndex."""
        # Ingest documents with different timestamps
        current_time = datetime.now()

        # Recent document
        recent_doc = sample_documents[0]
        await hyper_rag.hippo_index.store_node(
            create_hippo_node(content=recent_doc["content"], metadata=recent_doc["metadata"], timestamp=current_time)
        )

        # Old document (simulate as 1 week old)
        old_doc = sample_documents[1]
        old_timestamp = current_time - timedelta(days=7)
        await hyper_rag.hippo_index.store_node(
            create_hippo_node(content=old_doc["content"], metadata=old_doc["metadata"], timestamp=old_timestamp)
        )

        # Query and verify recency bias
        result = await hyper_rag.hippo_index.query_nodes(
            query="neural networks", limit=10, max_age_hours=24  # Only recent items
        )

        # Should find recent document but not old one
        recent_found = any("computational models" in r.content for r in result.nodes)
        assert recent_found

        # Query without time restriction
        all_result = await hyper_rag.hippo_index.query_nodes(query="neural networks", limit=10)

        # Should find both, but recent should have higher relevance
        assert len(all_result.nodes) >= len(result.nodes)

    async def test_hippo_episodic_chunking(self, hyper_rag):
        """Test episodic chunking and retrieval."""
        # Create episodic document with multiple events
        long_content = """
        First, the researcher initialized the neural network with random weights.
        Then, they loaded the training dataset containing 10,000 images.
        Next, they began the training process using backpropagation.
        After 100 epochs, the accuracy reached 85%.
        Finally, they evaluated the model on the test set.
        """

        result = await hyper_rag.hippo_index.store_episodic_document(
            create_episodic_document(
                content=long_content,
                metadata={"type": "experiment_log", "researcher": "Dr. Smith"},
                episode_type="research_experiment",
            )
        )

        assert result["success"]
        assert len(result["episodes"]) > 1  # Should be chunked into episodes

        # Query for specific episode
        query_result = await hyper_rag.hippo_index.query_nodes(query="training process backpropagation", limit=5)

        assert len(query_result.nodes) > 0
        training_episode = query_result.nodes[0]
        assert "backpropagation" in training_episode.content.lower()


class TestBayesianTrustGraphIntegration:
    """Test BayesianTrustGraph probabilistic reasoning."""

    async def test_graph_probabilistic_reasoning(self, hyper_rag, sample_documents):
        """Test Bayesian trust propagation in graph."""
        # Ingest documents to create graph nodes
        for doc in sample_documents:
            await hyper_rag.ingest_document(content=doc["content"], metadata=doc["metadata"], document_id=doc["id"])

        # Add explicit relationships with trust scores
        await hyper_rag.graph_system.add_relationship(
            Relationship(
                subject_id="neural_networks",
                predicate=RelationshipType.RELATES_TO,
                object_id="deep_learning",
                confidence=0.9,
                trust_score=0.8,
                evidence=["Both are machine learning concepts"],
            )
        )

        await hyper_rag.graph_system.add_relationship(
            Relationship(
                subject_id="deep_learning",
                predicate=RelationshipType.ENABLES,
                object_id="transformers",
                confidence=0.85,
                trust_score=0.9,
                evidence=["Transformers use deep learning"],
            )
        )

        # Query with trust propagation
        result = await hyper_rag.graph_system.retrieve_with_trust_propagation(
            query="neural networks", k=10, min_trust_score=0.3
        )

        assert len(result) > 0

        # Verify trust scores are propagated
        neural_net_nodes = [n for n in result if "neural" in n.content.lower()]
        assert len(neural_net_nodes) > 0

        # Higher trust nodes should appear first
        trust_scores = [n.trust_score for n in result]
        assert all(score >= 0.3 for score in trust_scores)

    async def test_belief_updating(self, hyper_rag):
        """Test Bayesian belief updating."""
        # Create initial belief
        initial_belief = {
            "proposition": "Neural networks are effective for image classification",
            "prior_probability": 0.7,
            "evidence": [],
        }

        belief_id = await hyper_rag.graph_system.store_belief(initial_belief)

        # Add supporting evidence
        evidence1 = {
            "type": "experimental_result",
            "description": "CNN achieved 95% accuracy on ImageNet",
            "reliability": 0.9,
            "supports_belief": True,
        }

        evidence2 = {
            "type": "research_paper",
            "description": "AlexNet paper showed breakthrough results",
            "reliability": 0.95,
            "supports_belief": True,
        }

        # Update belief with evidence
        updated_belief = await hyper_rag.graph_system.update_belief_with_evidence(belief_id, [evidence1, evidence2])

        # Posterior probability should be higher than prior
        assert updated_belief["posterior_probability"] > initial_belief["prior_probability"]
        assert len(updated_belief["evidence"]) == 2


class TestContextualVectorEngine:
    """Test ContextualVectorEngine with dual context tags."""

    async def test_dual_context_retrieval(self, hyper_rag, sample_documents):
        """Test retrieval with book and chapter context."""
        # Ingest documents with context metadata
        for doc in sample_documents:
            await hyper_rag.vector_engine.ingest_document(
                VectorDocument(
                    document_id=doc["id"],
                    content=doc["content"],
                    metadata=doc["metadata"],
                    book_context=doc["metadata"]["book"],
                    chapter_context=doc["metadata"]["chapter"],
                )
            )

        # Query with book context preference
        result = await hyper_rag.vector_engine.search(
            query="neural networks", k=5, context={"preferred_book": "Deep Learning Fundamentals"}
        )

        assert len(result) > 0

        # Results from preferred book should rank higher
        deep_learning_results = [doc for doc in result if doc.metadata.get("book") == "Deep Learning Fundamentals"]

        # Should find documents from the preferred book
        assert len(deep_learning_results) > 0

        # Test chapter-specific search
        chapter_result = await hyper_rag.vector_engine.search(
            query="neural networks",
            k=5,
            context={"preferred_book": "Deep Learning Fundamentals", "preferred_chapter": "1"},
        )

        # Should find chapter 1 content first
        chapter1_results = [doc for doc in chapter_result if doc.metadata.get("chapter") == "1"]
        assert len(chapter1_results) > 0

    async def test_contextual_chunking(self, hyper_rag):
        """Test intelligent contextual chunking."""
        long_document = """
        Chapter 1: Introduction to Neural Networks

        Neural networks are computational models inspired by biological neural networks.
        They consist of interconnected nodes that process information.
        The basic unit is called a neuron or node.

        Section 1.1: History of Neural Networks

        The concept of artificial neural networks dates back to the 1940s.
        McCulloch and Pitts proposed the first mathematical model.
        Perceptrons were developed in the 1950s by Frank Rosenblatt.

        Section 1.2: Modern Applications

        Today, neural networks are used in many fields.
        Computer vision applications include image recognition and object detection.
        Natural language processing uses neural networks for translation and generation.
        """

        result = await hyper_rag.vector_engine.ingest_document(
            VectorDocument(
                document_id="long_doc",
                content=long_document,
                metadata={"book": "Neural Network Guide", "chapter": "1"},
                book_context="Neural Network Guide",
                chapter_context="Chapter 1",
            )
        )

        # Verify chunks were created with proper context
        assert result["success"]
        assert len(result["chunks"]) > 1

        # Each chunk should maintain context information
        for chunk_info in result["chunks"]:
            chunk = chunk_info["chunk"]
            assert chunk.book_context == "Neural Network Guide"
            assert chunk.chapter_context == "Chapter 1"
            assert len(chunk.content) > 0


class TestGraphFixerIntegration:
    """Test GraphFixer gap detection and node proposals."""

    async def test_knowledge_gap_detection(self, hyper_rag, sample_documents):
        """Test detection of knowledge gaps."""
        # Ingest partial information
        partial_docs = sample_documents[:3]  # Only first 3 documents

        for doc in partial_docs:
            await hyper_rag.ingest_document(content=doc["content"], metadata=doc["metadata"], document_id=doc["id"])

        # Query for something that should reveal gaps
        query = "machine learning algorithms and their applications"

        query_result = await hyper_rag.query(query, mode=QueryMode.COMPREHENSIVE)

        # Run gap detection
        gaps = await hyper_rag.graph_fixer.detect_knowledge_gaps(query=query, retrieved_info=query_result.results)

        assert len(gaps) > 0

        # Should detect missing concepts
        gap_concepts = [gap.missing_concept for gap in gaps]

        # Might detect gaps like "reinforcement learning" or "graph networks"
        # since we only ingested 3 of 5 documents
        expected_gaps = ["reinforcement learning", "graph networks", "machine learning types"]
        any(any(expected in gap_concept.lower() for expected in expected_gaps) for gap_concept in gap_concepts)

        # At least some gaps should be conceptually relevant
        assert any(gap.confidence_score > 0.5 for gap in gaps)

    async def test_node_proposal_generation(self, hyper_rag):
        """Test automatic node proposal generation."""
        # Create sparse graph with obvious gaps
        await hyper_rag.graph_system.add_node(
            create_graph_node(
                content="Machine learning is a subset of AI",
                node_id="ml_concept",
                concepts=["machine_learning", "artificial_intelligence"],
            )
        )

        await hyper_rag.graph_system.add_node(
            create_graph_node(
                content="Supervised learning uses labeled data",
                node_id="supervised_concept",
                concepts=["supervised_learning", "labeled_data"],
            )
        )

        # No connection between ML and supervised learning - obvious gap

        # Run gap detection and node proposal
        gaps = await hyper_rag.graph_fixer.detect_knowledge_gaps(focus_area="machine learning")

        proposals = await hyper_rag.graph_fixer.propose_missing_nodes(gaps[:3])

        assert len(proposals) > 0

        # Should propose bridging concepts
        for proposal in proposals:
            assert proposal.confidence > 0.3
            assert len(proposal.proposed_content) > 0
            assert len(proposal.suggested_relationships) > 0

        # Look for bridging concepts
        bridging_proposals = [
            p
            for p in proposals
            if any(concept in p.proposed_content.lower() for concept in ["learning", "algorithm", "training"])
        ]

        assert len(bridging_proposals) > 0


class TestCreativityEngineIntegration:
    """Test CreativityEngine for non-obvious connections."""

    async def test_insight_discovery(self, hyper_rag, sample_documents):
        """Test discovery of non-obvious insights."""
        # Ingest all documents
        for doc in sample_documents:
            await hyper_rag.ingest_document(content=doc["content"], metadata=doc["metadata"], document_id=doc["id"])

        # Query that could lead to creative insights
        query = "neural networks and graph structures"

        query_result = await hyper_rag.query(query, mode=QueryMode.CREATIVE)

        assert query_result.success
        assert query_result.insights is not None

        insights = query_result.insights

        # Should find non-obvious connections
        assert "non_obvious_connections" in insights
        assert len(insights["non_obvious_connections"]) > 0

        # Should have analogical reasoning
        assert "analogical_insights" in insights

        # Should have creative synthesis
        assert "creative_synthesis" in insights

        # Verify quality of insights
        connections = insights["non_obvious_connections"]
        high_quality_connections = [conn for conn in connections if conn.get("confidence", 0) > 0.6]

        assert len(high_quality_connections) > 0

    async def test_cross_domain_analogies(self, hyper_rag):
        """Test cross-domain analogical reasoning."""
        # Add diverse domain knowledge
        domains = [
            {"content": "Neural networks have layers of connected neurons", "domain": "computer_science"},
            {"content": "Corporate organizations have hierarchical structures", "domain": "business"},
            {"content": "Biological ecosystems have food webs and energy flows", "domain": "biology"},
        ]

        for i, domain_doc in enumerate(domains):
            await hyper_rag.ingest_document(
                content=domain_doc["content"], metadata={"domain": domain_doc["domain"]}, document_id=f"domain_{i}"
            )

        # Query for cross-domain insights
        query = "hierarchical structures and information flow"

        insights = await hyper_rag.creativity_engine.discover_insights(
            query=query, retrieved_info=[], creativity_level=0.8  # Will retrieve internally  # High creativity
        )

        assert "analogical_insights" in insights

        analogies = insights["analogical_insights"]

        # Should find analogies between domains
        cross_domain_analogies = [analogy for analogy in analogies if analogy.get("cross_domain", False)]

        assert len(cross_domain_analogies) > 0

        # Verify analogy quality
        quality_analogies = [analogy for analogy in cross_domain_analogies if analogy.get("confidence", 0) > 0.5]

        assert len(quality_analogies) > 0


class TestIntegrationBridges:
    """Test integration bridges with edge devices, P2P, and fog computing."""

    async def test_edge_device_integration(self, hyper_rag):
        """Test edge device integration and optimization."""
        # Test mobile device context
        mobile_context = {
            "device_type": "mobile",
            "battery_level": 0.3,  # Low battery
            "network_type": "cellular",
            "available_memory_mb": 512,
        }

        # Query with mobile optimization
        result = await hyper_rag.query(
            query="neural networks", mode=QueryMode.FAST, context=mobile_context  # Should use fast mode for mobile
        )

        assert result.success
        assert result.execution_time_ms < 1000  # Should be optimized for mobile

        # Test edge device coordination
        edge_status = await hyper_rag.edge_device_bridge.get_edge_coordination_status()

        assert "registered_devices" in edge_status
        assert "resource_utilization" in edge_status
        assert "mobile_optimization_active" in edge_status

    async def test_p2p_network_integration(self, hyper_rag):
        """Test P2P network knowledge sharing."""
        # Test collaborative query
        collaborative_result = await hyper_rag.p2p_bridge.collaborative_query(
            query="machine learning applications", max_peer_responses=3, timeout_ms=5000
        )

        # Should handle gracefully even without active peers
        assert "local_results" in collaborative_result
        assert "peer_contributions" in collaborative_result
        assert "trust_scores" in collaborative_result

        # Test knowledge synchronization
        sync_status = await hyper_rag.p2p_bridge.get_sync_status()

        assert "last_sync_time" in sync_status
        assert "pending_updates" in sync_status
        assert "peer_count" in sync_status

    async def test_fog_computing_integration(self, hyper_rag):
        """Test fog computing distributed processing."""
        # Test distributed query
        distributed_result = await hyper_rag.fog_bridge.distributed_rag_query(
            query="deep learning architectures", query_mode="comprehensive", max_latency_ms=10000
        )

        # Should handle gracefully even without fog nodes
        if "error" not in distributed_result:
            assert "results" in distributed_result
            assert "distributed_processing" in distributed_result
            assert "execution_time_ms" in distributed_result

        # Test infrastructure status
        status = await hyper_rag.fog_bridge.get_fog_infrastructure_status()

        assert "infrastructure_health" in status
        assert "resource_utilization" in status
        assert "performance_metrics" in status


class TestSystemPerformance:
    """Test system performance and scalability."""

    async def test_concurrent_queries(self, hyper_rag, sample_documents):
        """Test concurrent query processing."""
        # Ingest documents
        for doc in sample_documents:
            await hyper_rag.ingest_document(content=doc["content"], metadata=doc["metadata"], document_id=doc["id"])

        # Create multiple concurrent queries
        queries = ["neural networks", "deep learning", "transformers", "reinforcement learning", "machine learning"]

        start_time = time.time()

        # Execute all queries concurrently
        tasks = [hyper_rag.query(query, mode=QueryMode.BALANCED) for query in queries]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # Convert to ms

        # Verify all queries succeeded
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == len(queries)

        # Concurrent execution should be faster than sequential
        avg_time_per_query = total_time / len(queries)
        assert avg_time_per_query < 2000  # Should average under 2 seconds per query

        # Verify result quality
        for result in successful_results:
            assert result.success
            assert len(result.results) > 0

    async def test_memory_usage_optimization(self, hyper_rag):
        """Test memory usage optimization."""
        # Get initial memory stats
        initial_stats = await hyper_rag.get_system_statistics()

        # Ingest many documents to test memory management
        large_content = "This is a large document. " * 1000  # ~20KB document

        for i in range(50):  # Ingest 50 large documents
            await hyper_rag.ingest_document(
                content=large_content + f" Document {i}",
                metadata={"doc_id": i, "size": "large"},
                document_id=f"large_doc_{i}",
            )

        # Get final memory stats
        final_stats = await hyper_rag.get_system_statistics()

        # Verify reasonable memory usage
        assert final_stats["hippo_nodes"] == initial_stats["hippo_nodes"] + 50
        assert final_stats["vector_documents"] == initial_stats["vector_documents"] + 50

        # Test cleanup
        cleanup_result = await hyper_rag.cleanup_old_memories(
            max_age_hours=1, max_hippo_nodes=30  # Very recent, should clean minimal  # Force cleanup based on count
        )

        assert cleanup_result["cleaned_up"]
        assert cleanup_result["nodes_removed"] > 0


@pytest.mark.asyncio
async def test_full_system_integration():
    """Integration test of the complete RAG system."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name

    try:
        # Initialize system
        config = {
            "database_path": temp_db,
            "enable_edge_device_integration": True,
            "enable_p2p_integration": True,
            "enable_fog_computing": True,
        }

        rag = HyperRAG(config)
        await rag.initialize()

        # Test full workflow

        # 1. Ingest knowledge
        knowledge_base = [
            "Artificial intelligence is the simulation of human intelligence in machines.",
            "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
            "Deep learning uses neural networks with multiple layers to model complex patterns.",
            "Natural language processing enables computers to understand and generate human language.",
            "Computer vision allows machines to interpret and understand visual information.",
        ]

        for i, content in enumerate(knowledge_base):
            result = await rag.ingest_document(
                content=content, metadata={"topic": "ai", "doc_id": i}, document_id=f"kb_{i}"
            )
            assert result["success"]

        # 2. Test different query modes
        query = "artificial intelligence and machine learning"

        # Fast query
        fast_result = await rag.query(query, mode=QueryMode.FAST)
        assert fast_result.success

        # Creative query with insights
        creative_result = await rag.query(query, mode=QueryMode.CREATIVE)
        assert creative_result.success
        assert creative_result.insights is not None

        # Analytical query with reasoning
        analytical_result = await rag.query(query, mode=QueryMode.ANALYTICAL)
        assert analytical_result.success
        assert analytical_result.analysis_results is not None

        # 3. Test knowledge gap detection and fixing
        gaps = await rag.graph_fixer.detect_knowledge_gaps(query="AI applications in healthcare")

        if gaps:
            proposals = await rag.graph_fixer.propose_missing_nodes(gaps[:2])
            assert len(proposals) > 0

        # 4. Test creativity engine
        insights = await rag.creativity_engine.discover_insights(
            query="connections between AI and biological intelligence",
            retrieved_info=fast_result.results,
            creativity_level=0.7,
        )

        assert "non_obvious_connections" in insights

        # 5. Test system statistics
        stats = await rag.get_system_statistics()
        assert stats["hippo_nodes"] == len(knowledge_base)
        assert stats["vector_documents"] == len(knowledge_base)
        assert stats["queries_processed"] >= 3

        # 6. Test system health
        health = await rag.get_system_health()
        assert health["overall_status"] == "healthy"
        assert health["component_status"]["hippo_index"] == "operational"
        assert health["component_status"]["graph_system"] == "operational"
        assert health["component_status"]["vector_engine"] == "operational"

        await rag.close()

        print("✅ Full system integration test passed!")

    finally:
        Path(temp_db).unlink(missing_ok=True)


if __name__ == "__main__":
    # Run the full integration test
    asyncio.run(test_full_system_integration())
