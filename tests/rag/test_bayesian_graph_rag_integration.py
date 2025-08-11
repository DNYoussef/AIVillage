"""
Comprehensive Integration Test for Bayesian Graph-Enhanced RAG System.

Tests the complete system including:
- Contextual tagging and intelligent chunking
- Bayesian trust graph construction
- Semantic relationship detection
- Trust propagation across graph
- Graph-enhanced retrieval with relationship traversal
"""

import asyncio
import json
import sys
import time

sys.path.append("src/production/rag/rag_system/core")

from codex_rag_integration import Document
from graph_enhanced_rag_pipeline import GraphEnhancedRAGPipeline


async def test_bayesian_graph_rag_system():
    """Test the complete Bayesian graph RAG system."""

    print("Bayesian Graph-Enhanced RAG System Integration Test")
    print("=" * 60)

    # Initialize the complete system
    print("[INIT] Initializing Graph-Enhanced RAG Pipeline...")
    pipeline = GraphEnhancedRAGPipeline(
        enable_intelligent_chunking=True,
        enable_contextual_tagging=True,
        enable_trust_graph=True,
        # Chunking parameters
        chunking_window_size=3,
        chunking_min_sentences=2,
        chunking_max_sentences=10,
        # Graph parameters
        graph_similarity_threshold=0.25,
        trust_decay_factor=0.85,
        max_propagation_hops=3,
        relationship_confidence_threshold=0.5,
    )
    print("Graph-Enhanced RAG Pipeline initialized successfully")

    # Create comprehensive test documents
    test_documents = [
        Document(
            id="deep_learning_basics",
            title="Deep Learning: Fundamentals and Applications",
            content="""
            Deep Learning Fundamentals
            
            Deep learning is a subset of machine learning that uses neural networks with multiple layers to model complex patterns in data. Unlike traditional machine learning algorithms, deep learning systems can automatically discover representations from raw data without manual feature engineering.
            
            Neural Network Architecture
            
            The foundation of deep learning lies in artificial neural networks inspired by biological neural systems. Each neuron processes input signals, applies weights and biases, and produces an output through an activation function. Multiple layers of neurons create the "deep" structure.
            
            Convolutional Neural Networks
            
            CNNs excel at processing grid-like data such as images. They use convolutional layers that apply filters to detect features like edges, textures, and patterns. Pooling layers reduce spatial dimensions while preserving important information.
            
            Applications in Computer Vision
            
            Deep learning has revolutionized computer vision tasks including image classification, object detection, and semantic segmentation. Models like ResNet and VGG have achieved superhuman performance on various visual recognition benchmarks.
            """,
            source_type="textbook",
            metadata={
                "author": "Dr. Neural Network Expert",
                "publication_date": "2024-01-10",
                "credibility_score": 0.95,
                "target_audience": "students",
            },
        ),
        Document(
            id="medical_ai_applications",
            title="Artificial Intelligence in Healthcare: Diagnostic Applications",
            content="""
            AI-Powered Medical Diagnosis
            
            Artificial intelligence is transforming healthcare through advanced diagnostic capabilities. Machine learning models can analyze medical images, predict disease progression, and assist healthcare professionals in making accurate diagnoses.
            
            Medical Image Analysis
            
            Convolutional neural networks have shown remarkable success in analyzing medical images such as X-rays, MRIs, and CT scans. These systems can detect abnormalities that human radiologists might miss, improving diagnostic accuracy and speed.
            
            Deep Learning in Radiology
            
            Deep learning models trained on large datasets of medical images can identify patterns associated with various diseases. For example, CNN architectures like U-Net are particularly effective for medical image segmentation tasks.
            
            Clinical Decision Support
            
            AI systems provide real-time decision support to healthcare professionals by analyzing patient data, medical history, and current symptoms. These tools help reduce diagnostic errors and improve patient outcomes.
            
            Challenges and Ethics
            
            Despite promising results, medical AI faces challenges including data privacy, algorithm bias, and the need for regulatory approval. Ensuring fairness and transparency in AI-driven healthcare decisions is crucial.
            """,
            source_type="research_paper",
            metadata={
                "author": "Medical AI Research Team",
                "publication_date": "2024-02-05",
                "credibility_score": 0.90,
                "target_audience": "healthcare_professionals",
            },
        ),
        Document(
            id="ai_ethics_healthcare",
            title="Ethical Considerations for AI in Medical Practice",
            content="""
            Ethical Framework for Medical AI
            
            The integration of artificial intelligence in healthcare raises important ethical considerations that must be addressed to ensure responsible deployment. Patient safety, privacy, and equitable access are paramount concerns.
            
            Algorithmic Bias in Healthcare
            
            AI systems can perpetuate or amplify existing healthcare disparities if training data lacks diversity. Biased algorithms may provide suboptimal care recommendations for underrepresented populations, highlighting the need for inclusive dataset curation.
            
            Privacy and Data Protection
            
            Medical AI systems require access to sensitive patient information, raising concerns about data privacy and security. Robust data governance frameworks must protect patient confidentiality while enabling beneficial AI applications.
            
            Transparency and Explainability
            
            Healthcare professionals and patients need to understand how AI systems reach their conclusions. Black-box algorithms that cannot explain their decision-making process may not be suitable for critical medical applications.
            
            Regulatory Oversight
            
            Government agencies like the FDA are developing frameworks for evaluating and approving AI-based medical devices. Rigorous testing and validation ensure that AI systems meet safety and efficacy standards before clinical deployment.
            """,
            source_type="policy_doc",
            metadata={
                "author": "Healthcare Ethics Committee",
                "publication_date": "2024-03-01",
                "credibility_score": 0.88,
                "target_audience": "policymakers",
            },
        ),
    ]

    # Index documents with complete graph integration
    print(f"\n[PROCESS] Indexing {len(test_documents)} documents with graph integration...")
    start_time = time.perf_counter()
    indexing_stats = pipeline.index_documents(test_documents)
    indexing_time = time.perf_counter() - start_time

    print(f"[SUCCESS] Document indexing completed in {indexing_time:.2f}s")
    print(f"  Documents Processed: {indexing_stats['documents_processed']}")
    print(f"  Total Chunks Created: {indexing_stats['chunks_created']}")
    print(f"  Chunks in Knowledge Graph: {indexing_stats.get('chunks_in_graph', 0)}")
    print(f"  Semantic Relationships: {indexing_stats.get('relationships_detected', 0)}")
    print(f"  Trust Propagation Time: {indexing_stats.get('trust_propagation_time_ms', 0):.1f}ms")
    print(f"  Graph Integration: {'Enabled' if indexing_stats.get('graph_integration_enabled', False) else 'Disabled'}")

    # Test graph-enhanced retrieval with various query types
    test_queries = [
        {
            "query": "How do convolutional neural networks work in medical image analysis?",
            "expected_concepts": ["CNN", "medical imaging", "deep learning"],
            "description": "Cross-domain technical query",
        },
        {
            "query": "What are the ethical concerns with AI bias in healthcare?",
            "expected_concepts": ["bias", "ethics", "healthcare AI"],
            "description": "Ethics and bias focus",
        },
        {
            "query": "How does deep learning help with computer vision tasks?",
            "expected_concepts": [
                "deep learning",
                "computer vision",
                "neural networks",
            ],
            "description": "Technical capability query",
        },
        {
            "query": "What privacy issues arise with medical AI systems?",
            "expected_concepts": ["privacy", "medical AI", "data protection"],
            "description": "Privacy-focused query",
        },
        {
            "query": "How do U-Net architectures work for image segmentation?",
            "expected_concepts": ["U-Net", "segmentation", "architecture"],
            "description": "Specific architecture query",
        },
    ]

    print("\n[TEST] Testing Graph-Enhanced Retrieval:")
    print("-" * 60)

    total_queries = len(test_queries)
    successful_retrievals = 0
    total_retrieval_time = 0

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected_concepts = test_case["expected_concepts"]
        description = test_case["description"]

        print(f"\nQuery {i}/{total_queries}: {description}")
        print(f"Question: {query}")

        # Perform retrieval with graph traversal
        start_time = time.perf_counter()
        results, metrics = await pipeline.retrieve_with_graph_enhanced_analysis(
            query=query,
            k=2,  # Get top 2 results
            enable_graph_traversal=True,
            trust_weight=0.35,
            centrality_weight=0.25,
            similarity_weight=0.40,
            min_trust_score=0.3,
            traversal_depth=2,
        )
        retrieval_time = (time.perf_counter() - start_time) * 1000
        total_retrieval_time += retrieval_time

        print(f"  Retrieval Time: {retrieval_time:.1f}ms")
        print(f"  Results Found: {len(results)}")
        print(f"  Graph Enhanced: {metrics.get('graph_enhanced_retrieval', False)}")

        if results:
            successful_retrievals += 1

            # Analyze best result
            best_result = results[0]
            metadata = best_result.metadata or {}

            print("  Top Result:")
            print(f"    Document: {best_result.document_id}")
            print(f"    Combined Score: {best_result.score:.3f}")
            print(f"    Trust Score: {metadata.get('trust_score', 0):.3f}")
            print(f"    Centrality Score: {metadata.get('centrality_score', 0):.3f}")
            print(f"    Semantic Similarity: {metadata.get('semantic_similarity', 0):.3f}")
            print(f"    Traversal Depth: {metadata.get('traversal_depth', 0)}")
            print(f"    Relationship: {metadata.get('relationship_type', 'direct')}")

            # Check concept coverage
            result_text = best_result.text.lower()
            concepts_found = sum(1 for concept in expected_concepts if concept.lower() in result_text)
            concept_coverage = concepts_found / len(expected_concepts)
            print(f"    Concept Coverage: {concept_coverage:.1%} ({concepts_found}/{len(expected_concepts)})")

            # Show snippet
            snippet = best_result.text[:150].replace("\n", " ").strip()
            print(f"    Snippet: {snippet}...")

            # Show relationship info if traversed
            if metadata.get("traversal_depth", 0) > 0:
                parent_chunk = metadata.get("parent_chunk")
                relationship_type = metadata.get("relationship_type")
                print(f"    Graph Traversal: Connected via '{relationship_type}' from {parent_chunk}")
        else:
            print("    No results found")

    # Calculate retrieval success rate
    success_rate = (successful_retrievals / total_queries) * 100
    avg_retrieval_time = total_retrieval_time / total_queries

    # Analyze graph structure and relationships
    print("\n[ANALYSIS] Knowledge Graph Analysis:")
    print("-" * 60)

    if pipeline.trust_graph and pipeline.trust_graph.chunk_nodes:
        # Get sample chunks for relationship analysis
        chunk_ids = list(pipeline.trust_graph.chunk_nodes.keys())

        print("Chunk Relationship Analysis:")
        for chunk_id in chunk_ids:
            analysis = pipeline.analyze_graph_relationships(chunk_id)

            if "error" not in analysis:
                graph_info = analysis["graph_analysis"]
                relationships = graph_info["relationships"]

                print(f"\n  Chunk: {chunk_id}")
                print(f"    Trust Score: {graph_info['trust_score']:.3f}")
                print(f"    Centrality: {graph_info['centrality_score']:.3f}")
                print(f"    Base Credibility: {graph_info['base_credibility']:.3f}")
                print(f"    Quality Score: {graph_info['quality_score']:.3f}")
                print(f"    Outgoing Relations: {relationships['total_outgoing']}")
                print(f"    Incoming Relations: {relationships['total_incoming']}")

                if relationships["outgoing"]:
                    rel_types = graph_info["relationship_types"]["outgoing_types"]
                    print(f"    Relationship Types: {', '.join(rel_types)}")

    # Get comprehensive performance metrics
    performance_metrics = pipeline.get_comprehensive_performance_metrics()

    print(f"\n{'='*60}")
    print("Final System Assessment")
    print("=" * 60)

    # Core metrics
    print("Retrieval Performance:")
    print(f"  Success Rate: {success_rate:.1f}% ({successful_retrievals}/{total_queries})")
    print(f"  Average Latency: {avg_retrieval_time:.1f}ms")
    print(f"  Target Met (<100ms): {'Yes' if avg_retrieval_time < 100 else 'No'}")

    # Graph integration metrics
    graph_integration = performance_metrics.get("graph_integration", {})
    print("\nGraph Integration:")
    print(f"  Status: {'Active' if graph_integration.get('enabled', False) else 'Inactive'}")
    print(f"  Chunks in Graph: {graph_integration.get('chunks_in_graph', 0)}")
    print(f"  Relationships: {graph_integration.get('relationships_detected', 0)}")
    print(f"  Trust Propagations: {graph_integration.get('trust_propagations', 0)}")

    # Graph structure analysis
    graph_stats = performance_metrics.get("graph_statistics", {})
    if "graph_structure" in graph_stats:
        structure = graph_stats["graph_structure"]
        trust_metrics = graph_stats.get("trust_metrics", {})

        print("\nKnowledge Graph Structure:")
        print(f"  Nodes: {structure.get('nodes', 0)}")
        print(f"  Edges: {structure.get('edges', 0)}")
        print(f"  Graph Density: {structure.get('density', 0):.3f}")
        print(f"  Connected: {'Yes' if structure.get('is_connected', False) else 'No'}")

        print("\nTrust Propagation Results:")
        print(f"  Average Trust: {trust_metrics.get('avg_trust_score', 0):.3f}")
        print(
            f"  Trust Range: {trust_metrics.get('min_trust_score', 0):.3f} - {trust_metrics.get('max_trust_score', 0):.3f}"
        )

        # Relationship type distribution
        rel_dist = graph_stats.get("relationship_distribution", {})
        if rel_dist:
            print("\nRelationship Types Found:")
            for rel_type, count in sorted(rel_dist.items(), key=lambda x: x[1], reverse=True):
                print(f"  {rel_type.title()}: {count}")

    # Quality metrics
    contextual_quality = performance_metrics.get("contextual_quality", {})
    print("\nContent Quality:")
    print(f"  Context Richness: {contextual_quality.get('avg_context_richness', 0):.3f}")
    print(f"  Context Coverage: {contextual_quality.get('context_coverage', 0):.1%}")

    # Final assessment
    print("\n[ASSESSMENT] System Status:")

    # Evaluation criteria
    graph_working = (
        graph_integration.get("enabled", False)
        and graph_integration.get("chunks_in_graph", 0) > 0
        and graph_integration.get("relationships_detected", 0) > 0
    )

    trust_working = graph_integration.get("trust_propagations", 0) > 0 and trust_metrics.get("avg_trust_score", 0) > 0

    retrieval_working = success_rate >= 80  # 80% success target
    performance_good = avg_retrieval_time < 100  # Sub-100ms target
    context_rich = contextual_quality.get("avg_context_richness", 0) > 0.6

    if graph_working and trust_working and retrieval_working and performance_good and context_rich:
        print("EXCELLENT: Bayesian Graph RAG System fully operational!")
        print("  - Knowledge graph construction: Working")
        print("  - Semantic relationship detection: Working")
        print("  - Bayesian trust propagation: Working")
        print("  - Graph-enhanced retrieval: Working")
        print("  - Performance targets: Met")
        status = "EXCELLENT"
    elif graph_working and trust_working and retrieval_working:
        print("GOOD: Core graph RAG functionality working")
        print("  - Graph and trust systems: Operational")
        print("  - Retrieval accuracy: Acceptable")
        print("  - May need performance optimization")
        status = "GOOD"
    elif graph_working:
        print("PARTIAL: Graph construction working, trust/retrieval issues")
        print("  - Check trust propagation and query processing")
        status = "PARTIAL"
    else:
        print("NEEDS WORK: Graph integration not functioning properly")
        print("  - Check document indexing and graph construction")
        status = "NEEDS_WORK"

    # Save test results
    test_results = {
        "timestamp": time.time(),
        "status": status,
        "performance": {
            "success_rate": success_rate,
            "avg_retrieval_time_ms": avg_retrieval_time,
            "total_queries": total_queries,
            "successful_retrievals": successful_retrievals,
        },
        "graph_metrics": graph_integration,
        "indexing_stats": indexing_stats,
        "query_results": [],
    }

    # Save detailed results
    with open("bayesian_graph_rag_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print("\nTest results saved to: bayesian_graph_rag_test_results.json")

    return status == "EXCELLENT"


if __name__ == "__main__":
    success = asyncio.run(test_bayesian_graph_rag_system())
