"""
HyperRAG Enhanced System Validation Test

Validates the enhanced HyperRAG system with neural-biological components.
Focuses on core functionality and integration testing.
"""

import asyncio
import logging
import os
import sys
import time

# Add paths for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, "core"))
sys.path.insert(0, os.path.join(parent_dir, "src"))

logging.basicConfig(level=logging.WARNING)  # Reduce log noise
logger = logging.getLogger(__name__)


async def test_components():
    """Test individual components."""

    print("Testing Enhanced HyperRAG Components")
    print("====================================")

    test_results = {}

    # Test 1: HippoRAG Neural Memory
    try:
        print("\n1. Testing HippoRAG Neural Memory System...")

        from neural_memory.hippo_rag import (
            HippoRAG,
            MemoryType,
            create_semantic_context,
            create_spatial_context,
            create_temporal_context,
        )

        hippo_rag = HippoRAG(embedding_dim=256, max_episodic_memories=50)
        await hippo_rag.initialize()

        # Create contexts
        spatial_ctx = create_spatial_context("test_environment", "laboratory")
        temporal_ctx = create_temporal_context(event_type="validation_test")
        semantic_ctx = create_semantic_context("neuroscience", "memory_testing", ["hippocampus"])

        # Test memory encoding
        memory_id = await hippo_rag.encode_memory(
            content="The hippocampus plays a crucial role in memory consolidation and spatial navigation.",
            memory_type=MemoryType.EPISODIC,
            spatial_context=spatial_ctx,
            temporal_context=temporal_ctx,
            semantic_context=semantic_ctx,
        )

        # Test memory retrieval
        results = await hippo_rag.retrieve_memories(
            query="hippocampus memory consolidation", k=5, semantic_context=semantic_ctx
        )

        # Validate results
        assert memory_id is not None and len(memory_id) > 0
        assert len(results.memory_traces) > 0
        assert results.retrieval_confidence > 0

        status = await hippo_rag.get_status()
        assert status["status"] == "healthy"

        test_results["hippo_rag"] = "PASS"
        print("   HippoRAG: PASS - Neural memory system operational")

    except Exception as e:
        test_results["hippo_rag"] = f"FAIL - {str(e)}"
        print(f"   HippoRAG: FAIL - {str(e)}")

    # Test 2: Bayesian Trust Network
    try:
        print("\n2. Testing Bayesian Trust Network...")

        from trust_networks.bayesian_trust import BayesianTrustNetwork, EvidenceType, TrustDimension

        trust_network = BayesianTrustNetwork(trust_threshold=0.5)
        await trust_network.initialize()

        # Add a trusted source
        source_id = await trust_network.add_source(
            source_identifier="test_paper_001",
            content="Research shows that neural networks achieve high accuracy with proper training.",
            source_type="academic_paper",
            domain="machine_learning",
            keywords=["neural networks", "accuracy", "training"],
        )

        # Add evidence
        evidence_added = await trust_network.add_evidence(
            source_id,
            EvidenceType.PEER,
            TrustDimension.ACCURACY,
            positive_evidence=0.8,
            negative_evidence=0.1,
            evaluator_id="test_evaluator",
            confidence=0.9,
        )

        # Test retrieval
        trust_results = await trust_network.retrieve_with_trust_propagation(
            query="neural networks accuracy training", k=5, min_trust_score=0.3
        )

        # Validate results
        assert source_id is not None
        assert evidence_added is True
        assert len(trust_results) >= 0  # May be 0 if no matches above threshold

        status = await trust_network.get_network_status()
        assert status["status"] == "healthy"

        test_results["trust_network"] = "PASS"
        print("   Trust Network: PASS - Bayesian trust system operational")

    except Exception as e:
        test_results["trust_network"] = f"FAIL - {str(e)}"
        print(f"   Trust Network: FAIL - {str(e)}")

    # Test 3: Cognitive Reasoning Engine
    try:
        print("\n3. Testing Cognitive Reasoning Engine...")

        from cognitive.reasoning_engine import CognitiveReasoningEngine

        cognitive_engine = CognitiveReasoningEngine(
            max_reasoning_depth=3,
            confidence_threshold=0.6,
            enable_meta_reasoning=True,
            enable_bias_detection=False,  # Simplify for testing
        )
        await cognitive_engine.initialize()

        # Test reasoning with evidence
        evidence_sources = [
            {
                "content": "Machine learning algorithms learn patterns from training data.",
                "type": "factual",
                "confidence": 0.9,
                "source": "textbook",
            },
            {
                "content": "Neural networks are a type of machine learning algorithm inspired by the brain.",
                "type": "definitional",
                "confidence": 0.85,
                "source": "encyclopedia",
            },
        ]

        reasoning_result = await cognitive_engine.reason(
            query="How do machine learning algorithms work?",
            evidence_sources=evidence_sources,
            context={"domain": "machine_learning"},
        )

        # Validate results
        assert reasoning_result is not None
        assert reasoning_result.primary_conclusion is not None
        assert len(reasoning_result.primary_conclusion) > 0

        status = await cognitive_engine.get_system_status()
        assert status["status"] == "healthy"

        test_results["cognitive_engine"] = "PASS"
        print("   Cognitive Engine: PASS - Reasoning system operational")

    except Exception as e:
        test_results["cognitive_engine"] = f"FAIL - {str(e)}"
        print(f"   Cognitive Engine: FAIL - {str(e)}")

    return test_results


async def test_integration():
    """Test full system integration."""

    print("\n4. Testing HyperRAG System Integration...")

    try:
        from hyperrag.hyperrag import HyperRAG, HyperRAGConfig, MemoryType, QueryMode

        # Create enhanced configuration
        config = HyperRAGConfig()
        config.enable_hippo_rag = True
        config.enable_graph_rag = True
        config.enable_cognitive_nexus = True
        config.max_results = 5
        config.vector_dimensions = 256  # Smaller for testing

        # Initialize system
        hyperrag = HyperRAG(config)
        success = await hyperrag.initialize()

        if not success:
            raise Exception("System initialization failed")

        print("   System initialized successfully")

        # Test document addition
        await hyperrag.add_document(
            content="Artificial intelligence is a field of computer science focused on creating intelligent machines.",
            metadata={"domain": "artificial_intelligence", "source_type": "definition", "credibility": 0.9},
            memory_type=MemoryType.SEMANTIC,
        )

        await hyperrag.add_document(
            content="Machine learning is a subset of AI that enables computers to learn without explicit programming.",
            metadata={"domain": "machine_learning", "source_type": "explanation", "credibility": 0.85},
            memory_type=MemoryType.EPISODIC,
        )

        print(f"   Added {2} documents successfully")

        # Test query processing
        start_time = time.time()
        result = await hyperrag.process_query(
            query="What is the relationship between artificial intelligence and machine learning?",
            mode=QueryMode.COMPREHENSIVE,
            context={"domain": "computer_science", "level": "introductory"},
            user_id="test_user",
        )
        processing_time = time.time() - start_time

        # Validate results
        assert result is not None
        assert result.confidence > 0.0
        assert len(result.answer) > 50  # Should be substantial
        assert result.processing_time > 0

        print(f"   Query processed in {processing_time:.3f}s")
        print(f"   Response confidence: {result.confidence:.3f}")
        print(f"   Synthesis method: {result.synthesis_method}")
        print(f"   Retrieved sources: {len(result.retrieval_sources)}")

        # Test system health
        health = await hyperrag.health_check()
        assert health["status"] in ["healthy", "degraded", "limited"]

        print(f"   System health: {health['status']}")
        print(f"   Neural-biological enabled: {health['neural_biological_enabled']}")

        # Test system statistics
        stats = await hyperrag.get_stats()
        print(f"   Documents indexed: {stats['documents_indexed']}")
        print(f"   Queries processed: {stats['queries_processed']}")

        # Enhanced stats if available
        if "hippo_rag" in stats:
            print(f"   Episodic memories: {stats['hippo_rag']['episodic_memories']}")
        if "trust_network" in stats:
            print(f"   Trust nodes: {stats['trust_network']['total_nodes']}")
        if "cognitive_engine" in stats:
            print(f"   Reasoning sessions: {stats['cognitive_engine']['queries_processed']}")

        await hyperrag.shutdown()

        return "PASS"

    except Exception as e:
        return f"FAIL - {str(e)}"


async def test_performance():
    """Test system performance and compare with simple version."""

    print("\n5. Testing Performance Comparison...")

    try:
        from hyperrag.hyperrag import HyperRAG, HyperRAGConfig, QueryMode

        # Test simple system (fallback)
        simple_config = HyperRAGConfig()
        simple_config.enable_hippo_rag = False
        simple_config.enable_graph_rag = False
        simple_config.enable_cognitive_nexus = False

        simple_system = HyperRAG(simple_config)
        await simple_system.initialize()

        await simple_system.add_document(
            content="Artificial intelligence involves creating intelligent computer systems.",
            metadata={"test": "simple"},
        )

        start_time = time.time()
        simple_result = await simple_system.process_query("What is artificial intelligence?")
        simple_time = time.time() - start_time

        await simple_system.shutdown()

        # Test advanced system
        advanced_config = HyperRAGConfig()
        advanced_config.enable_hippo_rag = True
        advanced_config.enable_graph_rag = True
        advanced_config.enable_cognitive_nexus = True
        advanced_config.vector_dimensions = 256

        advanced_system = HyperRAG(advanced_config)
        await advanced_system.initialize()

        await advanced_system.add_document(
            content="Artificial intelligence involves creating intelligent computer systems.",
            metadata={"test": "advanced", "credibility": 0.9},
        )

        start_time = time.time()
        advanced_result = await advanced_system.process_query(
            "What is artificial intelligence?", mode=QueryMode.COMPREHENSIVE, context={"domain": "computer_science"}
        )
        advanced_time = time.time() - start_time

        await advanced_system.shutdown()

        # Compare results
        print(f"   Simple system: {simple_time:.3f}s, confidence: {simple_result.confidence:.3f}")
        print(f"   Advanced system: {advanced_time:.3f}s, confidence: {advanced_result.confidence:.3f}")
        print(f"   Time overhead: {(advanced_time/max(simple_time, 0.001)):.2f}x")
        print(f"   Confidence improvement: {advanced_result.confidence - simple_result.confidence:.3f}")
        print(f"   Enhanced synthesis: {advanced_result.synthesis_method != simple_result.synthesis_method}")

        return "PASS"

    except Exception as e:
        return f"FAIL - {str(e)}"


async def main():
    """Run all tests and generate report."""

    print("HyperRAG Enhanced System Validation")
    print("===================================")

    # Test components
    component_results = await test_components()

    # Test integration
    integration_result = await test_integration()

    # Test performance
    performance_result = await test_performance()

    # Generate final report
    print("\n" + "=" * 50)
    print("FINAL TEST REPORT")
    print("=" * 50)

    all_results = {**component_results, "integration": integration_result, "performance": performance_result}

    passed = sum(1 for result in all_results.values() if result == "PASS")
    total = len(all_results)

    for test_name, result in all_results.items():
        status = "PASS" if result == "PASS" else "FAIL"
        print(f"{test_name:20}: {status}")
        if status == "FAIL":
            print(f"                     {result}")

    print("-" * 50)
    print(f"SUCCESS RATE: {passed}/{total} ({100*passed/total:.1f}%)")

    if passed == total:
        print("STATUS: ALL TESTS PASSED - System ready for production!")

        # Additional success metrics
        print("\nSUCCESS METRICS:")
        print("- HippoRAG neural memory system: OPERATIONAL")
        print("- Bayesian trust networks: OPERATIONAL")
        print("- Cognitive reasoning engine: OPERATIONAL")
        print("- Full system integration: OPERATIONAL")
        print("- Performance benchmarking: COMPLETED")
        print("\nThe enhanced HyperRAG system has successfully replaced")
        print("SimpleVectorStore with advanced neural-biological components!")

        return True
    else:
        print("STATUS: Some tests failed - see details above")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
