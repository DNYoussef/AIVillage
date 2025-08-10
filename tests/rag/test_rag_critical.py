#!/usr/bin/env python3
"""
Critical test of the RAG system to verify it actually works.
"""

import asyncio
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path('src/production/rag/rag_system/core')))

def test_basic_pipeline():
    """Test if the basic CODEX pipeline actually works."""
    
    print("=== Testing Basic CODEX Pipeline ===")
    
    try:
        from codex_rag_integration import CODEXRAGPipeline, Document
        
        # Try to instantiate
        print("Creating pipeline...")
        pipeline = CODEXRAGPipeline()
        
        # Check if it has the expected attributes
        required_attrs = ['embedder', 'index', 'cache', 'chunk_store']
        for attr in required_attrs:
            if not hasattr(pipeline, attr):
                print(f"FAIL: Missing attribute {attr}")
                return False
            print(f"PASS: Has {attr}")
            
        # Check if embedder actually works
        print("Testing embedder...")
        test_text = "This is a test sentence."
        embedding = pipeline.embedder.encode(test_text)
        
        if embedding is None or len(embedding) == 0:
            print("FAIL: Embedder returned empty result")
            return False
            
        print(f"PASS: Embedder produced {len(embedding)}-dim vector")
        
        # Test document creation
        print("Testing document indexing...")
        test_doc = Document(
            id="test_1",
            title="Test Document", 
            content="This is a test document for verifying the RAG pipeline functionality. It contains some sample text to check if indexing works properly.",
            source_type="test"
        )
        
        # Try to index document
        stats = pipeline.index_documents([test_doc])
        
        if stats["documents_processed"] != 1:
            print(f"FAIL: Expected 1 document processed, got {stats['documents_processed']}")
            return False
            
        if stats["chunks_created"] < 1:
            print(f"FAIL: No chunks created from document")
            return False
            
        print(f"PASS: Indexed {stats['documents_processed']} docs, {stats['chunks_created']} chunks")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Exception in basic pipeline test: {e}")
        traceback.print_exc()
        return False

async def test_basic_retrieval():
    """Test if retrieval actually works."""
    
    print("\n=== Testing Basic Retrieval ===")
    
    try:
        from codex_rag_integration import CODEXRAGPipeline, Document
        
        pipeline = CODEXRAGPipeline()
        
        # Index some test documents
        test_docs = [
            Document(
                id="doc1",
                title="Machine Learning",
                content="Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed. It uses algorithms to analyze data and make predictions.",
                source_type="test"
            ),
            Document(
                id="doc2", 
                title="Deep Learning",
                content="Deep learning is a subset of machine learning that uses neural networks with multiple layers. It can automatically learn representations from data.",
                source_type="test"
            ),
            Document(
                id="doc3",
                title="Natural Language Processing",
                content="Natural language processing (NLP) is a field of AI that helps computers understand and process human language. It includes tasks like translation and sentiment analysis.",
                source_type="test"
            )
        ]
        
        print("Indexing test documents...")
        stats = pipeline.index_documents(test_docs)
        
        if stats["documents_processed"] != 3:
            print(f"FAIL: Expected 3 docs, processed {stats['documents_processed']}")
            return False
            
        print(f"PASS: Indexed {stats['documents_processed']} documents")
        
        # Test retrieval
        print("Testing retrieval...")
        query = "machine learning algorithms"
        
        start_time = time.perf_counter()
        results, metrics = await pipeline.retrieve(query, k=5)
        latency = (time.perf_counter() - start_time) * 1000
        
        if not isinstance(results, list):
            print(f"FAIL: Results not a list: {type(results)}")
            return False
            
        if len(results) == 0:
            print("FAIL: No results returned")
            return False
            
        print(f"PASS: Retrieved {len(results)} results in {latency:.1f}ms")
        
        # Check result structure
        first_result = results[0]
        required_fields = ['chunk_id', 'document_id', 'text', 'score']
        
        for field in required_fields:
            if not hasattr(first_result, field):
                print(f"FAIL: Result missing field {field}")
                return False
                
        print(f"PASS: Results have proper structure")
        print(f"Top result: {first_result.text[:100]}...")
        print(f"Score: {first_result.score:.4f}")
        
        # Test if results are relevant
        result_text = first_result.text.lower()
        if "machine" in result_text or "learning" in result_text:
            print("PASS: Results appear relevant")
        else:
            print("WARN: Results may not be relevant")
            
        return True
        
    except Exception as e:
        print(f"FAIL: Exception in retrieval test: {e}")
        traceback.print_exc()
        return False

def test_enhanced_pipeline():
    """Test if the enhanced BayesRAG pipeline actually works."""
    
    print("\n=== Testing Enhanced BayesRAG Pipeline ===")
    
    try:
        from bayesrag_codex_enhanced import BayesRAGEnhancedPipeline, TrustMetrics
        
        # Try to instantiate
        print("Creating enhanced pipeline...")
        pipeline = BayesRAGEnhancedPipeline()
        
        # Check if it inherits from base pipeline
        if not hasattr(pipeline, 'embedder'):
            print("FAIL: Enhanced pipeline missing base functionality")
            return False
            
        # Check enhanced attributes
        enhanced_attrs = ['trust_cache', 'hierarchy_index', 'cross_reference_graph']
        for attr in enhanced_attrs:
            if not hasattr(pipeline, attr):
                print(f"WARN: Missing enhanced attribute {attr}")
                
        print("PASS: Enhanced pipeline created")
        
        # Test trust metrics
        test_metrics = TrustMetrics(
            base_score=0.8,
            citation_count=50,
            source_quality=0.9
        )
        
        trust_score = test_metrics.trust_score
        
        if trust_score <= 0 or trust_score > 1:
            print(f"FAIL: Invalid trust score {trust_score}")
            return False
            
        print(f"PASS: Trust metrics working, score: {trust_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Exception in enhanced pipeline test: {e}")
        traceback.print_exc()
        return False

def test_semantic_cache():
    """Test if semantic cache actually works."""
    
    print("\n=== Testing Semantic Cache ===")
    
    try:
        from semantic_cache_advanced import SemanticMultiTierCache
        
        print("Creating semantic cache...")
        cache = SemanticMultiTierCache()
        
        # Check tier structure
        if not hasattr(cache, 'hot_cache') or not hasattr(cache, 'warm_cache') or not hasattr(cache, 'cold_cache'):
            print("FAIL: Missing cache tiers")
            return False
            
        print("PASS: Cache tiers created")
        
        # Test basic operations
        print("Testing cache operations...")
        
        # This is async, so we need to run it
        async def test_cache_ops():
            test_query = "test query"
            test_results = [{"text": "test result"}]
            
            # Should be cache miss initially
            result = await cache.get(test_query)
            if result is not None:
                print("FAIL: Should be cache miss initially")
                return False
                
            # Store in cache
            await cache.set(test_query, test_results, trust_score=0.8)
            
            # Should be cache hit now
            result = await cache.get(test_query)
            if result is None:
                print("FAIL: Should be cache hit after storing")
                return False
                
            print("PASS: Basic cache operations work")
            return True
            
        return asyncio.run(test_cache_ops())
        
    except Exception as e:
        print(f"FAIL: Exception in cache test: {e}")
        traceback.print_exc()
        return False

def test_production_monitoring():
    """Test if production monitoring actually works."""
    
    print("\n=== Testing Production Monitoring ===")
    
    try:
        from production_monitoring import ProductionMonitor, HealthStatus
        from codex_rag_integration import CODEXRAGPipeline
        from semantic_cache_advanced import SemanticMultiTierCache
        
        # Create mock components
        pipeline = CODEXRAGPipeline()
        cache = SemanticMultiTierCache()
        
        print("Creating production monitor...")
        monitor = ProductionMonitor(pipeline, cache)
        
        # Check if it has health checks
        if not hasattr(monitor, 'health_checks'):
            print("FAIL: Monitor missing health checks")
            return False
            
        if len(monitor.health_checks) == 0:
            print("FAIL: No health checks configured")
            return False
            
        print(f"PASS: Monitor has {len(monitor.health_checks)} health checks")
        
        # Check circuit breakers
        if not hasattr(monitor, 'circuit_breakers'):
            print("FAIL: Monitor missing circuit breakers") 
            return False
            
        if len(monitor.circuit_breakers) == 0:
            print("FAIL: No circuit breakers configured")
            return False
            
        print(f"PASS: Monitor has {len(monitor.circuit_breakers)} circuit breakers")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Exception in monitoring test: {e}")
        traceback.print_exc()
        return False

async def test_end_to_end():
    """Test complete end-to-end functionality."""
    
    print("\n=== Testing End-to-End Functionality ===")
    
    try:
        from bayesrag_codex_enhanced import BayesRAGEnhancedPipeline
        from codex_rag_integration import Document
        
        # Create enhanced pipeline
        pipeline = BayesRAGEnhancedPipeline()
        
        # Index some realistic documents  
        docs = [
            Document(
                id="ai_history",
                title="History of Artificial Intelligence",
                content="Artificial intelligence was founded as an academic discipline in 1956. Early AI research focused on problem solving and symbolic methods. The field experienced several AI winters when funding dried up. Deep learning breakthroughs in the 2010s led to renewed interest.",
                source_type="wikipedia",
                metadata={
                    "categories": ["Technology", "History"],
                    "trust_score": 0.9,
                    "citation_count": 100
                }
            ),
            Document(
                id="ml_algorithms",
                title="Machine Learning Algorithms",
                content="Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning. Popular algorithms include linear regression, decision trees, neural networks, and support vector machines. Each has different strengths and use cases.",
                source_type="wikipedia", 
                metadata={
                    "categories": ["Technology", "AI"],
                    "trust_score": 0.85,
                    "citation_count": 75
                }
            ),
            Document(
                id="quantum_computing",
                title="Quantum Computing Principles",
                content="Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform calculations. Quantum computers could solve certain problems exponentially faster than classical computers. However, they are still in early development stages.",
                source_type="wikipedia",
                metadata={
                    "categories": ["Technology", "Physics"], 
                    "trust_score": 0.8,
                    "citation_count": 50
                }
            )
        ]
        
        print("Indexing realistic documents...")
        stats = pipeline.index_documents(docs)
        
        if stats["documents_processed"] != 3:
            print(f"FAIL: Expected 3 docs, got {stats['documents_processed']}")
            return False
            
        # Test trust-weighted retrieval
        print("Testing trust-weighted retrieval...")
        
        queries = [
            "history of artificial intelligence research",
            "machine learning algorithm types", 
            "quantum computing applications"
        ]
        
        all_passed = True
        
        for query in queries:
            print(f"\nQuery: {query}")
            
            start_time = time.perf_counter()
            
            # Use enhanced retrieval if available
            if hasattr(pipeline, 'retrieve_with_trust'):
                results, metrics = await pipeline.retrieve_with_trust(query, k=3)
            else:
                results, metrics = await pipeline.retrieve(query, k=3)
                
            latency = (time.perf_counter() - start_time) * 1000
            
            if len(results) == 0:
                print(f"FAIL: No results for query")
                all_passed = False
                continue
                
            print(f"Results: {len(results)}, Latency: {latency:.1f}ms")
            
            # Check if results have trust information
            top_result = results[0]
            if hasattr(top_result, 'trust_metrics') and top_result.trust_metrics:
                print(f"Trust score: {top_result.trust_metrics.trust_score:.3f}")
                
            if hasattr(top_result, 'bayesian_score'):
                print(f"Bayesian score: {top_result.bayesian_score:.3f}")
                
            # Check result relevance
            result_text = top_result.text.lower()
            query_words = query.lower().split()
            
            relevance_found = any(word in result_text for word in query_words if len(word) > 3)
            
            if relevance_found:
                print("PASS: Result appears relevant")
            else:
                print("WARN: Result relevance questionable")
                
            # Check latency target
            if latency > 1000:  # 1 second for this test
                print(f"WARN: High latency {latency:.1f}ms")
                
        return all_passed
        
    except Exception as e:
        print(f"FAIL: Exception in end-to-end test: {e}")
        traceback.print_exc()
        return False

def main():
    """Run critical tests of the RAG system."""
    
    print("CRITICAL RAG SYSTEM TEST")
    print("=" * 50)
    
    tests = [
        ("Basic Pipeline", test_basic_pipeline),
        ("Basic Retrieval", test_basic_retrieval),
        ("Enhanced Pipeline", test_enhanced_pipeline),
        ("Semantic Cache", test_semantic_cache),
        ("Production Monitoring", test_production_monitoring),
        ("End-to-End", test_end_to_end)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(test_func())
            else:
                result = test_func()
                
            results[test_name] = result
            
            if result:
                print(f"PASS: {test_name}")
            else:
                print(f"FAIL: {test_name}")
                
        except Exception as e:
            print(f"ERROR: {test_name} - {e}")
            results[test_name] = False
            
    # Summary
    print("\n" + "="*50)
    print("CRITICAL TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
        
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nVERDICT: RAG system appears to be FUNCTIONAL")
        print("- Core components instantiate properly")
        print("- Document indexing works")
        print("- Retrieval returns results")
        print("- Enhanced features are present")
    elif passed >= total * 0.7:
        print("\nVERDICT: RAG system is PARTIALLY FUNCTIONAL")
        print("- Core functionality works")
        print("- Some advanced features may have issues")
    else:
        print("\nVERDICT: RAG system is BROKEN or STUB")
        print("- Core functionality fails")
        print("- Implementation may be incomplete")
        
    return passed == total

if __name__ == "__main__":
    main()