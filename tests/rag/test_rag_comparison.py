#!/usr/bin/env python3
"""
Compare basic CODEX vs enhanced BayesRAG pipeline performance.
"""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path("src/production/rag/rag_system/core")))


async def compare_pipelines():
    """Compare basic vs enhanced pipeline."""

    from bayesrag_codex_enhanced import BayesRAGEnhancedPipeline
    from codex_rag_integration import CODEXRAGPipeline, Document

    # Test documents
    docs = [
        Document(
            id="ai_doc",
            title="AI Overview",
            content="Artificial intelligence (AI) is computer science focused on creating intelligent machines. Machine learning is a subset of AI that learns from data. Deep learning uses neural networks with multiple layers. Applications include computer vision, natural language processing, robotics, and autonomous vehicles.",
            source_type="test",
        ),
        Document(
            id="climate_doc",
            title="Climate Change",
            content="Climate change refers to long-term changes in global temperatures and weather patterns. Main causes include burning fossil fuels, deforestation, and industrial activities. Effects include rising sea levels, extreme weather, and ecosystem disruption. Solutions involve renewable energy and carbon reduction.",
            source_type="test",
        ),
    ]

    questions = [
        "What is artificial intelligence?",
        "What causes climate change?",
        "What are machine learning applications?",
        "How can we address climate change?",
    ]

    print("PIPELINE COMPARISON TEST")
    print("=" * 50)

    # Test Basic CODEX Pipeline
    print("\n1. BASIC CODEX PIPELINE")
    print("-" * 30)

    basic_pipeline = CODEXRAGPipeline()
    basic_stats = basic_pipeline.index_documents(docs)
    print(f"Indexed: {basic_stats['documents_processed']} docs, {basic_stats['chunks_created']} chunks")

    basic_results = []

    for q in questions:
        try:
            results, metrics = await basic_pipeline.retrieve(q, k=3)
            success = len(results) > 0
            latency = metrics.get("latency_ms", 0)

            print(f"Q: {q}")
            print(f"   Results: {len(results)}, Latency: {latency:.1f}ms")

            if results:
                print(f"   Answer: {results[0].text[:100]}...")

            basic_results.append({"question": q, "success": success, "latency": latency})

        except Exception as e:
            print(f"   ERROR: {e}")
            basic_results.append({"question": q, "success": False, "latency": 0})

    # Test Enhanced BayesRAG Pipeline
    print("\n2. ENHANCED BAYESRAG PIPELINE")
    print("-" * 30)

    enhanced_pipeline = BayesRAGEnhancedPipeline()
    enhanced_stats = enhanced_pipeline.index_documents(docs)
    print(f"Indexed: {enhanced_stats['documents_processed']} docs, {enhanced_stats['chunks_created']} chunks")

    enhanced_results = []

    for q in questions:
        try:
            results, metrics = await enhanced_pipeline.retrieve_with_trust(q, k=3)
            success = len(results) > 0
            latency = metrics.get("latency_ms", 0)

            print(f"Q: {q}")
            print(f"   Results: {len(results)}, Latency: {latency:.1f}ms")

            if results:
                print(f"   Answer: {results[0].text[:100]}...")
                if hasattr(results[0], "trust_metrics") and results[0].trust_metrics:
                    print(f"   Trust: {results[0].trust_metrics.trust_score:.3f}")

            enhanced_results.append({"question": q, "success": success, "latency": latency})

        except Exception as e:
            print(f"   ERROR: {e}")
            enhanced_results.append({"question": q, "success": False, "latency": 0})

    # Summary comparison
    print("\n3. COMPARISON SUMMARY")
    print("-" * 30)

    basic_success = sum(1 for r in basic_results if r["success"])
    enhanced_success = sum(1 for r in enhanced_results if r["success"])

    basic_avg_latency = sum(r["latency"] for r in basic_results if r["success"])
    basic_avg_latency = basic_avg_latency / basic_success if basic_success > 0 else 0

    enhanced_avg_latency = sum(r["latency"] for r in enhanced_results if r["success"])
    enhanced_avg_latency = enhanced_avg_latency / enhanced_success if enhanced_success > 0 else 0

    print(f"Basic Pipeline:    {basic_success}/{len(questions)} success, {basic_avg_latency:.1f}ms avg")
    print(f"Enhanced Pipeline: {enhanced_success}/{len(questions)} success, {enhanced_avg_latency:.1f}ms avg")

    if basic_success > enhanced_success:
        print("\nWARNING: Enhanced pipeline performs worse than basic!")
        print("This suggests issues with the BayesRAG enhancements.")
    elif enhanced_success == basic_success:
        print("\nINFO: Both pipelines perform similarly.")
        print("Enhanced features may not be affecting basic retrieval.")
    else:
        print("\nGOOD: Enhanced pipeline performs better than basic.")

    return {
        "basic_success": basic_success,
        "enhanced_success": enhanced_success,
        "basic_latency": basic_avg_latency,
        "enhanced_latency": enhanced_avg_latency,
    }


if __name__ == "__main__":
    results = asyncio.run(compare_pipelines())
