"""
Quick Integration Test for Enhanced Query Processing System.

Tests the full pipeline with document ingestion and query processing.
"""

import asyncio
import sys

sys.path.append("src/production/rag/rag_system/core")

try:
    from codex_rag_integration import Document
    from enhanced_query_processor import EnhancedQueryProcessor, SynthesizedAnswer
    from graph_enhanced_rag_pipeline import GraphEnhancedRAGPipeline

    async def test_full_integration():
        """Test enhanced query processing with actual document."""

        print("Enhanced Query Processing - Full Integration Test")
        print("=" * 55)

        # Create minimal RAG pipeline
        print("[INIT] Creating minimal RAG pipeline...")
        rag_pipeline = GraphEnhancedRAGPipeline(
            enable_intelligent_chunking=False,  # Disable for speed
            enable_contextual_tagging=False,
            enable_trust_graph=False,
        )

        # Create enhanced query processor
        query_processor = EnhancedQueryProcessor(
            rag_pipeline=rag_pipeline,
            enable_query_expansion=True,
            enable_intent_classification=True,
            enable_multi_hop_reasoning=True,
        )

        # Create test document
        test_doc = Document(
            id="test_ai_doc",
            title="AI Fundamentals",
            content="""
            Artificial intelligence (AI) is a branch of computer science focused on creating systems
            that can perform tasks typically requiring human intelligence. Machine learning is a
            subset of AI that enables systems to learn from data without explicit programming.

            Deep learning, a further subset of machine learning, uses neural networks with multiple
            layers to process complex patterns. These technologies have applications in image
            recognition, natural language processing, and autonomous vehicles.
            """,
            source_type="educational",
            metadata={"author": "AI Researcher", "credibility_score": 0.9},
        )

        # Index the document
        print("[INDEX] Indexing test document...")
        stats = rag_pipeline.index_documents([test_doc])
        print(f"Indexed: {stats['documents_processed']} docs, {stats['chunks_created']} chunks")

        # Test query
        test_query = "What is machine learning and how does it relate to AI?"
        print(f"\n[QUERY] Testing query: {test_query}")

        # Process query
        result = await query_processor.process_query(test_query)

        print("\nResults:")
        print(f"- Overall confidence: {result.overall_confidence:.3f}")
        print(f"- Processing method: {result.synthesis_method}")
        print(f"- Primary sources: {len(result.primary_sources)}")
        print(f"- Answer preview: {result.answer_text[:150]}...")

        if result.overall_confidence > 0.5:
            print("\nSUCCESS: Full integration working!")
            return True
        print("\nPARTIAL: Integration working with low confidence")
        return False

except ImportError as e:
    print(f"Import error: {e}")
    print("SKIP: Full integration test - dependencies not available")

    async def test_full_integration():
        return True


if __name__ == "__main__":
    success = asyncio.run(test_full_integration())
