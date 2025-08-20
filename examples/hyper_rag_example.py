"""Example usage of the HyperRAG orchestrator.

This script demonstrates how to create a fully enabled HyperRAG instance,
ingest a document and run a few queries using different `QueryMode` values.
"""

from __future__ import annotations

import asyncio

from packages.rag.core.hyper_rag import (
    MemoryType,
    QueryMode,
    create_hyper_rag,
)


async def main() -> None:
    """Run a small HyperRAG demo."""

    # Create system with all components enabled.
    rag = await create_hyper_rag(enable_all=True, fog_computing=True)

    # Store a sample document.
    await rag.store_document(
        content="""
        Machine learning is a subset of artificial intelligence that enables
        computers to learn and improve from experience without being explicitly
        programmed. Deep learning uses neural networks with multiple layers to
        automatically discover representations from data.
        """,
        title="Introduction to Machine Learning",
        memory_type=MemoryType.ALL,
    )

    # Run a few queries across different processing modes.
    queries = [
        ("What is machine learning?", QueryMode.FAST),
        ("How does deep learning work?", QueryMode.BALANCED),
        (
            "Explain AI and machine learning relationships",
            QueryMode.COMPREHENSIVE,
        ),
    ]

    for query, mode in queries:
        print(f"\nQuery: {query} (Mode: {mode.value})")
        result = await rag.query(query, mode=mode)
        print(f"Answer: {result.synthesized_answer.answer}")
        print(f"Confidence: {result.confidence_score:.3f}")
        print(f"Latency: {result.total_latency_ms:.1f}ms")
        print(f"Systems: {result.systems_used}")

    # Show system status.
    status = await rag.get_system_status()
    print(f"\nSystem Status: {status['statistics']}")

    await rag.close()


if __name__ == "__main__":
    asyncio.run(main())

