import pytest

from rag_system.wikipedia_storm_pipeline import WikipediaSTORMPipeline


@pytest.mark.asyncio
async def test_rag_pipeline_integration():
    # Create a RAG pipeline
    rag_pipeline = WikipediaSTORMPipeline()

    # Process a query
    results = await rag_pipeline.process_query("What is the capital of France?")

    # Check that the results are correct
    assert "Paris" in results
