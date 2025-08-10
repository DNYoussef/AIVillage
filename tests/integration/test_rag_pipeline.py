import pytest

from experimental.rag.storm.wikipedia_storm_pipeline import (
    WikipediaSTORMPipeline,
)


def test_rag_pipeline_integration():
    # Create a RAG pipeline with a minimal dataset
    data = [{"title": "France", "text": "France is a country in Europe."}]
    rag_pipeline = WikipediaSTORMPipeline(dataset=data)

    # Use a simple method to verify basic functionality
    related = rag_pipeline.find_related_articles("France", k=1)

    # Check that the pipeline returns expected related article
    assert related == ["France"]
