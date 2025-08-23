import pytest
from production.rag.rag_system.core.pipeline import Document, RAGPipeline


@pytest.mark.asyncio
async def test_simple_rag_cycle():
    pipeline = RAGPipeline(enable_cache=False, enable_graph=False)
    await pipeline.add_document(Document(id="1", text="sky is blue"))
    await pipeline.add_document(Document(id="2", text="grass is green"))

    answer = await pipeline.run_basic_rag("sky is blue", top_k=1)
    assert "sky is blue" in answer
