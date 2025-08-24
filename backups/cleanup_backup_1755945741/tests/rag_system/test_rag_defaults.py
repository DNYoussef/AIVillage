import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from packages.rag.rag_system.core.pipeline import Document, RAGPipeline


async def setup_pipeline():
    pipeline = RAGPipeline()
    await pipeline.add_document(Document(id="1", text="A test document."))
    return pipeline


def test_rag_pipeline_default():
    pipeline = asyncio.run(setup_pipeline())
    results = asyncio.run(pipeline.retrieve("test"))
    assert len(results) >= 1
