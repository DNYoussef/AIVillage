import asyncio
import sys

import numpy as np

sys.modules.pop("rag_system", None)
sys.modules.pop("rag_system.wikipedia_storm_pipeline", None)

from rag_system.wikipedia_storm_pipeline import (
    ContentDatabase,
    EducationalContentGenerator,
    OfflineOptimizedRAG,
    WikipediaSTORMPipeline,
)


async def run_pipeline(pipeline: WikipediaSTORMPipeline) -> None:
    await pipeline.process_wikipedia_for_education()


def test_wikipedia_storm_rag():
    dataset = [
        {"title": "History of Science", "text": "History and Science content for education."},
        {"title": "Sports", "text": "General sports article"},
    ]
    pipeline = WikipediaSTORMPipeline(dataset=dataset)
    asyncio.run(run_pipeline(pipeline))
    assert pipeline.get_processed_content("History of Science") is not None

    embeddings = np.random.rand(10, 32).astype("float32")
    metadata = [{"title": f"Article {i}"} for i in range(10)]
    db = ContentDatabase(embeddings=embeddings, metadata=metadata)
    rag = OfflineOptimizedRAG(max_size_mb=1)
    package = rag.prepare_for_mobile(db)
    assert package["size_bytes"] < rag.max_size

    generator = EducationalContentGenerator(pipeline)
    lesson = generator.generate_lesson(
        topic="History of Science",
        grade_level=3,
        cultural_context={"language": "en", "region": "global"},
    )
    assert lesson.topic == "History of Science"
    assert lesson.grade_level == 3
