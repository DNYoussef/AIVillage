from src.production.rag.rag_system.core.offline_defaults import (
    OfflineRAGPipeline,
)


def test_offline_rag_top1_accuracy():
    corpus = [
        {"text": "Apples are red fruits.", "metadata": {"title": "apple"}},
        {"text": "The sky is blue and clear.", "metadata": {"title": "sky"}},
        {"text": "Bananas are yellow and sweet.", "metadata": {"title": "banana"}},
    ]
    rag = OfflineRAGPipeline(load_builtin=False)
    for doc in corpus:
        rag.vector_store.add_document(doc["text"], doc["metadata"])
    result = rag.query("What color is the sky?", top_k=1)
    assert result["sources"][0]["metadata"]["title"] == "sky"
