"""Local mode method additions for RAG pipeline."""

# This file contains method patches for RAGPipeline to support local mode
# These should be integrated into the main pipeline.py file


def add_documents(self, docs):
    """Add documents - works in both local and normal mode."""
    if self.is_local:
        self.local_rag.add_documents(docs)
        self.metrics["documents_indexed"] += len(docs)
        return

    # Normal mode
    import asyncio

    for doc in docs:
        if isinstance(doc, dict):
            doc_obj = Document(
                id=doc.get("id", str(self._document_count)),
                text=doc.get("content", doc.get("text", "")),
                metadata=doc.get("metadata", {}),
            )
        else:
            doc_obj = doc
        asyncio.run(self.add_document(doc_obj))


def retrieve(self, query, top_k=5, **kwargs):
    """Retrieve documents - works in both local and normal mode."""
    if self.is_local:
        import time

        start = time.time()
        results = self.local_rag.retrieve(query, top_k)
        latency_ms = (time.time() - start) * 1000
        self.metrics["queries_processed"] += 1
        self.metrics["avg_query_time_ms"] = (
            self.metrics["avg_query_time_ms"] * (self.metrics["queries_processed"] - 1)
            + latency_ms
        ) / self.metrics["queries_processed"]
        return results

    # Normal mode - run async retrieve
    import asyncio

    return asyncio.run(self.retrieve_async(query, top_k, **kwargs))


def query(self, query, top_k=5, **kwargs):
    """Query alias for retrieve."""
    return self.retrieve(query, top_k, **kwargs)


def search(self, query, top_k=5, **kwargs):
    """Search alias for retrieve."""
    return self.retrieve(query, top_k, **kwargs)


def index(self, content, metadata=None):
    """Index a single document."""
    doc = {
        "id": f"doc_{self.metrics['documents_indexed']}",
        "content": content,
        "metadata": metadata or {},
    }
    self.add_documents([doc])


def ingest(self, docs):
    """Ingest documents alias."""
    self.add_documents(docs)
