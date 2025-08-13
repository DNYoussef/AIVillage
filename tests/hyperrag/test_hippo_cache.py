import asyncio
from dataclasses import dataclass
from datetime import datetime
import threading
import time

import numpy as np

from hyperrag.hippo_cache import CacheEntry, HippoCache


@dataclass
class Document:
    text: str


def _make_entry(embedding: np.ndarray) -> CacheEntry:
    return CacheEntry(
        query_embedding=embedding,
        retrieved_docs=[Document("doc")],
        relevance_scores=[1.0],
        citation_metadata={},
        timestamp=datetime.utcnow(),
    )


def test_lru_and_ttl() -> None:
    cache = HippoCache(max_size=2, ttl_hours=0.0001)  # ~0.36s TTL
    e1 = np.random.rand(64).astype("float32")
    e2 = np.random.rand(64).astype("float32")
    e3 = np.random.rand(64).astype("float32")
    cache.set("a", _make_entry(e1))
    cache.set("b", _make_entry(e2))
    cache.set("c", _make_entry(e3))  # evicts "a"
    assert cache.get(e1) is None
    cache.set("d", _make_entry(e1))
    time.sleep(0.5)  # expire
    assert cache.get(e1) is None


def test_thread_safety() -> None:
    cache = HippoCache(max_size=100)
    emb = np.random.rand(64).astype("float32")
    entry = _make_entry(emb)

    def worker(i: int) -> None:
        cache.set(f"k{i}", entry)
        cache.get(emb)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cache.metrics()["size"] >= 1


def test_async_warm_cache() -> None:
    cache = HippoCache(max_size=10)
    emb = np.random.rand(64).astype("float32")
    entry = _make_entry(emb)

    async def warm() -> None:
        await cache.warm_cache([("a", entry)])

    asyncio.run(warm())
    assert cache.get(emb) is not None


def test_semantic_similarity() -> None:
    cache = HippoCache(max_size=10)
    emb = np.random.rand(128).astype("float32")
    cache.set("a", _make_entry(emb))
    near = emb + np.random.normal(0, 0.001, emb.shape).astype("float32")
    assert cache.get(near) is not None


def test_metrics_hit_rate() -> None:
    cache = HippoCache(max_size=10)
    emb = np.random.rand(64).astype("float32")
    cache.set("a", _make_entry(emb))
    for _ in range(4):
        cache.get(emb)
    cache.get(np.random.rand(64).astype("float32"))  # miss
    assert cache.metrics()["hit_rate"] >= 0.6


def test_get_or_retrieve_latency() -> None:
    cache = HippoCache(max_size=10)
    emb = np.random.rand(64).astype("float32")

    def retrieve():
        return [Document("x")], [0.1], {}

    cache.get_or_retrieve("a", emb, retrieve)
    start = time.perf_counter()
    cache.get_or_retrieve("a", emb, retrieve)
    elapsed = (time.perf_counter() - start) * 1000
    assert elapsed < 10
