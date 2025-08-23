import time
from datetime import datetime

import numpy as np
from hyperrag.hippo_cache import CacheEntry, HippoCache


def test_cache_hit_latency_benchmark() -> None:
    cache = HippoCache(max_size=1000)
    emb = np.random.rand(64).astype("float32")
    entry = CacheEntry(
        query_embedding=emb,
        retrieved_docs=[],
        relevance_scores=[],
        citation_metadata={},
        timestamp=datetime.utcnow(),
    )
    cache.set("a", entry)
    start = time.perf_counter()
    for _ in range(100):
        assert cache.get(emb) is not None
    avg = (time.perf_counter() - start) * 1000 / 100
    assert avg < 10
