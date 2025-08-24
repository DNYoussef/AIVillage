from collections import OrderedDict
import time


class LRUCache:
    """Simple LRU cache used to mimic RAG caching behaviour."""

    def __init__(self, capacity: int = 1) -> None:
        self.capacity = capacity
        self.store: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key]
        return None

    def set(self, key: str, value: str) -> None:
        if key in self.store:
            self.store.move_to_end(key)
        self.store[key] = value
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)


class RAGService:
    def __init__(self, cache: LRUCache) -> None:
        self.cache = cache

    def query(self, q: str) -> tuple[str, bool]:
        cached = self.cache.get(q)
        if cached is not None:
            return cached, True
        # simulate retrieval cost
        time.sleep(0.001)
        answer = q.upper()
        self.cache.set(q, answer)
        return answer, False


def test_rag_cache_flow() -> None:
    cache = LRUCache(capacity=1)
    rag = RAGService(cache)

    start = time.perf_counter()
    ans, hit = rag.query("hello")
    cold_latency = time.perf_counter() - start
    assert not hit
    assert ans == "HELLO"

    start = time.perf_counter()
    ans2, hit2 = rag.query("hello")
    warm_latency = time.perf_counter() - start
    assert hit2
    assert ans2 == "HELLO"
    # warm cache should be very fast
    assert warm_latency < 0.01

    # cache eviction when different query fills single-entry cache
    rag.query("world")
    ans3, hit3 = rag.query("hello")
    assert not hit3
    assert cold_latency > warm_latency
