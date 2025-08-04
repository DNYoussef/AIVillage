# rag_system/utils/hippo_cache.py

from datetime import datetime
from typing import Any


class HippoCache:
    def __init__(self, max_size: int = 1000) -> None:
        self.cache = {}
        self.max_size = max_size

    def add(self, key: str, value: Any, timestamp: datetime) -> None:
        if len(self.cache) >= self.max_size:
            self._evict()
        self.cache[key] = {"value": value, "timestamp": timestamp, "frequency": 1}

    def get(self, key: str) -> Any | None:
        if key in self.cache:
            self.cache[key]["frequency"] += 1
            return self.cache[key]["value"]
        return None

    def _evict(self) -> None:
        # Implement eviction strategy based on frequency and recency
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: (x[1]["frequency"], x[1]["timestamp"]),
            reverse=True,
        )
        self.cache = dict(sorted_items[: self.max_size - 1])
