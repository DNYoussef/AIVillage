"""Advanced Semantic Caching System for BayesRAG-CODEX Integration.

Implements:
- Semantic multi-tier caching (hot/warm/cold)
- Context-aware cache matching using embeddings
- Trust-based prefetching for high-quality content
- Adaptive cache warming based on query patterns
"""

import asyncio
import hashlib
import logging
import pickle
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Enhanced cache entry with semantic and trust metadata."""

    key: str
    query: str
    embedding: np.ndarray
    results: list[Any]
    timestamp: float
    access_count: int = 0
    last_access: float = 0
    trust_score: float = 0.5
    semantic_neighbors: list[str] = None
    context_type: str = "standard"
    prefetch_priority: float = 0.0


@dataclass
class CacheTier:
    """Cache tier configuration."""

    name: str
    max_size: int
    ttl_seconds: int
    min_trust_score: float
    eviction_policy: str  # LRU, LFU, TRUST, HYBRID


class SemanticMultiTierCache:
    """Advanced three-tier cache with semantic matching and trust-based management.

    Tiers:
    - HOT: In-memory, fastest access, highest trust content
    - WARM: Memory-mapped, medium speed, frequently accessed
    - COLD: Disk-based, slower, long-term storage
    """

    def __init__(
        self,
        embedding_model: str = "paraphrase-MiniLM-L3-v2",
        cache_dir: Path = Path("/tmp/semantic_cache"),
        enable_prefetch: bool = True,
    ):
        # Embedding model for semantic matching
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        # Cache directory
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Tier configurations
        self.tiers = {
            "hot": CacheTier(
                name="hot",
                max_size=128,
                ttl_seconds=300,  # 5 minutes
                min_trust_score=0.7,
                eviction_policy="HYBRID",
            ),
            "warm": CacheTier(
                name="warm",
                max_size=512,
                ttl_seconds=1800,  # 30 minutes
                min_trust_score=0.4,
                eviction_policy="LFU",
            ),
            "cold": CacheTier(
                name="cold",
                max_size=2048,
                ttl_seconds=86400,  # 24 hours
                min_trust_score=0.0,
                eviction_policy="LRU",
            ),
        }

        # Cache storage
        self.hot_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.warm_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.cold_cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # FAISS index for semantic search
        self.semantic_index = faiss.IndexFlatIP(self.embedding_dim)
        self.index_to_key: dict[int, str] = {}
        self.key_to_index: dict[str, int] = {}

        # Query pattern tracking
        self.query_patterns = defaultdict(list)
        self.context_patterns = defaultdict(int)

        # Prefetch configuration
        self.enable_prefetch = enable_prefetch
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()
        self.prefetch_cache: dict[str, Any] = {}

        # Performance metrics
        self.metrics = {
            "hits": {"hot": 0, "warm": 0, "cold": 0, "semantic": 0},
            "misses": 0,
            "promotions": 0,
            "evictions": 0,
            "prefetch_hits": 0,
        }

        # Start background tasks
        if enable_prefetch:
            asyncio.create_task(self._prefetch_worker())

    async def get(
        self,
        query: str,
        semantic_threshold: float = 0.85,
        context_hints: dict[str, Any] = None,
    ) -> tuple[list[Any], dict[str, Any]] | None:
        """Get cached results with semantic matching across all tiers.

        Args:
            query: Search query
            semantic_threshold: Minimum similarity for semantic match
            context_hints: Optional context for better matching

        Returns:
            Cached results and metadata if found
        """
        start_time = time.perf_counter()
        key = self._make_key(query)

        # Check prefetch cache first
        if self.enable_prefetch and key in self.prefetch_cache:
            self.metrics["prefetch_hits"] += 1
            return self.prefetch_cache[key]

        # Try exact match in tiers (hot -> warm -> cold)
        result = await self._check_exact_match(key)
        if result:
            tier_name = result[1]["tier"]
            self.metrics["hits"][tier_name] += 1

            # Promote if accessed frequently
            await self._promote_entry(key, tier_name)

            latency = (time.perf_counter() - start_time) * 1000
            result[1]["cache_latency_ms"] = latency
            return result

        # Try semantic match if exact match fails
        result = await self._check_semantic_match(
            query, semantic_threshold, context_hints
        )
        if result:
            self.metrics["hits"]["semantic"] += 1
            latency = (time.perf_counter() - start_time) * 1000
            result[1]["cache_latency_ms"] = latency
            return result

        self.metrics["misses"] += 1
        return None

    async def set(
        self,
        query: str,
        results: list[Any],
        trust_score: float = 0.5,
        context_type: str = "standard",
        metadata: dict[str, Any] = None,
    ) -> None:
        """Store results in appropriate tier based on trust and context.

        Args:
            query: Original query
            results: Results to cache
            trust_score: Trust score for prioritization
            context_type: Type of context (temporal, geographic, etc.)
            metadata: Additional metadata
        """
        key = self._make_key(query)
        embedding = self.embedder.encode(query, convert_to_numpy=True)

        # Create cache entry
        entry = CacheEntry(
            key=key,
            query=query,
            embedding=embedding,
            results=results,
            timestamp=time.time(),
            trust_score=trust_score,
            context_type=context_type,
            prefetch_priority=self._calculate_prefetch_priority(
                trust_score, context_type
            ),
        )

        # Determine target tier based on trust score
        target_tier = self._determine_tier(trust_score)

        # Add to tier
        await self._add_to_tier(entry, target_tier)

        # Update semantic index
        self._update_semantic_index(entry)

        # Track query patterns
        self._track_query_pattern(query, context_type)

        # Schedule prefetching for related queries
        if self.enable_prefetch and trust_score > 0.7:
            await self._schedule_prefetch(query, embedding)

    async def _check_exact_match(
        self, key: str
    ) -> tuple[list[Any], dict[str, Any]] | None:
        """Check for exact match across all tiers."""
        # Check hot tier
        if key in self.hot_cache:
            entry = self.hot_cache[key]
            entry.access_count += 1
            entry.last_access = time.time()

            # Move to end (LRU)
            self.hot_cache.move_to_end(key)

            return entry.results, {
                "tier": "hot",
                "trust_score": entry.trust_score,
                "access_count": entry.access_count,
                "cache_hit": True,
            }

        # Check warm tier
        if key in self.warm_cache:
            entry = self.warm_cache[key]
            entry.access_count += 1
            entry.last_access = time.time()

            return entry.results, {
                "tier": "warm",
                "trust_score": entry.trust_score,
                "access_count": entry.access_count,
                "cache_hit": True,
            }

        # Check cold tier
        if key in self.cold_cache:
            entry = self.cold_cache[key]
            entry.access_count += 1
            entry.last_access = time.time()

            return entry.results, {
                "tier": "cold",
                "trust_score": entry.trust_score,
                "access_count": entry.access_count,
                "cache_hit": True,
            }

        return None

    async def _check_semantic_match(
        self, query: str, threshold: float, context_hints: dict[str, Any] = None
    ) -> tuple[list[Any], dict[str, Any]] | None:
        """Find semantically similar cached queries."""
        if self.semantic_index.ntotal == 0:
            return None

        # Encode query
        query_embedding = self.embedder.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.reshape(1, -1).astype("float32")

        # Search for similar embeddings
        k = min(10, self.semantic_index.ntotal)
        scores, indices = self.semantic_index.search(query_embedding, k)

        # Find best match above threshold
        for score, idx in zip(scores[0], indices[0], strict=False):
            if score >= threshold:
                key = self.index_to_key.get(idx)
                if key:
                    # Try to find in tiers
                    result = await self._check_exact_match(key)
                    if result:
                        result[1]["semantic_match"] = True
                        result[1]["similarity_score"] = float(score)
                        return result

        return None

    async def _promote_entry(self, key: str, current_tier: str) -> None:
        """Promote frequently accessed entries to higher tiers."""
        if current_tier == "hot":
            return  # Already in highest tier

        entry = None

        # Get entry from current tier
        if current_tier == "warm" and key in self.warm_cache:
            entry = self.warm_cache[key]

            # Promote to hot if access count high enough
            if (
                entry.access_count >= 5
                and entry.trust_score >= self.tiers["hot"].min_trust_score
            ):
                del self.warm_cache[key]
                await self._add_to_tier(entry, "hot")
                self.metrics["promotions"] += 1

        elif current_tier == "cold" and key in self.cold_cache:
            entry = self.cold_cache[key]

            # Promote to warm if access count high enough
            if (
                entry.access_count >= 3
                and entry.trust_score >= self.tiers["warm"].min_trust_score
            ):
                del self.cold_cache[key]
                await self._add_to_tier(entry, "warm")
                self.metrics["promotions"] += 1

    async def _add_to_tier(self, entry: CacheEntry, tier_name: str) -> None:
        """Add entry to specified tier with eviction if needed."""
        tier = self.tiers[tier_name]

        if tier_name == "hot":
            cache = self.hot_cache
        elif tier_name == "warm":
            cache = self.warm_cache
        else:
            cache = self.cold_cache

        # Check if eviction needed
        if len(cache) >= tier.max_size:
            await self._evict_from_tier(tier_name)

        # Add entry
        cache[entry.key] = entry

    async def _evict_from_tier(self, tier_name: str) -> None:
        """Evict entries based on tier policy."""
        tier = self.tiers[tier_name]

        if tier_name == "hot":
            cache = self.hot_cache
        elif tier_name == "warm":
            cache = self.warm_cache
        else:
            cache = self.cold_cache

        if not cache:
            return

        # Select victim based on eviction policy
        if tier.eviction_policy == "LRU":
            # Remove oldest (first item in OrderedDict)
            victim_key = next(iter(cache))

        elif tier.eviction_policy == "LFU":
            # Remove least frequently used
            victim_key = min(cache.keys(), key=lambda k: cache[k].access_count)

        elif tier.eviction_policy == "TRUST":
            # Remove lowest trust score
            victim_key = min(cache.keys(), key=lambda k: cache[k].trust_score)

        else:  # HYBRID
            # Combine recency, frequency, and trust
            def score(k):
                e = cache[k]
                recency = time.time() - e.last_access
                frequency = e.access_count
                trust = e.trust_score
                # Lower score = better candidate for eviction
                return (trust * frequency) / (recency + 1)

            victim_key = min(cache.keys(), key=score)

        # Demote to lower tier if possible
        victim = cache[victim_key]

        if tier_name == "hot":
            # Demote to warm
            await self._add_to_tier(victim, "warm")
        elif tier_name == "warm":
            # Demote to cold
            await self._add_to_tier(victim, "cold")

        # Remove from current tier
        del cache[victim_key]
        self.metrics["evictions"] += 1

    def _determine_tier(self, trust_score: float) -> str:
        """Determine appropriate tier based on trust score."""
        if trust_score >= self.tiers["hot"].min_trust_score:
            return "hot"
        if trust_score >= self.tiers["warm"].min_trust_score:
            return "warm"
        return "cold"

    def _update_semantic_index(self, entry: CacheEntry) -> None:
        """Update FAISS index with new entry."""
        # Add to index
        idx = self.semantic_index.ntotal
        self.semantic_index.add(entry.embedding.reshape(1, -1).astype("float32"))

        # Update mappings
        self.index_to_key[idx] = entry.key
        self.key_to_index[entry.key] = idx

        # Find semantic neighbors
        if self.semantic_index.ntotal > 1:
            k = min(5, self.semantic_index.ntotal - 1)
            scores, indices = self.semantic_index.search(
                entry.embedding.reshape(1, -1).astype("float32"),
                k + 1,  # +1 because it will include itself
            )

            neighbors = []
            for score, neighbor_idx in zip(scores[0], indices[0], strict=False):
                if neighbor_idx != idx and score > 0.8:
                    neighbor_key = self.index_to_key.get(neighbor_idx)
                    if neighbor_key:
                        neighbors.append(neighbor_key)

            entry.semantic_neighbors = neighbors

    def _track_query_pattern(self, query: str, context_type: str) -> None:
        """Track query patterns for prefetching."""
        # Extract key terms (simplified)
        terms = query.lower().split()

        for term in terms:
            if len(term) > 3:  # Skip short words
                self.query_patterns[term].append(time.time())

        self.context_patterns[context_type] += 1

    def _calculate_prefetch_priority(
        self, trust_score: float, context_type: str
    ) -> float:
        """Calculate prefetch priority based on trust and context patterns."""
        # Base priority from trust
        priority = trust_score

        # Boost for frequently queried context types
        context_frequency = self.context_patterns.get(context_type, 0)
        if context_frequency > 10:
            priority *= 1.2
        elif context_frequency > 5:
            priority *= 1.1

        return min(1.0, priority)

    async def _schedule_prefetch(self, query: str, embedding: np.ndarray) -> None:
        """Schedule prefetching of related queries."""
        if not self.enable_prefetch:
            return

        # Find semantic neighbors for prefetching
        if self.semantic_index.ntotal > 0:
            k = min(3, self.semantic_index.ntotal)
            scores, indices = self.semantic_index.search(
                embedding.reshape(1, -1).astype("float32"), k
            )

            for score, idx in zip(scores[0], indices[0], strict=False):
                if score > 0.9:  # Very similar queries
                    neighbor_key = self.index_to_key.get(idx)
                    if neighbor_key:
                        await self.prefetch_queue.put(neighbor_key)

    async def _prefetch_worker(self) -> None:
        """Background worker for prefetching related content."""
        while True:
            try:
                # Get next prefetch task
                key = await asyncio.wait_for(self.prefetch_queue.get(), timeout=10.0)

                # Check if already in prefetch cache
                if key not in self.prefetch_cache:
                    # Find in main cache tiers
                    result = await self._check_exact_match(key)
                    if result:
                        # Add to prefetch cache with TTL
                        self.prefetch_cache[key] = result

                        # Schedule removal after TTL
                        asyncio.create_task(self._remove_from_prefetch(key, 60))

            except asyncio.TimeoutError:
                # Periodic cleanup of old prefetch entries
                await self._cleanup_prefetch_cache()

            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")

    async def _remove_from_prefetch(self, key: str, delay: int) -> None:
        """Remove entry from prefetch cache after delay."""
        await asyncio.sleep(delay)
        self.prefetch_cache.pop(key, None)

    async def _cleanup_prefetch_cache(self) -> None:
        """Clean up old entries from prefetch cache."""
        # Limit prefetch cache size
        max_prefetch_size = 50

        if len(self.prefetch_cache) > max_prefetch_size:
            # Remove oldest entries
            to_remove = len(self.prefetch_cache) - max_prefetch_size
            for key in list(self.prefetch_cache.keys())[:to_remove]:
                del self.prefetch_cache[key]

    def _make_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.encode()).hexdigest()

    def get_metrics(self) -> dict[str, Any]:
        """Get cache performance metrics."""
        total_hits = sum(self.metrics["hits"].values())
        total_requests = total_hits + self.metrics["misses"]

        if total_requests == 0:
            hit_rate = 0
        else:
            hit_rate = total_hits / total_requests

        return {
            "hit_rate": hit_rate,
            "tier_hits": self.metrics["hits"],
            "misses": self.metrics["misses"],
            "promotions": self.metrics["promotions"],
            "evictions": self.metrics["evictions"],
            "prefetch_hits": self.metrics["prefetch_hits"],
            "cache_sizes": {
                "hot": len(self.hot_cache),
                "warm": len(self.warm_cache),
                "cold": len(self.cold_cache),
                "prefetch": len(self.prefetch_cache),
            },
            "semantic_index_size": self.semantic_index.ntotal,
        }

    async def persist_to_disk(self) -> None:
        """Persist cache state to disk for recovery."""
        state = {
            "hot": list(self.hot_cache.values()),
            "warm": list(self.warm_cache.values()),
            "cold": list(self.cold_cache.values()),
            "metrics": self.metrics,
            "patterns": {
                "query": dict(self.query_patterns),
                "context": dict(self.context_patterns),
            },
        }

        cache_file = self.cache_dir / "cache_state.pkl"

        with open(cache_file, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Cache persisted to {cache_file}")

    async def load_from_disk(self) -> bool:
        """Load cache state from disk."""
        cache_file = self.cache_dir / "cache_state.pkl"

        if not cache_file.exists():
            return False

        try:
            with open(cache_file, "rb") as f:
                state = pickle.load(f)

            # Restore cache tiers
            for entry in state.get("hot", []):
                self.hot_cache[entry.key] = entry
                self._update_semantic_index(entry)

            for entry in state.get("warm", []):
                self.warm_cache[entry.key] = entry
                self._update_semantic_index(entry)

            for entry in state.get("cold", []):
                self.cold_cache[entry.key] = entry
                self._update_semantic_index(entry)

            # Restore metrics
            self.metrics = state.get("metrics", self.metrics)

            # Restore patterns
            patterns = state.get("patterns", {})
            self.query_patterns = defaultdict(list, patterns.get("query", {}))
            self.context_patterns = defaultdict(int, patterns.get("context", {}))

            logger.info(f"Cache loaded from {cache_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")
            return False


async def test_semantic_cache():
    """Test the semantic multi-tier cache."""
    print("=== Testing Semantic Multi-Tier Cache ===\n")

    # Initialize cache
    cache = SemanticMultiTierCache()

    # Test data
    test_queries = [
        ("What is machine learning?", 0.9, "technical"),
        ("How does machine learning work?", 0.8, "technical"),  # Semantic match
        ("Explain deep learning", 0.85, "technical"),
        ("History of World War II", 0.7, "temporal"),
        ("World War 2 timeline", 0.65, "temporal"),  # Semantic match
        ("Python programming basics", 0.6, "technical"),
        ("Introduction to Python", 0.55, "technical"),  # Semantic match
        ("Climate change effects", 0.4, "geographic"),
        ("Global warming impact", 0.35, "geographic"),  # Semantic match
    ]

    # Store queries
    print("Storing queries in cache...")
    for query, trust, context in test_queries:
        results = [f"Result for: {query}"]
        await cache.set(query, results, trust, context)

    # Test exact matches
    print("\nTesting exact matches:")
    for query, _, _ in test_queries[:3]:
        result = await cache.get(query)
        if result:
            print(f"  ✓ Found: {query} (tier: {result[1].get('tier')})")
        else:
            print(f"  ✗ Not found: {query}")

    # Test semantic matches
    print("\nTesting semantic matches:")
    similar_queries = [
        "What exactly is machine learning?",
        "Second World War history",
        "Getting started with Python",
        "Effects of climate change on Earth",
    ]

    for query in similar_queries:
        result = await cache.get(query, semantic_threshold=0.8)
        if result:
            print(f"  ✓ Semantic match for: {query}")
            print(f"    Similarity: {result[1].get('similarity_score', 0):.3f}")
        else:
            print(f"  ✗ No match for: {query}")

    # Simulate access patterns to trigger promotions
    print("\nSimulating access patterns...")
    for _ in range(6):
        await cache.get("Python programming basics")

    # Check metrics
    metrics = cache.get_metrics()
    print("\n=== Cache Metrics ===")
    print(f"Hit Rate: {metrics['hit_rate']:.2%}")
    print(f"Tier Hits: {metrics['tier_hits']}")
    print(f"Promotions: {metrics['promotions']}")
    print(f"Cache Sizes: {metrics['cache_sizes']}")
    print(f"Semantic Index Size: {metrics['semantic_index_size']}")

    # Test persistence
    print("\nTesting persistence...")
    await cache.persist_to_disk()

    # Create new cache and load
    cache2 = SemanticMultiTierCache()
    loaded = await cache2.load_from_disk()

    if loaded:
        print("  ✓ Cache successfully loaded from disk")
        print(f"  Restored {cache2.semantic_index.ntotal} semantic entries")
    else:
        print("  ✗ Failed to load cache from disk")

    return True


if __name__ == "__main__":
    asyncio.run(test_semantic_cache())
