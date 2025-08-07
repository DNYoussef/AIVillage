# RAG High-Performance Cache Architecture Specification

## Executive Summary

The AIVillage RAG system is 48% complete with a functional pipeline but lacks caching, resulting in query latencies exceeding requirements. This specification designs a multi-tier caching layer targeting <10ms latency for cached queries, with intelligent eviction, semantic similarity matching, and future credit system integration.

## Current State Analysis

### Existing RAG Query Flow

```
1. Query Entry (EnhancedRAGPipeline.process)
   ├─> Latent Space Activation (~50-100ms)
   ├─> Hybrid Retrieval (~200-500ms)
   │   ├─> Low-level retrieval (GraphStore)
   │   └─> High-level retrieval (VectorStore with FAISS/Qdrant)
   ├─> Reasoning Engine (~100-200ms)
   └─> Cognitive Integration (~50-100ms)

Total Latency: 400-900ms (uncached)
```

### Identified Bottlenecks
- **Vector Search**: FAISS/Qdrant queries take 100-300ms
- **Graph Traversal**: NetworkX operations scale poorly
- **Embedding Generation**: 50-100ms per query
- **No Result Caching**: Every query recomputes full pipeline

### Existing Cache Infrastructure
- **HippoCache**: Basic LRU implementation (1000 items max)
- **No semantic matching**: Exact key matching only
- **No persistence**: In-memory only
- **No metrics**: No hit/miss tracking

## Proposed Multi-Tier Cache Architecture

### 1. Cache Hierarchy

```python
class MultiTierRAGCache:
    """Three-tier caching system for RAG queries."""

    def __init__(self, config: CacheConfig):
        # L1: Hot cache - exact matches (<1ms)
        self.l1_cache = ExactMatchCache(
            max_size_mb=512,
            max_items=10000,
            ttl_seconds=3600
        )

        # L2: Warm cache - semantic similarity (<5ms)
        self.l2_cache = SemanticCache(
            max_size_mb=2048,
            max_items=50000,
            similarity_threshold=0.95,
            index_type="HNSW"  # Hierarchical Navigable Small World
        )

        # L3: Cold cache - compressed historical (<10ms)
        self.l3_cache = CompressedCache(
            max_size_mb=8192,
            compression="lz4",
            index_type="IVF",  # Inverted File Index
            persistence_path="cache/rag_cache.db"
        )
```

### 2. Cache Key Strategy

#### Primary Key Generation
```python
@dataclass
class CacheKey:
    """Composite cache key for RAG queries."""

    # Core components
    query_hash: str  # SHA256 of normalized query
    embedding_hash: str  # Hash of query embedding (first 64 dims)

    # Context components
    timestamp_bucket: int  # Hour-level bucket for temporal queries
    user_context: str | None  # User/session identifier
    knowledge_version: int  # Knowledge base version

    # Metadata for eviction
    access_count: int = 0
    last_access: float = 0.0
    creation_time: float = 0.0
    citation_required: bool = False

    def to_string(self) -> str:
        """Generate deterministic string key."""
        components = [
            self.query_hash[:16],
            self.embedding_hash[:8],
            str(self.timestamp_bucket),
            self.knowledge_version
        ]
        if self.user_context:
            components.append(self.user_context[:8])
        return ":".join(components)

    def semantic_signature(self) -> np.ndarray:
        """Generate semantic signature for similarity matching."""
        # Use first 128 dimensions of embedding for fast comparison
        return self.embedding_vector[:128]
```

#### Query Normalization
```python
class QueryNormalizer:
    """Normalize queries for better cache hits."""

    def normalize(self, query: str) -> str:
        # Lowercase and strip
        normalized = query.lower().strip()

        # Remove punctuation except necessary
        normalized = re.sub(r'[^\w\s\-\']', ' ', normalized)

        # Normalize whitespace
        normalized = ' '.join(normalized.split())

        # Expand common contractions
        contractions = {
            "what's": "what is",
            "won't": "will not",
            "can't": "cannot",
            # ... more contractions
        }
        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)

        # Sort question words to handle variations
        if normalized.startswith(('what', 'where', 'when', 'who', 'why', 'how')):
            parts = normalized.split(' ', 1)
            if len(parts) > 1:
                normalized = parts[0] + ' ' + ' '.join(sorted(parts[1].split()))

        return normalized
```

### 3. Cache Data Structures

#### Cached Entry Format
```python
@dataclass
class CachedRAGResult:
    """Complete cached RAG result."""

    # Query information
    original_query: str
    normalized_query: str
    query_embedding: np.ndarray

    # Results
    activated_knowledge: dict[str, Any]
    retrieved_documents: list[RetrievalResult]
    reasoning_result: dict[str, Any]
    integrated_result: dict[str, Any]

    # Metadata
    latency_ms: float
    cache_tier: int  # Which tier served this
    timestamp: float
    citations: list[Citation]
    confidence_score: float

    # Compression
    is_compressed: bool = False
    compression_ratio: float = 1.0

    def compress(self) -> bytes:
        """Compress for L3 storage."""
        # Keep only essential fields
        essential = {
            'query': self.normalized_query,
            'result': self.integrated_result,
            'docs': [{'id': d.id, 'score': d.score} for d in self.retrieved_documents[:5]],
            'citations': self.citations
        }
        return lz4.frame.compress(pickle.dumps(essential))

    def to_response(self) -> dict[str, Any]:
        """Convert to API response format."""
        return {
            "query": self.original_query,
            "result": self.integrated_result,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence_score,
            "cache_hit": True,
            "cache_tier": self.cache_tier,
            "latency_ms": self.latency_ms
        }
```

### 4. Semantic Similarity Cache

```python
class SemanticCache:
    """L2 cache with semantic similarity matching."""

    def __init__(self, config: SemanticCacheConfig):
        self.config = config
        self.index = self._build_index()
        self.entries: dict[str, CachedRAGResult] = {}
        self.embedding_dim = 768

    def _build_index(self) -> faiss.Index:
        """Build HNSW index for fast similarity search."""
        # HNSW parameters for <5ms search on 50k vectors
        index = faiss.IndexHNSWFlat(
            self.embedding_dim,
            32,  # Number of neighbors
            faiss.METRIC_INNER_PRODUCT
        )
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 16
        return index

    async def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.95
    ) -> list[tuple[str, float]]:
        """Find semantically similar cached queries."""
        # Normalize for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        # Search index
        scores, indices = self.index.search(
            query_norm.reshape(1, -1).astype('float32'),
            k
        )

        # Filter by threshold and return
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.entry_keys):
                key = self.entry_keys[idx]
                results.append((key, float(score)))

        return results

    def add_entry(self, key: str, embedding: np.ndarray, result: CachedRAGResult):
        """Add entry to semantic cache."""
        # Add to index
        norm_embedding = embedding / np.linalg.norm(embedding)
        self.index.add(norm_embedding.reshape(1, -1).astype('float32'))

        # Store entry
        self.entries[key] = result
        self.entry_keys.append(key)

        # Trigger eviction if needed
        if len(self.entries) > self.config.max_items:
            self._evict_entries()
```

### 5. Cache Eviction Strategy

```python
class AdaptiveEvictionPolicy:
    """Multi-factor eviction policy."""

    def __init__(self):
        self.weights = {
            'recency': 0.3,      # Time since last access
            'frequency': 0.25,   # Access count
            'semantic': 0.2,     # Semantic diversity
            'size': 0.15,        # Memory footprint
            'confidence': 0.1    # Result confidence
        }

    def compute_eviction_score(self, entry: CachedRAGResult, context: EvictionContext) -> float:
        """Compute eviction priority (higher = more likely to evict)."""

        # Recency score (exponential decay)
        age_hours = (time.time() - entry.last_access) / 3600
        recency_score = 1 - math.exp(-age_hours / 24)  # 24hr half-life

        # Frequency score (inverse log frequency)
        freq_score = 1 / (1 + math.log(1 + entry.access_count))

        # Semantic diversity (keep diverse entries)
        semantic_score = self._compute_semantic_redundancy(entry, context)

        # Size score (prefer evicting large entries)
        size_mb = sys.getsizeof(entry) / (1024 * 1024)
        size_score = min(1.0, size_mb / 10)  # Normalize to 10MB

        # Confidence score (keep high-confidence results)
        confidence_score = 1 - entry.confidence_score

        # Weighted combination
        total_score = (
            self.weights['recency'] * recency_score +
            self.weights['frequency'] * freq_score +
            self.weights['semantic'] * semantic_score +
            self.weights['size'] * size_score +
            self.weights['confidence'] * confidence_score
        )

        # Special cases
        if entry.citation_required:
            total_score *= 0.5  # Less likely to evict cited entries

        if context.memory_pressure > 0.9:
            total_score *= 1.5  # More aggressive eviction under pressure

        return total_score

    def _compute_semantic_redundancy(self, entry: CachedRAGResult, context: EvictionContext) -> float:
        """Measure semantic redundancy with other cached entries."""
        if not context.embedding_clusters:
            return 0.5

        # Find nearest cluster centroid
        min_distance = float('inf')
        for centroid in context.embedding_clusters:
            distance = np.linalg.norm(entry.query_embedding - centroid)
            min_distance = min(min_distance, distance)

        # Convert distance to redundancy score
        redundancy = 1 / (1 + min_distance)
        return redundancy
```

### 6. Cache Operations

#### Cache Lookup Flow
```python
class CacheManager:
    """Main cache management interface."""

    async def get(
        self,
        query: str,
        context: QueryContext | None = None
    ) -> CachedRAGResult | None:
        """Multi-tier cache lookup."""
        start_time = time.perf_counter()

        # Generate cache key
        normalized = self.normalizer.normalize(query)
        key = self._generate_key(normalized, context)

        # L1: Exact match
        if result := self.l1_cache.get(key.to_string()):
            self._record_hit('L1', time.perf_counter() - start_time)
            return result

        # Generate embedding for L2/L3
        embedding = await self._get_embedding(normalized)

        # L2: Semantic match
        similar_keys = await self.l2_cache.search(embedding, k=3)
        for similar_key, similarity in similar_keys:
            if similarity >= self.config.similarity_threshold:
                if result := self.l2_cache.get(similar_key):
                    # Promote to L1
                    self.l1_cache.put(key.to_string(), result)
                    self._record_hit('L2', time.perf_counter() - start_time)
                    return result

        # L3: Cold storage
        if self.config.use_cold_storage:
            if compressed := await self.l3_cache.get_async(embedding):
                result = self._decompress_result(compressed)
                # Promote to L2 and L1
                self.l2_cache.add_entry(key.to_string(), embedding, result)
                self.l1_cache.put(key.to_string(), result)
                self._record_hit('L3', time.perf_counter() - start_time)
                return result

        self._record_miss(time.perf_counter() - start_time)
        return None

    async def put(
        self,
        query: str,
        result: dict[str, Any],
        context: QueryContext | None = None
    ) -> None:
        """Add result to cache."""
        # Prepare cached entry
        normalized = self.normalizer.normalize(query)
        embedding = await self._get_embedding(normalized)

        cached_result = CachedRAGResult(
            original_query=query,
            normalized_query=normalized,
            query_embedding=embedding,
            **result,
            timestamp=time.time(),
            cache_tier=0
        )

        # Add to all tiers
        key = self._generate_key(normalized, context)

        # L1: Hot cache
        self.l1_cache.put(key.to_string(), cached_result)

        # L2: Semantic cache
        self.l2_cache.add_entry(key.to_string(), embedding, cached_result)

        # L3: Async compression and storage
        if self.config.use_cold_storage:
            asyncio.create_task(
                self._compress_and_store(key.to_string(), cached_result)
            )
```

### 7. Performance Optimization

#### Batching and Prefetching
```python
class PrefetchStrategy:
    """Intelligent query prefetching."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.query_patterns = defaultdict(list)
        self.prefetch_queue = asyncio.Queue(maxsize=100)

    async def analyze_query(self, query: str) -> list[str]:
        """Generate prefetch candidates."""
        candidates = []

        # Related questions
        candidates.extend(self._generate_related_questions(query))

        # Follow-up patterns
        if pattern := self._identify_pattern(query):
            candidates.extend(self.query_patterns[pattern][:3])

        # Temporal variations
        if self._is_temporal_query(query):
            candidates.extend(self._generate_temporal_variants(query))

        return candidates

    async def prefetch(self, candidates: list[str]) -> None:
        """Asynchronously prefetch candidate queries."""
        for candidate in candidates:
            await self.prefetch_queue.put(candidate)

    async def prefetch_worker(self) -> None:
        """Background worker for prefetching."""
        while True:
            candidate = await self.prefetch_queue.get()

            # Check if already cached
            if await self.cache_manager.get(candidate):
                continue

            # Execute pipeline with low priority
            try:
                result = await self._execute_low_priority_query(candidate)
                await self.cache_manager.put(candidate, result)
            except Exception:
                pass  # Ignore prefetch failures
```

#### Memory Management
```python
class MemoryManager:
    """Adaptive memory management for cache."""

    def __init__(self, target_memory_mb: int = 4096):
        self.target_memory = target_memory_mb * 1024 * 1024
        self.current_usage = 0
        self.pressure_threshold = 0.85

    def get_memory_pressure(self) -> float:
        """Return current memory pressure (0-1)."""
        return self.current_usage / self.target_memory

    def should_evict(self) -> bool:
        """Determine if eviction is needed."""
        return self.get_memory_pressure() > self.pressure_threshold

    def estimate_entry_size(self, entry: CachedRAGResult) -> int:
        """Estimate memory footprint of entry."""
        # Base size
        size = sys.getsizeof(entry)

        # Add embedding size
        if entry.query_embedding is not None:
            size += entry.query_embedding.nbytes

        # Add document sizes
        for doc in entry.retrieved_documents:
            size += sys.getsizeof(doc.content)

        return size
```

### 8. Integration Points

#### Pipeline Integration
```python
class CachedRAGPipeline(EnhancedRAGPipeline):
    """RAG pipeline with integrated caching."""

    def __init__(self, config: UnifiedConfig | None = None):
        super().__init__(config)
        self.cache_manager = CacheManager(
            CacheConfig.from_unified_config(config)
        )
        self.metrics = CacheMetrics()

    async def process(self, query: str) -> dict[str, Any]:
        """Process query with cache lookup."""
        start_time = time.perf_counter()

        # Try cache first
        if cached := await self.cache_manager.get(query):
            self.metrics.record_hit(time.perf_counter() - start_time)
            return cached.to_response()

        # Execute pipeline
        result = await super().process(query)

        # Cache result
        await self.cache_manager.put(query, result)

        self.metrics.record_miss(time.perf_counter() - start_time)
        return result
```

#### Credit System Integration
```python
class CreditAwareCache:
    """Cache with credit/billing awareness."""

    def __init__(self, credit_manager: CreditManager):
        self.credit_manager = credit_manager
        self.cache_manager = CacheManager()

    async def get_with_credits(
        self,
        query: str,
        user_id: str
    ) -> tuple[dict[str, Any], float]:
        """Get result and return credit cost."""

        # Check cache
        if cached := await self.cache_manager.get(query):
            # Cached queries cost 10% of original
            cost = self.credit_manager.CACHE_HIT_COST
            await self.credit_manager.deduct(user_id, cost)
            return cached.to_response(), cost

        # Full pipeline execution
        result = await self.execute_pipeline(query)
        cost = self.credit_manager.calculate_cost(result)
        await self.credit_manager.deduct(user_id, cost)

        # Cache for future discount
        await self.cache_manager.put(query, result)

        return result, cost
```

### 9. Monitoring and Metrics

```python
class CacheMetrics:
    """Cache performance monitoring."""

    def __init__(self):
        self.hits = Counter()  # Per tier
        self.misses = 0
        self.latencies = defaultdict(list)
        self.evictions = Counter()

    def get_hit_rate(self) -> float:
        """Calculate overall hit rate."""
        total_hits = sum(self.hits.values())
        total_requests = total_hits + self.misses
        return total_hits / max(1, total_requests)

    def get_tier_distribution(self) -> dict[str, float]:
        """Get hit distribution across tiers."""
        total_hits = sum(self.hits.values())
        if total_hits == 0:
            return {}
        return {
            tier: count / total_hits
            for tier, count in self.hits.items()
        }

    def get_p99_latency(self, tier: str | None = None) -> float:
        """Get P99 latency for tier or overall."""
        latencies = self.latencies[tier] if tier else sum(self.latencies.values(), [])
        if not latencies:
            return 0.0
        return np.percentile(latencies, 99)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Hit rate
        lines.append(f'rag_cache_hit_rate {self.get_hit_rate()}')

        # Per-tier metrics
        for tier, count in self.hits.items():
            lines.append(f'rag_cache_hits_total{{tier="{tier}"}} {count}')

        # Latency percentiles
        for tier in ['L1', 'L2', 'L3']:
            if tier in self.latencies:
                p50 = np.percentile(self.latencies[tier], 50)
                p99 = np.percentile(self.latencies[tier], 99)
                lines.append(f'rag_cache_latency_ms{{tier="{tier}",quantile="0.5"}} {p50}')
                lines.append(f'rag_cache_latency_ms{{tier="{tier}",quantile="0.99"}} {p99}')

        return '\n'.join(lines)
```

### 10. Configuration

```yaml
# cache_config.yaml
cache:
  enabled: true

  l1_hot_cache:
    max_size_mb: 512
    max_items: 10000
    ttl_seconds: 3600

  l2_semantic_cache:
    max_size_mb: 2048
    max_items: 50000
    similarity_threshold: 0.95
    index_type: "HNSW"
    index_params:
      neighbors: 32
      ef_construction: 40
      ef_search: 16

  l3_cold_cache:
    enabled: true
    max_size_mb: 8192
    compression: "lz4"
    persistence_path: "cache/rag_cache.db"
    index_type: "IVF"

  eviction:
    strategy: "adaptive"
    memory_pressure_threshold: 0.85
    check_interval_seconds: 60

  prefetching:
    enabled: true
    max_queue_size: 100
    worker_threads: 2

  monitoring:
    export_interval_seconds: 60
    prometheus_port: 9090
```

## Implementation Plan

### Phase 1: Core Cache Infrastructure (Week 1)
1. Implement multi-tier cache structure
2. Build cache key generation and normalization
3. Create basic eviction policy
4. Add memory management

### Phase 2: Semantic Caching (Week 2)
1. Implement HNSW index for L2 cache
2. Add semantic similarity search
3. Optimize embedding storage
4. Test similarity thresholds

### Phase 3: Performance Optimization (Week 3)
1. Add prefetching strategy
2. Implement compression for L3
3. Optimize cache promotion/demotion
4. Add batching support

### Phase 4: Integration & Monitoring (Week 4)
1. Integrate with RAG pipeline
2. Add credit system hooks
3. Implement Prometheus metrics
4. Performance testing and tuning

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| L1 Cache Hit Latency | <1ms | N/A |
| L2 Cache Hit Latency | <5ms | N/A |
| L3 Cache Hit Latency | <10ms | N/A |
| Overall Hit Rate | >70% | 0% |
| Memory Usage | <4GB | N/A |
| Compression Ratio (L3) | >5:1 | N/A |

## Risk Mitigation

1. **Memory Overflow**: Adaptive eviction with pressure monitoring
2. **Cache Coherence**: TTL and version-based invalidation
3. **Semantic Drift**: Periodic reindexing of L2 cache
4. **Cold Start**: Warmup from historical queries
5. **Citation Accuracy**: Preserve full documents for cited results

## Conclusion

This cache architecture provides a production-ready solution for achieving <10ms latency on cached RAG queries through:
- Multi-tier caching with semantic similarity
- Intelligent eviction and prefetching
- Efficient memory management
- Seamless integration with existing pipeline
- Future-proof credit system hooks

The modular design allows incremental deployment and testing while maintaining backward compatibility with the existing 48% complete RAG system.
