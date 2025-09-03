# HippoCache Usage Guide

The `HippoCache` provides a high‑performance semantic cache for the unified RAG
stack.  It stores embeddings and retrieved documents, evicting entries with an
LRU policy and optional time‑to‑live.

This guide complements the
[Unified RAG Module Structure](../unified_rag_module_structure.md).

## Configuration

`HippoCache` can be tuned through its constructor:

| Option | Description | Default |
| --- | --- | --- |
| `max_size` | Maximum number of cached entries | `10_000` |
| `ttl_hours` | Hours before an entry expires | `24` |
| `similarity_threshold` | Minimum cosine similarity for a hit | `0.95` |

## Cache API

### `get(query_embedding)`
Returns a `CacheEntry` whose embedding is above the similarity threshold or
`None` if no suitable entry exists.

### `set(key, entry)`
Inserts or updates an entry keyed by `key`.  Embeddings are normalized and the
cache respects the configured size limit.

### `get_or_retrieve(key, query_embedding, retrieval_fn)`
Attempts `get`; on a miss, `retrieval_fn` is invoked to fetch documents and
scores, which are then cached under `key`.

### `metrics()`
Exposes basic performance metrics:

```python
{
    "hit_rate": float,       # cache hits / total lookups
    "avg_latency_ms": float, # mean lookup latency
    "size": int,             # live entry count
}
```

## Example Integration with `UnifiedRAGSystem`

```python
from datetime import datetime
import numpy as np

from unified_rag.core.unified_rag_system import UnifiedRAGSystem
from integrations.clients.py_aivillage.rag.hippo_cache import CacheEntry

rag = UnifiedRAGSystem()
await rag.initialize()

embedding = np.random.rand(768)

def retrieve_docs():
    # Domain‑specific retrieval logic
    return ["doc"], [1.0], {"source": "demo"}

entry = rag.cache.get_or_retrieve("demo", embedding, retrieve_docs)
print(rag.cache.metrics())
```

