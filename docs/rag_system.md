# RAG System Documentation

## Overview

The Retrieval-Augmented Generation (RAG) system is a core component that enables knowledge management, retrieval, and generation across the AI Village. It combines vector-based and graph-based approaches for efficient information processing.

## Architecture

### Components

1. **Vector Store**
   - Implementation: FAISS
   - Location: `rag_system/retrieval/vector_store.py`
   - Purpose: Efficient similarity search
   - Features:
     - Dense vector embeddings
     - Fast nearest neighbor search
     - Batch processing support

2. **Graph Store**
   - Implementation: NetworkX
   - Location: `rag_system/retrieval/graph_store.py`
   - Purpose: Relationship management
   - Features:
     - Knowledge graph representation
     - Relationship tracking
     - Path analysis

3. **Hybrid Retriever**
   - Location: `rag_system/retrieval/hybrid_retriever.py`
   - Purpose: Combined retrieval strategy
   - Features:
     - Vector similarity search
     - Graph traversal
     - Result fusion

4. **Query Processor**
   - Location: `rag_system/processing/query_processor.py`
   - Purpose: Query understanding and optimization
   - Features:
     - Query parsing
     - Intent recognition
     - Context integration

### Data Flow

1. Input Processing:
   ```
   User Query → Query Processor → Intent Recognition → Query Optimization
   ```

2. Retrieval:
   ```
   Optimized Query → Hybrid Retriever → [Vector Store + Graph Store] → Combined Results
   ```

3. Knowledge Integration:
   ```
   New Information → Knowledge Validator → Graph Update → Vector Update
   ```

## Usage

### Basic Query
```python
from rag_system.core.pipeline import RAGPipeline

pipeline = RAGPipeline()
results = await pipeline.query("How does the King agent make decisions?")
```

### Knowledge Addition
```python
from rag_system.core.knowledge_manager import KnowledgeManager

manager = KnowledgeManager()
await manager.add_knowledge(
    concept="decision_making",
    content="Process of evaluating options...",
    relationships=[("requires", "data_analysis")]
)
```

### Graph Traversal
```python
from rag_system.retrieval.graph_store import GraphStore

graph_store = GraphStore()
path = await graph_store.find_path(
    start="user_input",
    end="agent_response"
)
```

## Integration with Agents

### King Agent
- Uses RAG for decision context
- Updates knowledge with decisions
- Queries for similar past decisions

### Sage Agent
- Primary knowledge curator
- Performs research through RAG
- Updates knowledge base

### Magi Agent
- Queries for code patterns
- Stores tool implementations
- Links code to concepts

## Performance Optimization

### Caching
```python
from rag_system.core.performance_optimization import cache_result

@cache_result(ttl=3600)
async def get_frequent_query(query: str):
    return await rag_pipeline.query(query)
```

### Batch Processing
```python
from rag_system.processing.batch_processor import BatchProcessor

processor = BatchProcessor()
results = await processor.process_batch(queries, batch_size=100)
```

### Async Operations
```python
async def parallel_query(queries: List[str]):
    tasks = [rag_pipeline.query(q) for q in queries]
    return await asyncio.gather(*tasks)
```

## Monitoring

### Performance Metrics
```python
from rag_system.core.monitoring import RAGMonitor

monitor = RAGMonitor()
metrics = await monitor.get_metrics()
print(f"Query latency: {metrics['average_latency']}ms")
```

### Health Checks
```python
from rag_system.core.health import HealthChecker

checker = HealthChecker()
status = await checker.check_rag_system()
print(f"System status: {status}")
```

## Error Handling

### Retry Logic
```python
from rag_system.error_handling.error_handler import retry_operation

@retry_operation(max_retries=3)
async def safe_query(query: str):
    return await rag_pipeline.query(query)
```

### Fallback Strategies
```python
try:
    results = await hybrid_retriever.query(query)
except RetrievalError:
    results = await vector_store.simple_query(query)
```

## Configuration

### System Settings
```yaml
rag_system:
  vector_store:
    dimension: 768
    index_type: "IVF"
    metric: "cosine"
  graph_store:
    backend: "networkx"
    persistence: true
  retrieval:
    top_k: 5
    threshold: 0.7
```

### Performance Tuning
```yaml
optimization:
  cache_size: 1000
  batch_size: 100
  max_concurrent: 10
  timeout: 30
```

## Testing

### Unit Tests
```python
def test_vector_retrieval():
    store = VectorStore()
    results = store.query("test query")
    assert len(results) > 0
```

### Integration Tests
```python
async def test_hybrid_retrieval():
    retriever = HybridRetriever()
    results = await retriever.query("test")
    assert results.vector_results and results.graph_results
```

## Maintenance

### Database Cleanup
```python
from rag_system.maintenance import Maintainer

maintainer = Maintainer()
await maintainer.cleanup_old_entries(days=30)
```

### Index Optimization
```python
from rag_system.optimization import Optimizer

optimizer = Optimizer()
await optimizer.optimize_indices()
```

## Best Practices

1. Query Formation
   - Be specific and clear
   - Include relevant context
   - Use structured queries when possible

2. Knowledge Management
   - Validate new information
   - Maintain relationships
   - Regular cleanup

3. Performance
   - Use batch processing
   - Implement caching
   - Monitor system health

4. Error Handling
   - Implement retries
   - Use fallback strategies
   - Log all errors

## Common Issues

1. Retrieval Quality
   - Check embedding quality
   - Verify knowledge graph connections
   - Adjust similarity thresholds

2. Performance
   - Monitor index size
   - Check cache hit rates
   - Optimize batch sizes

3. Integration
   - Verify API compatibility
   - Check authentication
   - Monitor timeouts
