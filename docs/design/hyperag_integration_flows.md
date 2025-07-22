# HypeRAG Integration Flows

## 5. Integration Flow Specifications

### 5.1 Standard Query Flow

The standard query flow represents the typical path for non-creative queries.

```mermaid
sequenceDiagram
    participant Agent
    participant MCP
    participant Planner
    participant VectorRet as Vector Retriever
    participant PPR as PPR Retriever
    participant RelGAT as Rel-GAT
    participant Constructor
    participant Reasoner
    participant Guardian
    participant KGTrie as KG-Trie
    participant Client

    Agent->>MCP: POST /v1/hyperag/query
    MCP->>MCP: Authenticate & Authorize
    MCP->>Planner: Analyze query complexity

    Planner->>Planner: Classify (simple/complex)
    Planner->>Planner: Select strategy
    Planner->>VectorRet: Initial retrieval

    VectorRet->>VectorRet: k-NN search
    VectorRet->>PPR: Top-k candidates

    PPR->>PPR: Load α-profile
    PPR->>PPR: Personalized PageRank
    PPR->>RelGAT: Ranked entities

    RelGAT->>RelGAT: Apply attention weights
    RelGAT->>RelGAT: Multi-hop reasoning
    RelGAT->>Constructor: Scored results

    Constructor->>Constructor: Build context
    Constructor->>Reasoner: Structured input

    Reasoner->>Reasoner: Chain-of-thought
    Reasoner->>Guardian: Draft answer

    Guardian->>Guardian: Fact check
    Guardian->>Guardian: Policy check
    Guardian->>Guardian: Utility score

    alt Low confidence or high risk
        Guardian->>KGTrie: Activate constrained mode
        KGTrie->>KGTrie: Safe token generation
        KGTrie->>Guardian: Constrained answer
    end

    Guardian->>MCP: Final response
    MCP->>Agent: JSON response
```

### 5.2 Creativity Mode Flow

Creative queries follow a divergent path with additional validation.

```mermaid
sequenceDiagram
    participant Agent
    participant MCP
    participant Planner
    participant DivRetriever as Divergent Retriever
    participant HiddenLink as Hidden-Link Scanner
    participant Innovator
    participant Guardian
    participant Cache

    Agent->>MCP: POST /v1/hyperag/creative
    MCP->>Planner: mode=CREATIVE

    Planner->>DivRetriever: Activate divergent search

    DivRetriever->>DivRetriever: Ignore obvious paths
    DivRetriever->>DivRetriever: Random walk sampling
    DivRetriever->>DivRetriever: Calculate surprise scores

    par Parallel exploration
        DivRetriever->>HiddenLink: Find non-obvious connections
        HiddenLink->>HiddenLink: Transitive closure analysis
        HiddenLink->>HiddenLink: Community bridging
    and
        DivRetriever->>Innovator: Request creative bridges
        Innovator->>Innovator: Analogy generation
        Innovator->>Innovator: Metaphor discovery
    end

    HiddenLink->>DivRetriever: Hidden connections
    Innovator->>DivRetriever: Creative proposals

    DivRetriever->>Guardian: Combined results

    Guardian->>Guardian: Semantic coherence check
    Guardian->>Guardian: Novelty vs utility balance
    Guardian->>Guardian: Tag as "creative"

    Guardian->>Cache: Store for reuse
    Guardian->>MCP: Creative response
    MCP->>Agent: Bridges + explanations
```

### 5.3 Repair Cycle Flow

Automated graph repair workflow with human-in-the-loop option.

```mermaid
sequenceDiagram
    participant Monitor as GDC Monitor
    participant Extractor as GDC Extractor
    participant Innovator
    participant Guardian
    participant Hippo as Hippo-Index
    participant KG as Hypergraph-KG
    participant Human

    Monitor->>Extractor: Scheduled scan

    loop For each GDC rule
        Extractor->>KG: Execute Cypher query
        KG->>Extractor: Violations found
    end

    Extractor->>Innovator: Batch violations

    Innovator->>Innovator: Load repair templates
    Innovator->>Innovator: Generate proposals (LLM)
    Innovator->>Guardian: Repair proposals

    Guardian->>Guardian: Semantic utility check
    Guardian->>Guardian: Impact analysis

    alt High impact or low confidence
        Guardian->>Human: Request review
        Human->>Guardian: Approve/Reject/Modify
    end

    Guardian->>Guardian: Make decision

    alt Approved
        Guardian->>Hippo: Apply to working memory
        Hippo->>Hippo: Mark as quarantined
        Hippo->>Hippo: Set consolidation timer
    else Rejected
        Guardian->>Guardian: Log rejection reason
    end

    Note over Hippo,KG: Nightly consolidation
    Hippo->>KG: Migrate stable repairs
    KG->>KG: Update version history
```

### 5.4 Personalization Flow

Digital Twin integration with α-profile management.

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant MCP
    participant ProfileMgr as Profile Manager
    participant RelGAT
    participant Hippo
    participant Analytics

    User->>Agent: Query with user_id
    Agent->>MCP: Include user context

    MCP->>ProfileMgr: Load α-profile(user_id)

    alt Cold start user
        ProfileMgr->>ProfileMgr: Use global priors
        ProfileMgr->>ProfileMgr: Initialize profile
    else Existing user
        ProfileMgr->>ProfileMgr: Load from cache
    end

    ProfileMgr->>RelGAT: α-weights
    ProfileMgr->>Hippo: User episodic filter

    RelGAT->>RelGAT: Personalized attention
    Hippo->>Hippo: User-specific memories

    Note over MCP: Standard retrieval flow

    MCP->>Analytics: Log interaction
    Analytics->>Analytics: Update user model

    Analytics->>ProfileMgr: α-weight updates
    ProfileMgr->>ProfileMgr: Incremental learning
    ProfileMgr->>ProfileMgr: Diversity injection
```

### 5.5 Adapter Loading Flow

Hot-swapping domain and user adapters.

```mermaid
sequenceDiagram
    participant Agent
    participant MCP
    participant Registry as LoRA Registry
    participant Guardian
    participant Cache as Model Cache
    participant Loader as Hot-Swap Loader
    participant GPU

    Agent->>MCP: Request with adapter_id
    MCP->>Registry: Lookup adapter

    Registry->>Registry: Check signature
    Registry->>Guardian: Verify not revoked

    Guardian->>Guardian: Validate signature
    Guardian->>Registry: Approved

    Registry->>Cache: Check if loaded

    alt Not in cache
        Cache->>Registry: Download URL
        Registry->>Cache: S3/GCS path
        Cache->>Cache: Download weights
        Cache->>Loader: Load request

        Loader->>GPU: Allocate memory
        Loader->>Loader: Memory-map weights
        Loader->>Loader: Apply LoRA layers
        Loader->>Cache: Ready
    end

    Cache->>MCP: Adapter active
    MCP->>MCP: Route to adapted model

    Note over MCP: Process with adapter

    MCP->>Agent: Enhanced response
```

### 5.6 Nightly Consolidation Flow

Batch process for memory consolidation and maintenance.

```mermaid
sequenceDiagram
    participant Scheduler
    participant Consolidator
    participant Hippo
    participant KG
    participant GDC as GDC Validator
    participant HiddenLink
    participant Backup

    Scheduler->>Consolidator: 02:00 UTC trigger

    Consolidator->>Hippo: Query consolidation candidates
    Hippo->>Consolidator: Stable episodic memories

    loop For each batch
        Consolidator->>Consolidator: Calculate importance
        Consolidator->>KG: Create hyperedges
        KG->>KG: Update embeddings
        KG->>KG: Version tracking
        Consolidator->>Hippo: Mark consolidated
    end

    par Parallel maintenance
        Consolidator->>GDC: Full graph validation
        GDC->>GDC: Run all rules
        GDC->>Consolidator: New violations
    and
        Consolidator->>HiddenLink: Batch scan
        HiddenLink->>HiddenLink: Find creative paths
        HiddenLink->>Consolidator: Cache results
    and
        Consolidator->>Hippo: Garbage collection
        Hippo->>Hippo: Remove expired TTL
        Hippo->>Consolidator: Space reclaimed
    end

    Consolidator->>Backup: Snapshot
    Backup->>Backup: Incremental backup

    Consolidator->>Scheduler: Report complete
```

## Integration Patterns

### Event-Driven Architecture

```yaml
Events:
  QueryReceived:
    - Log to audit trail
    - Update metrics
    - Start trace

  RetrievalComplete:
    - Cache results
    - Update user profile
    - Trigger reasoning

  GuardianDecision:
    - Log decision
    - Update policy stats
    - Notify watchers

  RepairApplied:
    - Version graph
    - Notify agents
    - Schedule validation

  AdapterLoaded:
    - Warm cache
    - Update routing
    - Log usage
```

### Circuit Breaker Pattern

```yaml
CircuitBreakers:
  DivergentRetriever:
    failure_threshold: 5
    timeout: 5s
    half_open_tests: 3

  ExternalFactChecker:
    failure_threshold: 3
    timeout: 3s
    fallback: use_cached

  GuardianGate:
    failure_threshold: 10
    timeout: 10s
    fallback: conservative_mode
```

### Saga Pattern for Multi-Step Operations

```yaml
RepairSaga:
  steps:
    - name: DetectViolation
      compensate: LogSkipped

    - name: GenerateProposal
      compensate: DiscardProposal

    - name: ValidateProposal
      compensate: RejectProposal

    - name: ApplyToHippo
      compensate: RollbackHippo

    - name: ScheduleConsolidation
      compensate: CancelSchedule
```

## Performance Optimizations

### Caching Strategy

```yaml
CacheLayers:
  L1_Redis:
    - Query results (TTL: 1h)
    - User profiles (TTL: 24h)
    - Adapter metadata (TTL: 7d)

  L2_DuckDB:
    - Materialized PPR scores
    - Precomputed embeddings
    - Aggregate statistics

  L3_CDN:
    - Static adapters
    - Common responses
    - Documentation
```

### Batch Processing

```yaml
BatchQueues:
  RepairQueue:
    max_batch_size: 100
    max_wait_time: 5s
    processing_interval: 1m

  EmbeddingQueue:
    max_batch_size: 512
    max_wait_time: 100ms
    gpu_optimized: true

  ConsolidationQueue:
    max_batch_size: 10000
    processing_time: "02:00-04:00 UTC"
```

### Connection Pooling

```yaml
ConnectionPools:
  Neo4j:
    min_connections: 10
    max_connections: 100
    acquisition_timeout: 5s

  Redis:
    min_connections: 20
    max_connections: 200

  DuckDB:
    connection_mode: multi-threaded
    memory_limit: "8GB"
```

## Monitoring and Observability

### Key Metrics

```yaml
Metrics:
  Latency:
    - query_total_ms
    - retrieval_depth_avg
    - guardian_decision_ms

  Throughput:
    - queries_per_second
    - repairs_per_hour
    - consolidations_per_day

  Quality:
    - confidence_avg
    - guardian_override_rate
    - creative_success_rate

  Resources:
    - memory_usage_gb
    - gpu_utilization_pct
    - cache_hit_rate
```

### Distributed Tracing

```yaml
TracePoints:
  - Request received
  - Auth completed
  - Planning finished
  - Each retrieval stage
  - Guardian decision
  - Response sent

TraceContext:
  - request_id
  - user_id
  - agent_id
  - mode
  - confidence
```

### Health Checks

```yaml
HealthEndpoints:
  /health/live:
    - Basic connectivity

  /health/ready:
    - All components initialized
    - Minimum cache warm

  /health/startup:
    - Detailed component status
    - Resource availability
    - Dependency checks
```
