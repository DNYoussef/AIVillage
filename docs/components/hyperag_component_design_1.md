# HypeRAG Component Design

## 3. Component Specifications

### 3.1 MCP Core

The MCP Core provides the foundational infrastructure for all HypeRAG operations.

#### Architecture
```yaml
MCPCore:
  components:
    ConnectionHandler:
      - WebSocket connection pooling
      - HTTP/2 multiplexing
      - Circuit breaker patterns
      - Graceful degradation

    AuthenticationManager:
      - JWT token validation
      - mTLS certificate verification
      - API key management
      - Session caching

    AuthorizationEngine:
      - RBAC with YAML policies
      - Dynamic permission loading
      - Operation-level gating
      - Audit trail generation

    ModelInjectionRegistry:
      - Hot-swappable models
      - Version management
      - Compatibility checking
      - Resource isolation
```

#### Connection Handler Details
```python
ConnectionConfig:
  max_connections: 1000
  connection_timeout: 30s
  idle_timeout: 300s
  keepalive_interval: 60s

  pooling:
    min_size: 10
    max_size: 100
    queue_size: 500

  circuit_breaker:
    failure_threshold: 5
    timeout: 60s
    half_open_requests: 3
```

#### Authentication Flow
```
1. TLS handshake with client certificate
2. Extract JWT from Authorization header
3. Validate JWT signature and claims
4. Check certificate revocation list
5. Cache auth decision (TTL: 5 min)
6. Return auth context with permissions
```

#### Model Injection Interface
```yaml
AgentReasoningModel:
  interface:
    - encode(text: str) -> Embedding
    - reason(context: Context) -> Reasoning
    - generate(prompt: str) -> str

  requirements:
    - Stateless operations
    - Thread-safe
    - GPU-aware
    - Checkpointing support
```

### 3.2 Memory Layer

Dual-memory system inspired by hippocampal-neocortical interactions.

#### Hippo-Index (Episodic Memory)

```yaml
HippoIndex:
  storage:
    primary: DuckDB
      - Columnar format for analytics
      - Window functions for recency
      - Embedded for edge deployment

    cache: Redis
      - LRU eviction
      - TTL-based expiry
      - Pub/Sub for updates

  features:
    FastWrites:
      - Append-only log
      - Batch inserts (1000/batch)
      - Async I/O

    NoveltyDetection:
      - Embedding similarity threshold
      - Semantic surprise scoring
      - Anomaly detection

    GDCIntegration:
      - Rule violation flags
      - Inline validation
      - Repair queue
```

#### DuckDB Schema
```sql
CREATE TABLE hippo_nodes (
    id VARCHAR PRIMARY KEY,
    content TEXT NOT NULL,
    embedding FLOAT[768] NOT NULL,
    user_id VARCHAR,
    episodic BOOLEAN DEFAULT true,
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ttl INTEGER,
    novelty_score FLOAT,
    gdc_flags VARCHAR[],
    importance_score FLOAT DEFAULT 0.5,
    consolidation_status VARCHAR DEFAULT 'pending',

    -- Indexes
    INDEX idx_embedding (embedding) USING HNSW,
    INDEX idx_user_created (user_id, created DESC),
    INDEX idx_ttl (created, ttl) WHERE ttl IS NOT NULL,
    INDEX idx_consolidation (consolidation_status, importance_score DESC)
);
```

#### Hypergraph-KG (Semantic Memory)

```yaml
HypergraphKG:
  storage:
    graph: Neo4j
      - ACID transactions
      - Cypher query language
      - Causal clustering

    embeddings: Qdrant
      - HNSW indexing
      - Distributed sharding
      - Filtering support

  features:
    Hyperedges:
      - N-ary relationships
      - Property graphs
      - Temporal versioning

    BayesianConfidence:
      - Prior/posterior updates
      - Uncertainty propagation
      - Evidence accumulation

    AlphaWeights:
      - Per-user attention
      - Relation-specific
      - Adaptive learning
```

#### Neo4j Schema
```cypher
// Node types
CREATE CONSTRAINT entity_id ON (e:Entity) ASSERT e.id IS UNIQUE;
CREATE CONSTRAINT hyperedge_id ON (h:Hyperedge) ASSERT h.id IS UNIQUE;

// Hyperedge structure
CREATE (h:Hyperedge {
    id: 'he_' + randomUUID(),
    relation: 'related_to',
    confidence: 0.8,
    timestamp: datetime(),
    tags: ['verified'],
    popularity_rank: 0,
    alpha_weight: null,
    source_docs: ['doc1', 'doc2']
})

// N-ary connections
MATCH (e1:Entity), (e2:Entity), (e3:Entity)
CREATE (e1)-[:PARTICIPATES {role: 'subject'}]->(h)
CREATE (e2)-[:PARTICIPATES {role: 'object'}]->(h)
CREATE (e3)-[:PARTICIPATES {role: 'context'}]->(h)
```

### 3.3 Retrieval Stack

Multi-stage retrieval with progressive refinement.

#### Stage 1: Vector Retriever
```yaml
VectorRetriever:
  algorithm: HNSW
  parameters:
    m: 16                    # Number of bi-directional links
    ef_construction: 200     # Size of dynamic candidate list
    ef_search: 100          # Size of search candidate list

  optimizations:
    - Product quantization for compression
    - Batch query processing
    - GPU acceleration (if available)
    - Approximate nearest neighbor
```

#### Stage 2: PPR Retriever
```yaml
PPRRetriever:
  algorithm: PersonalizedPageRank
  parameters:
    alpha: 0.85             # Damping factor
    max_iterations: 100     # Convergence limit
    tolerance: 1e-6         # Convergence threshold

  personalization:
    - Seed from query entities
    - User history weighting
    - Context-aware damping

  optimizations:
    - Incremental computation
    - Sparse matrix operations
    - Distributed calculation
```

#### Stage 3: Rel-GAT Rescorer
```yaml
RelGATRescorer:
  architecture:
    layers: 3
    heads: 8
    hidden_dim: 256
    dropout: 0.1

  features:
    - Relation-aware attention
    - User-specific α-weights
    - Multi-hop reasoning
    - Confidence propagation

  training:
    - Online learning from feedback
    - Contrastive loss
    - Importance sampling
```

#### Stage 4: Divergent Retriever
```yaml
DivergentRetriever:
  modes:
    CREATIVE:
      - Ignore high-probability paths
      - Maximize semantic distance
      - Explore rare connections

    REPAIR:
      - Focus on inconsistencies
      - Find alternative paths
      - Suggest missing links

  algorithms:
    SurpriseScoring:
      - KL divergence from expected
      - Mutual information gain
      - Novelty detection

    HiddenLinkScanning:
      - Transitive closure analysis
      - Latent factor discovery
      - Community bridging
```

### 3.4 Planning Engine

Intelligent query planning with mode selection.

```yaml
PlanningEngine:
  components:
    ComplexityClassifier:
      features:
        - Query length
        - Entity count
        - Relation types
        - Historical latency

      output:
        - simple/complex/creative
        - Estimated cost
        - Recommended depth

    StrategySelector:
      strategies:
        BreadthFirst:
          - Low latency priority
          - Shallow exploration

        DepthFirst:
          - Accuracy priority
          - Deep reasoning chains

        BeamSearch:
          - Balanced approach
          - Pruning low confidence

        MonteCarlo:
          - Creative exploration
          - Random walks

    ModeController:
      modes:
        NORMAL:
          - Standard retrieval flow
          - Efficiency focus

        CREATIVE:
          - Divergent thinking
          - Surprise optimization

        REPAIR:
          - Consistency checking
          - Gap identification

    RePlanner:
      triggers:
        - Low confidence results
        - Timeout approaching
        - User feedback
        - Cost overrun
```

### 3.5 Innovator Agent (Graph Doctor)

Automated graph repair and enhancement system.

```yaml
InnovatorAgent:
  components:
    GDCEngine:
      rules:
        - Consistency checks
        - Completeness validation
        - Accuracy verification
        - Relevance scoring

    TemplateEncoder:
      templates:
        MissingRelation:
          prompt: "Entity {A} and {B} are related but missing connection"

        InconsistentConfidence:
          prompt: "Hyperedge {H} confidence conflicts with evidence"

        OrphanedEntity:
          prompt: "Entity {E} has no connections"

    RepairGenerator:
      llm_config:
        model: "gpt-4"
        temperature: 0.3
        max_tokens: 500

      constraints:
        - Preserve existing facts
        - Minimize changes
        - Maintain confidence bounds

    DivergentBridgeFinder:
      algorithm:
        - Random walk sampling
        - Embedding space exploration
        - Analogy detection
        - Metaphor generation
```

#### GDC Rule Examples
```cypher
// Missing inverse relations
MATCH (a:Entity)-[r:CAUSES]->(b:Entity)
WHERE NOT EXISTS((b)-[:CAUSED_BY]->(a))
RETURN a, b, r AS violation

// Confidence inconsistency
MATCH (h:Hyperedge)<-[:PARTICIPATES]-(e:Entity)
WITH h, COUNT(e) as participant_count, h.confidence as conf
WHERE participant_count > 5 AND conf < 0.3
RETURN h AS violation

// Isolated entities
MATCH (e:Entity)
WHERE NOT EXISTS((e)-[:PARTICIPATES]->(:Hyperedge))
RETURN e AS violation
```

### 3.6 Guardian Gate

Safety validation and policy enforcement layer.

```yaml
GuardianGate:
  components:
    ExternalFactChecker:
      sources:
        - Wikipedia API
        - Scientific databases
        - Fact-checking services

      validation:
        - Cross-reference claims
        - Check source reliability
        - Temporal consistency

    SemanticUtilityScorer:
      metrics:
        Relevance:
          - Query alignment
          - Context preservation
          - Information gain

        Coherence:
          - Logical consistency
          - Narrative flow
          - Factual accuracy

        Novelty:
          - Surprise value
          - Knowledge expansion
          - Creative merit

    PolicyEngine:
      policies:
        ContentPolicy:
          - Harmful content detection
          - PII protection
          - Copyright compliance

        QualityPolicy:
          - Minimum confidence threshold
          - Source diversity requirement
          - Recency preferences

        SafetyPolicy:
          - Hallucination detection
          - Fact verification
          - Uncertainty bounds

    DecisionEngine:
      actions:
        APPLY:
          - Direct application
          - Audit log entry

        QUARANTINE:
          - Temporary isolation
          - Review queue
          - TTL expiration

        REJECT:
          - Block operation
          - Detailed reasoning
          - Appeal process
```

#### Guardian Decision Flow
```
1. Receive request (query/repair/creative)
2. Extract claims and entities
3. Run external fact checks (parallel)
4. Calculate semantic utility scores
5. Apply policy rules
6. Make decision (apply/quarantine/reject)
7. Log decision with reasoning
8. Trigger KG-Trie if high risk
```

### 3.7 Generation Layer

Response generation with safety controls.

#### Standard Decoder
```yaml
StandardDecoder:
  architecture:
    - Transformer-based
    - Attention caching
    - Beam search

  features:
    - Confidence scoring
    - Token probabilities
    - Length control

  optimizations:
    - KV-cache
    - Quantization
    - Speculative decoding
```

#### KG-Trie Constrained Decoder
```yaml
KGTrieDecoder:
  structure:
    TrieNode:
      - Token ID
      - Valid next tokens
      - Confidence score
      - Knowledge ground

  construction:
    - Build from knowledge graph
    - Update incrementally
    - Prune low confidence

  decoding:
    - Constrain token selection
    - Backtrack on dead ends
    - Maintain multiple beams

  triggers:
    - High risk queries
    - Low confidence context
    - Safety policy override
```

### 3.8 LoRA-Kit

Adapter management for personalization and domain adaptation.

```yaml
LoRAKit:
  components:
    TrainingPipeline:
      stages:
        DataPreparation:
          - Template alignment
          - Quality filtering
          - Deduplication

        Training:
          - Low-rank decomposition
          - Gradient accumulation
          - Early stopping

        Validation:
          - Perplexity measurement
          - Task-specific metrics
          - Overfitting detection

    AdapterRegistry:
      storage:
        - S3/GCS for weights
        - PostgreSQL for metadata
        - Redis for hot cache

      versioning:
        - Semantic versioning
        - Dependency tracking
        - Rollback support

    SignatureSystem:
      process:
        - SHA-256 weight hashing
        - Guardian review
        - Digital signature
        - Certificate chain

    HotSwapLoader:
      features:
        - Zero-downtime updates
        - Memory-mapped loading
        - Lazy initialization
        - Resource isolation
```

#### Adapter Training Configuration
```yaml
LoRAConfig:
  rank: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["q_proj", "v_proj"]

  training:
    learning_rate: 1e-4
    batch_size: 8
    gradient_accumulation: 4
    max_steps: 1000
    warmup_steps: 100

  data:
    min_samples: 1000
    max_samples: 100000
    validation_split: 0.1
```

## Component Integration Points

### Message Bus Architecture
```yaml
MessageBus:
  broker: RabbitMQ/Kafka

  topics:
    queries: "hyperag.queries"
    retrievals: "hyperag.retrievals"
    repairs: "hyperag.repairs"
    decisions: "hyperag.guardian.decisions"

  patterns:
    - Request-Reply for sync operations
    - Publish-Subscribe for events
    - Dead Letter Queue for failures
```

### Service Mesh Configuration
```yaml
ServiceMesh:
  proxy: Envoy/Istio

  features:
    - mTLS between services
    - Circuit breaking
    - Retry policies
    - Load balancing
    - Observability
```

### Monitoring Integration
```yaml
Monitoring:
  metrics: Prometheus
  tracing: Jaeger
  logging: ELK Stack

  key_metrics:
    - Request latency (p50, p95, p99)
    - Retrieval depth
    - Cache hit rates
    - Guardian overrides
    - Repair success rate
```
