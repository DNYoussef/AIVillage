# HypeRAG Data Models

## Overview

This document defines the core data models for HypeRAG v3, supporting dual-memory architecture, creativity modes, graph repair, and personalization features.

## Core Data Models

### 1. Hyperedge

Represents n-ary relationships in the knowledge hypergraph.

```yaml
Hyperedge:
  id: string                    # UUID format: "he_<uuid>"
  entities: string[]            # n-ary entity references
  relation: string              # Relation type/name
  confidence: float             # Bayesian confidence [0.0, 1.0]
  timestamp: datetime           # Creation/update time
  tags: string[]                # ["creative", "repaired", "verified", "quarantined"]
  popularity_rank: int          # Usage-based ranking
  alpha_weight: float|null      # Rel-GAT personalization weight
  source_docs: string[]         # Document IDs for provenance
  embedding: vector             # 768-dim semantic embedding

  # Extended fields
  uncertainty: float            # Epistemic uncertainty measure
  decay_rate: float            # Time-based relevance decay
  access_count: int            # Read frequency tracking
  last_accessed: datetime      # For LRU policies
  creator_agent: string        # Agent that created this edge
  validator_agent: string|null # Guardian validation record
  repair_history: RepairLog[]  # Audit trail of modifications
```

### 2. HippoNode

Episodic memory nodes with fast write and novelty detection.

```yaml
HippoNode:
  id: string                    # UUID format: "hn_<uuid>"
  content: string               # Raw text content
  user_id: string|null          # Digital Twin association
  episodic: bool                # True for working memory
  created: datetime             # Creation timestamp
  novelty_score_cache: float|null # Pre-computed novelty
  gdc_flags: string[]           # Graph diagnostic criteria violations
  embedding: vector             # 768-dim embedding

  # Extended fields
  consolidation_status: enum    # {pending, migrating, consolidated}
  ttl: int|null                # Time-to-live in seconds
  importance_score: float       # For consolidation priority
  context_window: string[]      # Surrounding node IDs
  emotional_valence: float|null # -1.0 to 1.0 sentiment
  confidence_delta: float       # Change since creation
  access_pattern: AccessLog[]   # Detailed usage tracking
```

### 3. GDCSpec

Graph Diagnostic Criteria specifications for automatic repair.

```yaml
GDCSpec:
  id: string                    # Format: "gdc_<category>_<number>"
  description: string           # Human-readable rule description
  cypher: string               # Neo4j query to detect violations
  severity: string             # Enum: {low, medium, high, critical}
  suggested_action: string     # Repair recommendation

  # Extended fields
  category: string             # {consistency, completeness, accuracy, relevance}
  auto_repair: bool            # Can be fixed automatically
  repair_template: string|null # LLM prompt template
  validation_query: string     # Cypher to verify repair
  false_positive_rate: float   # Historical accuracy
  last_triggered: datetime|null
  trigger_count: int
  exemptions: string[]         # Node/edge IDs to ignore
```

### 4. RepairLog

Audit trail for graph modifications.

```yaml
RepairLog:
  id: string
  timestamp: datetime
  gdc_violation: string        # GDCSpec ID
  original_state: object       # JSON snapshot before repair
  proposed_fix: object         # Innovator suggestion
  guardian_decision: string    # {approved, rejected, quarantined}
  final_state: object|null     # JSON snapshot after repair
  confidence: float
  impact_score: float          # Estimated graph-wide effect
  rollback_available: bool
```

### 5. AccessLog

Detailed tracking for access patterns and personalization.

```yaml
AccessLog:
  timestamp: datetime
  agent_id: string
  operation: string            # {read, write, traverse}
  latency_ms: int
  cache_hit: bool
  retrieval_depth: int
  confidence_threshold: float
```

### 6. AlphaProfile

Rel-GAT personalization weights per user/agent.

```yaml
AlphaProfile:
  id: string                   # Format: "alpha_<user_id>"
  user_id: string
  relation_weights: dict       # {relation_type: weight}
  entity_weights: dict         # {entity_id: weight}
  global_prior: float          # Default weight for unknowns
  learning_rate: float
  update_count: int
  last_updated: datetime

  # Extended fields
  cold_start: bool
  diversity_score: float       # Prevents filter bubbles
  exploration_bonus: float     # Encourages novel paths
  decay_schedule: object       # Time-based weight decay
```

### 7. LoRAAdapter

Domain and user-specific model adaptations.

```yaml
LoRAAdapter:
  id: string                   # SHA-256 hash
  name: string
  description: string
  domain: string|null          # Target domain
  user_id: string|null         # Target user

  # Technical fields
  base_model: string           # Model architecture
  rank: int                    # LoRA rank
  alpha: float                 # LoRA alpha parameter
  target_modules: string[]     # Modified layers

  # Binary data
  weights_url: string          # S3/GCS URL
  weights_hash: string         # SHA-256 of weights

  # Metadata
  created: datetime
  creator_agent: string
  training_samples: int
  validation_score: float

  # Security
  guardian_signature: string   # Guardian approval
  trusted: bool
  revoked: bool
  revocation_reason: string|null
```

### 8. CreativeQuery

Specialized query format for creativity mode.

```yaml
CreativeQuery:
  id: string
  source_concept: string
  target_concept: string|null   # Optional for open-ended
  mode: string                 # {divergent, analogical, combinatorial}

  # Constraints
  max_hops: int                # Graph traversal limit
  min_surprise: float          # Novelty threshold
  time_budget_ms: int          # Computation limit
  avoid_paths: string[]        # Explicit exclusions

  # Results
  bridges_found: Bridge[]
  computation_time_ms: int
  guardian_vetted: bool
```

### 9. Bridge

Creative connection between concepts.

```yaml
Bridge:
  id: string
  path: string[]               # Node IDs in sequence
  relations: string[]          # Edge types traversed
  surprise_score: float        # Semantic distance metric
  confidence: float
  explanation: string          # LLM-generated rationale
  tags: string[]              # ["metaphor", "analogy", "synthesis"]
```

### 10. GuardianDecision

Validation record for safety checks.

```yaml
GuardianDecision:
  id: string
  timestamp: datetime
  request_type: string         # {query, repair, adapter, creative}
  request_id: string

  # Decision process
  semantic_score: float        # Meaning preservation
  utility_score: float         # Usefulness measure
  safety_score: float          # Risk assessment
  policy_violations: string[]  # Failed rules

  # Outcome
  decision: string            # {apply, quarantine, reject}
  reasoning: string           # Explanation
  override_available: bool    # Human review option
  ttl: int|null              # Auto-expire quarantine
```

## Relationships

### Primary Relationships

```cypher
// Hyperedge connects multiple entities
(e1:Entity)-[:PARTICIPATES_IN]->(h:Hyperedge)<-[:PARTICIPATES_IN]-(e2:Entity)

// HippoNodes reference entities
(hn:HippoNode)-[:MENTIONS]->(e:Entity)

// Consolidation flow
(hn:HippoNode)-[:CONSOLIDATES_TO]->(h:Hyperedge)

// Repair tracking
(rl:RepairLog)-[:REPAIRS]->(h:Hyperedge)
(rl:RepairLog)-[:TRIGGERED_BY]->(gdc:GDCSpec)

// Personalization
(ap:AlphaProfile)-[:WEIGHTS]->(h:Hyperedge)
(ap:AlphaProfile)-[:BELONGS_TO]->(u:User)

// Adapter usage
(la:LoRAAdapter)-[:USED_BY]->(a:Agent)
(la:LoRAAdapter)-[:SIGNED_BY]->(gd:GuardianDecision)
```

## Indexes and Constraints

### Performance Indexes
```sql
-- DuckDB (Hippo-Index)
CREATE INDEX idx_hippo_embedding ON hippo_nodes USING hnsw(embedding);
CREATE INDEX idx_hippo_user_time ON hippo_nodes(user_id, created DESC);
CREATE INDEX idx_hippo_novelty ON hippo_nodes(novelty_score_cache DESC);
CREATE INDEX idx_hippo_ttl ON hippo_nodes(created, ttl) WHERE ttl IS NOT NULL;

-- Neo4j (Hypergraph-KG)
CREATE INDEX hyperedge_confidence FOR (h:Hyperedge) ON (h.confidence);
CREATE INDEX hyperedge_timestamp FOR (h:Hyperedge) ON (h.timestamp);
CREATE INDEX entity_embedding FOR (e:Entity) ON (e.embedding);
CREATE FULLTEXT INDEX entity_content FOR (e:Entity) ON (e.content);
```

### Constraints
```sql
-- Uniqueness
CREATE CONSTRAINT unique_hyperedge_id ON (h:Hyperedge) ASSERT h.id IS UNIQUE;
CREATE CONSTRAINT unique_hippo_id ON (hn:HippoNode) ASSERT hn.id IS UNIQUE;
CREATE CONSTRAINT unique_adapter_hash ON (la:LoRAAdapter) ASSERT la.weights_hash IS UNIQUE;

-- Data integrity
ALTER TABLE hippo_nodes ADD CONSTRAINT valid_confidence CHECK (confidence BETWEEN 0.0 AND 1.0);
ALTER TABLE alpha_profiles ADD CONSTRAINT positive_weights CHECK (global_prior > 0);
```

## Migration Schemas

### Hippo to Hypergraph Consolidation
```yaml
ConsolidationBatch:
  id: string
  start_time: datetime
  end_time: datetime|null

  # Selection criteria
  min_importance: float
  max_nodes: int
  user_filter: string[]|null

  # Results
  nodes_processed: int
  edges_created: int
  failures: object[]

  # State
  status: string              # {running, completed, failed}
  checkpoint: string|null     # Resume point
```

### Version Migration
```yaml
SchemaVersion:
  version: string             # Semantic version
  applied: datetime
  migrations: string[]        # Applied migration scripts
  rollback_available: bool
```

## Data Governance

### Retention Policies
- **HippoNodes**: 30-day default TTL, extended by access
- **Hyperedges**: Permanent, with decay_rate for relevance
- **RepairLogs**: 90-day retention, then archived
- **AccessLogs**: 7-day detail, then aggregated

### Privacy Controls
- **PII Masking**: Automatic detection and redaction
- **Right to Forget**: User-triggered cascade deletion
- **Audit Trail**: Immutable log with encryption

### Backup Strategy
- **Hot Backup**: Continuous replication to standby
- **Cold Backup**: Daily snapshots to object storage
- **Point-in-Time Recovery**: 7-day window
