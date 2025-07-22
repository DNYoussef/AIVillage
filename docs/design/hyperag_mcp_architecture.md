# HypeRAG MCP Server Architecture Overview (v3)

## Executive Summary

HypeRAG v3 extends the dual-memory architecture with creativity modes, automatic graph repair, safety validation, and personalization. This document outlines the complete architectural design for the MCP server implementation.

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               HypeRAG MCP Server Core                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐   │
│  │  Connection     │    │  Authentication  │    │    Model Injection          │   │
│  │  Pooling        │◄───┤  & Authorization  │◄───┤    Registry                 │   │
│  └─────────────────┘    └──────────────────┘    └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              Dual-Memory Layer                                       │
├──────────────────────────────────┬──────────────────────────────────────────────────┤
│        Hippo-Index (Episodic)    │         Hypergraph-KG (Semantic)                 │
│  ┌────────────────────────────┐  │  ┌───────────────────────────────────────────┐  │
│  │  • DuckDB columnar store   │  │  │  • Neo4j hyperedges                       │  │
│  │  • Redis TTL cache         │  │  │  • Qdrant embeddings                      │  │
│  │  • Fast episodic writes    │◄─┼─►│  • Bayesian confidence                    │  │
│  │  • Novelty scoring         │  │  │  • Version control                        │  │
│  │  • GDC flags               │  │  │  • α-weight profiles                      │  │
│  └────────────────────────────┘  │  └───────────────────────────────────────────┘  │
└──────────────────────────────────┴──────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                             Retrieval Stack                                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐  │
│  │   Vector     │    │     PPR      │    │   Rel-GAT    │    │   Divergent     │  │
│  │   k-NN       │───►│  Retriever   │───►│  Rescorer    │───►│   Retriever     │  │
│  │  (Base)      │    │ (PageRank)   │    │ (α-weights)  │    │(Creative/Repair)│  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                        ┌─────────────────┴──────────────────┐
                        ▼                                    ▼
┌───────────────────────────────────┐    ┌──────────────────────────────────────────┐
│       Planning Engine             │    │         Innovator Agent                    │
├───────────────────────────────────┤    ├──────────────────────────────────────────┤
│  • Complexity classifier          │    │  • Graph Doctor (GDC-based)               │
│  • Strategy selector              │    │  • Template encoder                        │
│  • Mode: {NORMAL,CREATIVE,REPAIR}│    │  • LLM repair proposals                    │
│  • Re-planning triggers          │    │  • Divergent bridge finder                │
│  • Cost-aware search             │    │  • Hidden-link scanner                     │
└───────────────────────────────────┘    └──────────────────────────────────────────┘
                        │                                    │
                        └─────────────────┬──────────────────┘
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            Guardian Gate                                             │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐   ┌─────────────────┐   ┌──────────────────┐                  │
│  │  External Fact │   │ Semantic-Utility │   │     Policy       │                  │
│  │     Hooks      │───┤     Scoring      │───┤   Enforcement    │                  │
│  └────────────────┘   └─────────────────┘   └──────────────────┘                  │
│                               │                                                      │
│                               ▼                                                      │
│                    ┌─────────────────────┐                                          │
│                    │  Decision Engine    │                                          │
│                    │ {apply, quarantine, │                                          │
│                    │  reject}            │                                          │
│                    └─────────────────────┘                                          │
│                               │                                                      │
│  ┌────────────────────────────┼────────────────────────────┐                       │
│  │         LoRA Adapter       ▼        KG-Trie Trigger     │                       │
│  │           Signing      (if high risk)   Control         │                       │
│  └──────────────────────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                        ┌─────────────────┴──────────────────┐
                        ▼                                    ▼
┌───────────────────────────────────┐    ┌──────────────────────────────────────────┐
│     Generation Layer              │    │         LoRA-Kit                          │
├───────────────────────────────────┤    ├──────────────────────────────────────────┤
│  • Standard decoder               │    │  • Adapter training pipeline               │
│  • KG-Trie constrained decoder   │    │  • Template bank                           │
│  • Structure-safe token control  │    │  • Registry (SHA + Guardian sig)          │
│  • Confidence propagation        │    │  • Hot-swap loader                        │
└───────────────────────────────────┘    └──────────────────────────────────────────┘
```

## Component Interactions

### 1. Query Flow
```
Agent Request → MCP Auth → Planning Engine → Retrieval Stack → Guardian Gate → Generation
     ↓                          ↓                    ↓              ↓            ↓
  Auth Token              Mode Selection      Memory Access    Validation    Response
```

### 2. Creativity Mode Flow
```
Creative Query → Divergent Retriever → Hidden-Link Scan → Guardian Vetting → Tagged Results
      ↓                  ↓                    ↓                 ↓                ↓
  mode=CREATIVE    Ignore Obvious      Find Surprises      Safety Check    "creative" tag
```

### 3. Repair Workflow
```
GDC Violation → Innovator Agent → Repair Proposals → Guardian Review → Apply/Quarantine
      ↓               ↓                 ↓                  ↓               ↓
  Rule Match    Graph Doctor      LLM Generation      Utility Score    Update Hippo
```

### 4. Nightly Consolidation
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Nightly Batch Process                             │
├─────────────────────────────────────────────────────────────────────┤
│  1. Hippo-Index → Hypergraph-KG migration (stable episodic)        │
│  2. Hidden-link batch scanning (Divergent Retriever)               │
│  3. GDC full-graph validation                                       │
│  4. α-weight profile updates (Rel-GAT training)                    │
│  5. Garbage collection (expired TTL entries)                       │
│  6. Backup & versioning                                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Architectural Decisions

### 1. Brain-Inspired Dual Memory
- **Hippo-Index**: Fast, episodic memory with novelty detection (hippocampus)
- **Hypergraph-KG**: Structured, semantic memory with rich relationships (cortex)
- **Consolidation**: Nightly transfer of stable memories from Hippo to KG

### 2. Creativity & Remote Association
- **Divergent Retriever**: Deliberately ignores high-probability paths
- **Surprise Scoring**: Measures semantic distance between connected concepts
- **Hidden-Link Scanner**: Batch process to find non-obvious connections

### 3. Automatic Graph Repair
- **GDC (Graph Diagnostic Criteria)**: Rule-based violation detection
- **Innovator Agent**: LLM-powered repair proposal generation
- **Template Encoding**: Structured approach to graph modifications

### 4. Safety & Validation
- **Guardian Gate**: Multi-stage validation pipeline
- **Semantic-Utility Scoring**: Balance between novelty and usefulness
- **KG-Trie Fallback**: Constrained generation for high-risk scenarios

### 5. Personalization
- **Rel-GAT α-profiles**: User-specific attention weights
- **Digital Twin Integration**: Per-user episodic memories
- **LoRA Adapters**: Domain and user-specific model fine-tuning

## Security Architecture

### Authentication Flow
```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Agent  │────►│   TLS   │────►│   JWT   │────►│  RBAC   │
│ Client  │◄────│  mTLS   │◄────│  Token  │◄────│ Policy  │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
```

### Audit Trail
```
Every operation logged with:
- Timestamp
- Agent ID
- Operation type
- Resource accessed
- Guardian decision
- Confidence scores
- Reasoning path
```

## Performance Considerations

### 1. Caching Strategy
- **L1 Cache**: Redis for hot queries (TTL: 1 hour)
- **L2 Cache**: DuckDB materialized views (TTL: 24 hours)
- **L3 Cache**: Pre-computed PPR scores (TTL: 7 days)

### 2. Scaling Approach
- **Horizontal**: Multiple MCP server instances behind load balancer
- **Vertical**: GPU acceleration for embeddings and Rel-GAT
- **Edge Deployment**: Lite mode with reduced retrieval depth

### 3. Resource Limits
- **Divergent Search**: Max depth 3, time budget 2s
- **Repair Operations**: Batched, max 100 per cycle
- **Guardian Reviews**: Async queue with priority ordering

## Monitoring & Observability

### Key Metrics
- Query latency (p50, p95, p99)
- Creativity mode usage %
- Repair proposals (submitted/accepted/rejected)
- Guardian override frequency
- Cache hit rates
- Memory growth rates

### Health Checks
```
/health/live     - Basic connectivity
/health/ready    - All components initialized
/health/startup  - Detailed component status
```

## Deployment Architecture

### Container Structure
```
hyperag-mcp/
├── mcp-server     (Main API service)
├── hippo-index    (DuckDB + Redis)
├── hypergraph-kg  (Neo4j + Qdrant)
├── guardian-gate  (Validation service)
├── innovator      (Repair agent)
└── lora-registry  (Adapter management)
```

### Environment Variables
```yaml
HYPERAG_MODE: production
HYPERAG_AUTH_ENABLED: true
HYPERAG_GUARDIAN_THRESHOLD: 0.8
HYPERAG_CREATIVITY_DEPTH: 3
HYPERAG_REPAIR_BATCH_SIZE: 100
HYPERAG_CACHE_TTL: 3600
```

## Risk Mitigation

### 1. Creativity Mode Risks
- **Risk**: Hallucinated connections
- **Mitigation**: Guardian semantic-utility filter + confidence thresholds

### 2. Repair Operation Risks
- **Risk**: Destructive graph modifications
- **Mitigation**: Quarantine first, human review for high-impact changes

### 3. Personalization Risks
- **Risk**: Filter bubbles from α-weights
- **Mitigation**: Diversity injection, periodic α-weight reset option

### 4. Performance Risks
- **Risk**: Divergent search explosion
- **Mitigation**: Depth bounds, time budgets, circuit breakers

## Future Extensions

1. **Multi-Modal Hyperedges**: Images, audio in knowledge graph
2. **Federated Learning**: Cross-agent α-weight sharing
3. **Quantum-Inspired Retrieval**: Superposition states for ambiguous queries
4. **Neuromorphic Hardware**: Spike-based retrieval acceleration
