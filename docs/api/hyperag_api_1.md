# HypeRAG API Design

## 4. API Specifications

### Base Configuration

```yaml
API:
  version: v1
  base_url: https://api.hyperag.aivillage/v1

  protocols:
    - HTTPS with TLS 1.3
    - HTTP/2 for multiplexing
    - WebSocket for streaming

  authentication:
    - Bearer token (JWT)
    - API key (legacy)
    - mTLS (enterprise)

  rate_limiting:
    default: 100 req/min
    authenticated: 1000 req/min
    enterprise: 10000 req/min
```

### API Routes

| Route | Method | Auth | Description |
|-------|--------|------|-------------|
| `/v1/hyperag/query` | POST | read | Normal/creative/repair queries with mode selection |
| `/v1/hyperag/creative` | POST | read | Divergent bridge search between concepts |
| `/v1/hyperag/repair` | POST | write | Submit GDC violations for repair |
| `/v1/hyperag/guardian/validate` | POST | write | Manually vet proposal sets |
| `/v1/hyperag/adapter/upload` | POST | write | Upload LoRA adapter for Guardian signing |
| `/v1/hyperag/adapter/list` | GET | read | List available adapters |
| `/v1/hyperag/adapter/{id}/activate` | POST | read | Activate specific adapter |
| `/v1/hyperag/knowledge/add` | POST | write | Add knowledge to Hippo-Index |
| `/v1/hyperag/knowledge/search` | POST | read | Search knowledge base |
| `/v1/hyperag/graph/visualize` | GET | read | Get graph visualization data |
| `/v1/hyperag/health` | GET | none | Health check endpoint |
| `/v1/hyperag/metrics` | GET | admin | Prometheus metrics |

### Detailed Route Specifications

#### POST /v1/hyperag/query

Standard query interface with mode selection and planning hints.

**Request:**
```json
{
  "query": "What are the connections between quantum computing and biological neural networks?",
  "mode": "NORMAL",  // NORMAL | CREATIVE | REPAIR
  "user_id": "user_123",
  "plan_hints": {
    "max_depth": 3,
    "time_budget_ms": 2000,
    "confidence_threshold": 0.7,
    "include_explanations": true
  },
  "context": {
    "previous_queries": ["quantum computing basics"],
    "domain": "neuroscience",
    "language": "en"
  },
  "personalization": {
    "use_alpha_profile": true,
    "adapter_id": "neuroscience_expert_v2"
  }
}
```

**Response:**
```json
{
  "request_id": "req_abc123",
  "status": "success",
  "mode_used": "NORMAL",
  "timing": {
    "total_ms": 1523,
    "retrieval_ms": 234,
    "reasoning_ms": 1089,
    "generation_ms": 200
  },
  "result": {
    "answer": "Quantum computing and biological neural networks share several fascinating connections...",
    "confidence": 0.85,
    "reasoning_path": [
      {
        "step": 1,
        "action": "vector_retrieval",
        "entities": ["quantum_computing", "neural_networks"],
        "confidence": 0.9
      },
      {
        "step": 2,
        "action": "ppr_expansion",
        "discovered": ["quantum_biology", "coherence"],
        "confidence": 0.8
      }
    ],
    "sources": [
      {
        "id": "doc_789",
        "title": "Quantum Effects in Neural Microtubules",
        "relevance": 0.92
      }
    ],
    "hyperedges_used": [
      {
        "id": "he_456",
        "relation": "theoretical_connection",
        "entities": ["quantum_tunneling", "synaptic_transmission"],
        "confidence": 0.75
      }
    ]
  },
  "guardian_decision": {
    "action": "APPLY",
    "semantic_score": 0.89,
    "utility_score": 0.91,
    "safety_score": 0.95,
    "policy_checks": ["content_safe", "factually_grounded"]
  },
  "metadata": {
    "cache_hit": false,
    "fallback_used": false,
    "kg_trie_activated": false
  }
}
```

#### POST /v1/hyperag/creative

Specialized endpoint for creative/divergent thinking tasks.

**Request:**
```json
{
  "source_concept": "machine learning",
  "target_concept": "jazz improvisation",
  "creativity_parameters": {
    "mode": "analogical",  // divergent | analogical | combinatorial
    "max_hops": 5,
    "min_surprise": 0.7,
    "time_budget_ms": 5000,
    "avoid_paths": ["music_ai", "algorithmic_composition"]
  },
  "user_id": "user_123"
}
```

**Response:**
```json
{
  "request_id": "req_creative_789",
  "status": "success",
  "bridges_found": [
    {
      "id": "bridge_001",
      "path": [
        "machine_learning",
        "pattern_recognition",
        "improvisation_patterns",
        "jazz_improvisation"
      ],
      "relations": [
        "enables",
        "similar_to",
        "fundamental_to"
      ],
      "surprise_score": 0.82,
      "confidence": 0.73,
      "explanation": "Both ML and jazz improvisation involve recognizing and creatively varying patterns...",
      "tags": ["analogy", "pattern-based"]
    },
    {
      "id": "bridge_002",
      "path": [
        "machine_learning",
        "exploration_exploitation",
        "musical_exploration",
        "jazz_improvisation"
      ],
      "surprise_score": 0.91,
      "confidence": 0.68,
      "explanation": "The exploration-exploitation tradeoff in ML mirrors jazz musicians' balance...",
      "tags": ["metaphor", "process-similarity"]
    }
  ],
  "computation_time_ms": 4821,
  "guardian_vetted": true,
  "creative_metrics": {
    "novelty": 0.87,
    "usefulness": 0.79,
    "surprise": 0.85
  }
}
```

#### POST /v1/hyperag/repair

Submit graph inconsistencies for automated repair.

**Request:**
```json
{
  "violation_type": "gdc_violation",  // gdc_violation | manual_report | anomaly
  "details": {
    "gdc_spec_id": "gdc_consistency_003",
    "violation_id": "viol_123",
    "affected_entities": ["entity_a", "entity_b"],
    "description": "Missing inverse relation between CAUSES and CAUSED_BY"
  },
  "proposed_action": "add_inverse_relation",
  "priority": "medium",
  "requester": "agent_king"
}
```

**Response:**
```json
{
  "request_id": "req_repair_456",
  "status": "success",
  "repair_proposals": [
    {
      "id": "proposal_001",
      "description": "Add CAUSED_BY relation from entity_b to entity_a",
      "confidence": 0.92,
      "impact_analysis": {
        "affected_nodes": 2,
        "affected_edges": 1,
        "consistency_improvement": 0.15
      },
      "cypher_query": "MATCH (a:Entity {id: 'entity_a'}), (b:Entity {id: 'entity_b'}) CREATE (b)-[:CAUSED_BY {confidence: 0.8}]->(a)"
    },
    {
      "id": "proposal_002",
      "description": "Adjust confidence scores for consistency",
      "confidence": 0.78,
      "alternative": true
    }
  ],
  "guardian_review": {
    "recommendation": "APPLY",
    "reasoning": "Proposal maintains graph consistency without information loss"
  },
  "estimated_execution_time_ms": 150
}
```

#### POST /v1/hyperag/guardian/validate

Manual validation interface for Guardian reviews.

**Request:**
```json
{
  "validation_request": {
    "type": "repair_proposal",  // repair_proposal | creative_result | adapter_upload
    "id": "proposal_001",
    "override_automatic": true,
    "reviewer_notes": "Domain expert approval required"
  },
  "decision": "APPROVE",  // APPROVE | REJECT | QUARANTINE
  "conditions": {
    "quarantine_ttl_hours": 24,
    "require_second_review": false
  },
  "reviewer_id": "expert_001"
}
```

**Response:**
```json
{
  "validation_id": "val_789",
  "status": "success",
  "original_decision": "QUARANTINE",
  "new_decision": "APPROVE",
  "decision_record": {
    "timestamp": "2024-01-20T10:30:00Z",
    "reviewer": "expert_001",
    "reasoning": "Manual review confirms factual accuracy",
    "policy_overrides": ["auto_quarantine_creative"]
  },
  "action_taken": "Applied to knowledge graph",
  "audit_trail_id": "audit_123"
}
```

#### POST /v1/hyperag/adapter/upload

Upload domain or user-specific LoRA adapters.

**Request (multipart/form-data):**
```
POST /v1/hyperag/adapter/upload
Content-Type: multipart/form-data

--boundary
Content-Disposition: form-data; name="metadata"
Content-Type: application/json

{
  "name": "medical_terminology_v3",
  "description": "Medical domain adapter trained on PubMed",
  "domain": "medical",
  "base_model": "llama2-7b",
  "training_config": {
    "rank": 16,
    "alpha": 32,
    "samples": 50000,
    "validation_score": 0.94
  }
}

--boundary
Content-Disposition: form-data; name="weights"; filename="adapter_weights.bin"
Content-Type: application/octet-stream

[Binary adapter weights]

--boundary--
```

**Response:**
```json
{
  "adapter_id": "adapt_med_789",
  "status": "success",
  "validation_results": {
    "checksum_valid": true,
    "format_valid": true,
    "compatibility_check": "passed",
    "performance_baseline": 0.92
  },
  "guardian_signature": {
    "status": "signed",
    "signature": "sha256:abcd1234...",
    "signed_at": "2024-01-20T11:00:00Z",
    "validity_period_days": 90
  },
  "storage_location": "s3://hyperag-adapters/medical/adapt_med_789",
  "activation_url": "/v1/hyperag/adapter/adapt_med_789/activate"
}
```

### Common Response Envelopes

All responses follow a consistent structure:

```json
{
  "request_id": "unique_request_identifier",
  "status": "success | error | partial",
  "timestamp": "ISO8601 timestamp",
  "version": "v1",

  "result": {
    // Route-specific response data
  },

  "confidence": 0.0-1.0,

  "guardian_decision": {
    "action": "APPLY | QUARANTINE | REJECT",
    "semantic_score": 0.0-1.0,
    "utility_score": 0.0-1.0,
    "safety_score": 0.0-1.0,
    "reasoning": "Human-readable explanation"
  },

  "reasoning_path": [
    // Step-by-step reasoning trace
  ],

  "metadata": {
    "processing_time_ms": 1234,
    "cache_hit": boolean,
    "model_version": "string",
    "adapter_used": "adapter_id or null"
  },

  "error": {  // Only present on errors
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {},
    "trace_id": "debugging_trace_id"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTH_REQUIRED` | 401 | Missing or invalid authentication |
| `PERMISSION_DENIED` | 403 | Insufficient permissions for operation |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `INVALID_REQUEST` | 400 | Malformed request |
| `MODE_UNSUPPORTED` | 400 | Requested mode not available |
| `CONFIDENCE_TOO_LOW` | 422 | Result confidence below threshold |
| `GUARDIAN_REJECTED` | 422 | Guardian Gate rejected operation |
| `TIMEOUT` | 504 | Operation exceeded time budget |
| `INTERNAL_ERROR` | 500 | Server error |

### Streaming Endpoints

For real-time updates, WebSocket connections are available:

```javascript
// WebSocket connection
ws://api.hyperag.aivillage/v1/stream

// Subscribe to reasoning updates
{
  "action": "subscribe",
  "channel": "reasoning_steps",
  "query_id": "req_abc123"
}

// Receive updates
{
  "channel": "reasoning_steps",
  "query_id": "req_abc123",
  "step": 3,
  "action": "exploring_creative_path",
  "confidence": 0.72,
  "entities_discovered": ["new_concept"],
  "timestamp": "2024-01-20T10:30:45.123Z"
}
```

### Authentication Details

#### JWT Token Structure
```json
{
  "sub": "user_123",
  "role": "sage",
  "permissions": ["read", "write"],
  "adapters": ["medical_v3", "physics_v2"],
  "exp": 1705750000,
  "iat": 1705746400,
  "iss": "hyperag.aivillage"
}
```

#### API Key Header
```
X-API-Key: hrag_prod_1234567890abcdef
```

#### mTLS Certificate Requirements
- Certificate must be signed by trusted CA
- CN must match registered agent/user ID
- Certificate must not be revoked

### Rate Limiting Headers

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1705750000
X-RateLimit-Reset-After: 3600
```

### CORS Configuration

```
Access-Control-Allow-Origin: https://app.aivillage
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization, X-API-Key
Access-Control-Max-Age: 86400
```

### OpenAPI Specification

Full OpenAPI 3.0 specification available at:
```
GET /v1/hyperag/openapi.json
```

### SDK Support

Official SDKs available for:
- Python: `pip install hyperag-sdk`
- JavaScript/TypeScript: `npm install @aivillage/hyperag`
- Go: `go get github.com/aivillage/hyperag-go`
- Rust: `cargo add hyperag`
