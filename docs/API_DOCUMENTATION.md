# AIVillage API Documentation

This document provides comprehensive API documentation for all AIVillage services and endpoints.

## Overview

AIVillage exposes multiple API layers:
- **Production APIs**: Stable, versioned endpoints for production use
- **Infrastructure APIs**: Core system management and deployment
- **Experimental APIs**: Development endpoints with evolving schemas
- **Internal APIs**: Inter-service communication endpoints

## Production APIs

### Agent Forge Model Server

**Base URL**: `http://localhost:8080` (configurable)

Production model serving infrastructure with FastAPI.

#### POST /generate
Generate text using deployed models.

**Request Body**:
```json
{
  "prompt": "string",
  "max_tokens": 100,
  "temperature": 0.7,
  "model_id": "string (optional)"
}
```

**Response**:
```json
{
  "text": "generated text",
  "tokens_used": 42,
  "model_id": "model_identifier",
  "generation_time": 1.23
}
```

**Example**:
```bash
curl -X POST "http://localhost:8080/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "max_tokens": 150,
    "temperature": 0.8
  }'
```

#### GET /health
Health check endpoint for monitoring.

**Response**:
```json
{
  "status": "healthy",
  "uptime": 3600,
  "models_loaded": 2,
  "memory_usage": "2.1GB",
  "timestamp": "2025-07-31T10:00:00Z"
}
```

#### GET /info
Model server information and capabilities.

**Response**:
```json
{
  "server_version": "1.0.0",
  "loaded_models": [
    {
      "model_id": "compressed_llama",
      "size": "4.2GB",
      "compression_ratio": "4.8x",
      "status": "ready"
    }
  ],
  "supported_features": ["compression", "quantization", "batching"]
}
```

#### GET /metrics
Prometheus-compatible metrics endpoint.

**Response**: Prometheus metrics format
```
# HELP model_inference_requests_total Total number of inference requests
# TYPE model_inference_requests_total counter
model_inference_requests_total 1234

# HELP model_inference_duration_seconds Time spent on inference
# TYPE model_inference_duration_seconds histogram
model_inference_duration_seconds_bucket{le="0.1"} 100
```

### Communications Credit System

**Base URL**: `http://localhost:8081` (configurable)

Mesh networking and resource allocation system.

#### POST /users
Create a new user in the credit system.

**Request Body**:
```json
{
  "username": "alice",
  "initial_balance": 1000
}
```

**Response**:
```json
{
  "username": "alice",
  "balance": 1000,
  "created_at": "2025-07-31T10:00:00Z",
  "user_id": "uuid-string"
}
```

#### GET /balance/{username}
Get current credit balance for a user.

**Response**:
```json
{
  "username": "alice",
  "balance": 750,
  "last_updated": "2025-07-31T10:30:00Z"
}
```

#### POST /transfer
Transfer credits between users.

**Request Body**:
```json
{
  "from_user": "alice",
  "to_user": "bob",
  "amount": 100,
  "memo": "Payment for computation"
}
```

**Response**:
```json
{
  "transaction_id": "txn_12345",
  "status": "completed",
  "from_balance": 650,
  "to_balance": 350,
  "timestamp": "2025-07-31T10:45:00Z"
}
```

#### POST /earn
Earn credits for computational work.

**Request Body**:
```json
{
  "username": "alice",
  "task_type": "model_compression",
  "work_units": 10,
  "quality_score": 0.95
}
```

**Response**:
```json
{
  "username": "alice",
  "credits_earned": 95,
  "new_balance": 745,
  "work_verified": true
}
```

#### GET /transactions/{username}
Get transaction history for a user.

**Query Parameters**:
- `limit`: Number of transactions to return (default: 50)
- `offset`: Pagination offset (default: 0)

**Response**:
```json
[
  {
    "transaction_id": "txn_12345",
    "type": "transfer",
    "amount": -100,
    "balance_after": 650,
    "timestamp": "2025-07-31T10:45:00Z",
    "memo": "Payment for computation"
  }
]
```

#### GET /supply
Get total credit supply information.

**Response**:
```json
{
  "total_supply": 1000000,
  "circulating_supply": 750000,
  "locked_supply": 250000,
  "active_users": 1234
}
```

## Experimental APIs

### Gateway Service (Development)

**Base URL**: `http://localhost:8000`

API gateway for experimental microservices.

#### POST /v1/chat
Main chat endpoint with routing to appropriate services.

**Request Body**:
```json
{
  "message": "What is the weather today?",
  "user_id": "user_123",
  "context": {
    "session_id": "session_456",
    "preferences": {}
  }
}
```

**Response**:
```json
{
  "response": "I don't have access to real-time weather data...",
  "agent": "sage",
  "confidence": 0.85,
  "processing_time": 1.2
}
```

### Twin Service (Development)

**Base URL**: `http://localhost:8001`

Digital twin processing service.

#### POST /v1/chat
Process chat messages with AI agents.

**Request Body**:
```json
{
  "message": "Explain neural networks",
  "agent_type": "sage",
  "context": {}
}
```

**Response**:
```json
{
  "response": "Neural networks are computational models...",
  "agent": "sage",
  "sources": ["internal_knowledge", "retrieved_docs"],
  "confidence": 0.92
}
```

#### POST /v1/query
RAG query processing with document retrieval.

**Request Body**:
```json
{
  "query": "How does model compression work?",
  "top_k": 5,
  "threshold": 0.7
}
```

**Response**:
```json
{
  "answer": "Model compression reduces model size...",
  "sources": [
    {
      "document": "compression_guide.pdf",
      "relevance": 0.95,
      "excerpt": "Relevant text excerpt..."
    }
  ],
  "retrieved_count": 3
}
```

#### POST /v1/upload
Upload documents for vector store indexing.

**Request Body**: Multipart form data
- `file`: Document file
- `metadata`: JSON metadata (optional)

**Response**:
```json
{
  "document_id": "doc_12345",
  "status": "indexed",
  "chunks_created": 15,
  "processing_time": 3.2
}
```

#### GET /v1/embeddings
Generate embeddings for text (placeholder).

**Query Parameters**:
- `text`: Text to embed

**Response**:
```json
{
  "embeddings": [0.1, -0.3, 0.7, ...],
  "dimensions": 768,
  "model": "sentence-transformers"
}
```

#### POST /v1/evidence
Submit evidence pack for analysis.

**Request Body**:
```json
{
  "evidence": {
    "type": "performance_metrics",
    "data": {...},
    "timestamp": "2025-07-31T10:00:00Z"
  }
}
```

**Response**:
```json
{
  "evidence_id": "evidence_12345",
  "analysis": "preliminary_results",
  "confidence": 0.78
}
```

### Wave Bridge Service (Development)

**Base URL**: `http://localhost:8082`

Advanced tutoring system with prompt engineering.

#### POST /whatsapp/webhook
WhatsApp integration webhook (experimental).

**Request Body**: WhatsApp webhook format

**Response**: Status acknowledgment

## Internal APIs

### Production Component APIs

#### Compression API

```python
from production.compression.model_compression import CompressionPipeline

# Programmatic API
pipeline = CompressionPipeline()
compressed_model = pipeline.compress(model, method='bitnet')
```

#### Evolution API

```python
from production.evolution.evomerge import EvolutionaryTournament

# Programmatic API
tournament = EvolutionaryTournament()
winner = tournament.evolve_population(models)
```

#### RAG API

```python
from production.rag.rag_system import RAGPipeline

# Programmatic API
rag = RAGPipeline()
rag.index_documents(documents)
response = rag.generate_with_context(query)
```

## Authentication & Security

### Development Authentication

For development endpoints, authentication is optional but recommended:

```bash
# Using API key header
curl -H "X-API-Key: your-dev-key" "http://localhost:8000/v1/chat"
```

### Production Authentication

Production endpoints require authentication:

```bash
# Using bearer token
curl -H "Authorization: Bearer your-token" "http://localhost:8080/generate"
```

## Error Handling

All APIs use consistent error response format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Detailed error message",
    "details": {
      "field": "validation error details"
    }
  },
  "timestamp": "2025-07-31T10:00:00Z",
  "request_id": "req_12345"
}
```

### Common Error Codes

- `INVALID_REQUEST`: Malformed request data
- `AUTHENTICATION_FAILED`: Invalid or missing credentials  
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `RESOURCE_NOT_FOUND`: Requested resource doesn't exist
- `INTERNAL_ERROR`: Server-side error
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable

## Rate Limiting

APIs implement rate limiting to ensure fair usage:

- **Production APIs**: 1000 requests/hour per API key
- **Development APIs**: 100 requests/minute per IP
- **Internal APIs**: No rate limiting

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1643723400
```

## WebSocket APIs (Planned)

Future real-time communication endpoints:

### Model Streaming
```javascript
const ws = new WebSocket('ws://localhost:8080/stream');
ws.send(JSON.stringify({
  type: 'generate',
  prompt: 'Tell me a story...',
  stream: true
}));
```

### Agent Communication
```javascript
const ws = new WebSocket('ws://localhost:8001/agent-stream');
ws.send(JSON.stringify({
  type: 'chat',
  agent: 'magi',
  message: 'Research quantum computing'
}));
```

## SDK and Client Libraries

### Python SDK (Planned)

```python
from aivillage import AIVillageClient

client = AIVillageClient(api_key="your-key")

# Model inference
response = client.generate("Explain AI", max_tokens=100)

# Credit operations
balance = client.credits.get_balance("username")
client.credits.transfer("alice", "bob", 100)

# Document upload
doc_id = client.documents.upload("path/to/document.pdf")
```

### JavaScript SDK (Planned)

```javascript
import { AIVillageClient } from 'aivillage-js';

const client = new AIVillageClient({ apiKey: 'your-key' });

// Model inference
const response = await client.generate({
  prompt: 'Explain AI',
  maxTokens: 100
});

// Credit operations
const balance = await client.credits.getBalance('username');
```

## Monitoring and Observability

### Health Check Endpoints

All services expose health check endpoints:
- `GET /health`: Basic health status
- `GET /healthz`: Kubernetes-style health check
- `GET /ready`: Readiness probe
- `GET /metrics`: Prometheus metrics

### Logging

All APIs log requests and responses:
- Request/response logging at INFO level
- Error logging at ERROR level
- Performance metrics at DEBUG level
- Security events at WARN level

### Tracing

Distributed tracing support (planned):
- OpenTelemetry integration
- Jaeger-compatible traces
- Cross-service correlation IDs

## API Versioning

APIs follow semantic versioning:
- `/v1/`: Stable API, backward compatible changes only
- `/v2beta/`: Beta API, may have breaking changes
- `/experimental/`: Experimental endpoints, no stability guarantees

Deprecation policy:
- 6 months notice for breaking changes
- 12 months support for deprecated endpoints
- Migration guides provided for version transitions

---

This documentation is continuously updated as the AIVillage platform evolves. For the latest API specifications, see the interactive API docs at `/docs` endpoints on each service.