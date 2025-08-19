# AIVillage API Versioning & Public API Documentation

## Overview

AIVillage exposes a public API through two main services:
- **Gateway Service** (Port 8000): Entry point with rate limiting, security headers, and health cascading
- **Twin Service** (Port 8001): Core AI functionality with chat, query, upload, and debugging capabilities

## Service Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Gateway :8000 │───▶│    Twin :8001   │
│  • Rate Limiting│    │  • Chat Engine  │
│  • CORS         │    │  • RAG Pipeline │
│  • Security     │    │  • File Upload  │
│  • Health Check │    │  • Debug Tools  │
└─────────────────┘    └─────────────────┘
```

## API Versioning Strategy

### Current Version: v1

All new endpoints use the `/v1/` prefix:
- `/v1/chat` - Chat endpoint
- `/v1/query` - RAG query endpoint
- `/v1/upload` - File upload endpoint
- `/v1/evidence` - Evidence submission
- `/v1/debug/*` - Debug endpoints (non-production)

### Deprecation Middleware

The development server (`packages/core/bin/server.py`) includes comprehensive deprecation middleware that:

1. **Maps legacy routes to new endpoints:**
   ```python
   DEPRECATED_ROUTES = {
       "/query": "Use POST /v1/query via Twin service",
       "/upload": "Use POST /v1/upload via Twin service",
       "/status": "Use GET /healthz",
       "/bayes": "Use GET /v1/debug/bayes via Twin service",
       "/logs": "Use GET /v1/debug/logs via Twin service",
       "/v1/explanation": "Use POST /v1/evidence via Twin service",
       "/explain": "Use POST /explain via Twin service",
   }
   ```

2. **Adds deprecation headers:**
   - `X-Deprecated: [migration message]`
   - `X-Deprecation-Date: 2025-02-01`

3. **Logs deprecation warnings** (when not in dev mode)

## Gateway Service API (Port 8000)

### Health Check
```http
GET /healthz
```

**Response:**
```json
{
  "gateway": "ok",
  "twin": "ok|degraded",
  "details": {
    "status_code": 200,
    "response": "..."
  }
}
```

### Metrics (Prometheus)
```http
GET /metrics
```

**Response:** Prometheus metrics format
- `gw_requests_total` - Request counter by path
- `gw_rate_limited_total` - Rate limit counter
- `gw_latency_seconds` - Request latency histogram

### Chat Proxy
```http
POST /v1/chat
Content-Type: application/json
Authorization: Bearer <api-key>

{
  "message": "Hello, how can you help?",
  "conversation_id": "uuid-optional",
  "user_id": "user-123"
}
```

**Response:**
```json
{
  "response": "I can help with...",
  "conversation_id": "generated-uuid",
  "timestamp": "2025-01-15T10:30:00Z",
  "processing_time_ms": 250,
  "calibrated_prob": 0.85
}
```

### Rate Limiting
- **Limit:** 100 requests per 60 seconds per IP
- **Response:** 429 Too Many Requests
- **Headers:** `X-Process-Time` added to all responses

### Security Headers
All responses include:
- `Strict-Transport-Security: max-age=63072000; includeSubDomains`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`

## Twin Service API (Port 8001)

### Health Check
```http
GET /healthz
```

**Response:**
```json
{
  "status": "ok|unhealthy",
  "version": "0.2.0",
  "model_loaded": true,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Chat Endpoint
```http
POST /v1/chat
Content-Type: application/json

{
  "message": "What is machine learning?",
  "conversation_id": "optional-uuid",
  "user_id": "user-123"
}
```

**Response:**
```json
{
  "response": "Machine learning is...",
  "conversation_id": "conv-uuid-123",
  "timestamp": "2025-01-15T10:30:00Z",
  "processing_time_ms": 150,
  "calibrated_prob": 0.92
}
```

### RAG Query
```http
POST /v1/query
Content-Type: application/json

{
  "query": "What are the benefits of renewable energy?"
}
```

**Response:**
```json
{
  "response": "Renewable energy offers several benefits...",
  "chunks": [
    {
      "id": "chunk-123",
      "text": "Solar and wind power...",
      "score": 0.89,
      "source_uri": "doc://renewable-energy-guide"
    }
  ],
  "processing_time_ms": 320
}
```

### File Upload
```http
POST /v1/upload
Content-Type: multipart/form-data

file: [binary file data]
```

**Response:**
```json
{
  "status": "uploaded",
  "filename": "document.pdf",
  "size": 1024000,
  "message": "File processed successfully"
}
```

**File Constraints:**
- Max size: 10MB (configurable)
- Allowed types: `.pdf`, `.txt`, `.md`, `.docx`
- Content must be UTF-8 decodable

### Graph Explanation
```http
POST /explain
Content-Type: application/json

{
  "src": "concept-a",
  "dst": "concept-b",
  "hops": 5
}
```

**Response:**
```json
{
  "nodes": ["concept-a", "intermediate", "concept-b"],
  "edges": [
    {"from": "concept-a", "to": "intermediate", "weight": 0.8},
    {"from": "intermediate", "to": "concept-b", "weight": 0.7}
  ],
  "hops": 2,
  "found": true,
  "processing_ms": 45.2
}
```

### Evidence Submission
```http
POST /v1/evidence
Content-Type: application/json

{
  "id": "evidence-123",
  "query": "What is photosynthesis?",
  "chunks": [...]
}
```

**Response:**
```json
{
  "status": "ok"
}
```

### User Data Deletion (GDPR)
```http
DELETE /v1/user/{user_id}
```

**Response:**
```json
{
  "deleted_conversations": 3
}
```

## Debug Endpoints (Non-Production)

### Bayes Network Snapshot
```http
GET /v1/debug/bayes
```

### Knowledge Tracker Logs
```http
GET /v1/debug/logs
```

## Authentication

### API Key Authentication
Include API key in requests:

**Header:**
```http
Authorization: Bearer your-api-key-here
```

**Query Parameter (fallback):**
```http
GET /healthz?api_key=your-api-key-here
```

## Error Responses

### Standard Error Format
```json
{
  "detail": "Error message",
  "error_code": "VALIDATION_ERROR",
  "timestamp": "2025-01-15T10:30:00Z",
  "request_id": "req-uuid-123"
}
```

### Common HTTP Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Invalid request format or parameters |
| 401 | Unauthorized | Missing or invalid API key |
| 429 | Too Many Requests | Rate limit exceeded (100 req/60s) |
| 500 | Internal Server Error | Unexpected server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Rate Limit Response
```http
HTTP/1.1 429 Too Many Requests
X-Process-Time: 0.002

{
  "detail": "Rate limit exceeded",
  "retry_after": 45,
  "limit": 100,
  "window": 60
}
```

## OpenAPI Documentation

### Gateway OpenAPI
```http
GET /openapi.json
```

### Twin OpenAPI
```http
GET /openapi.json
```

Both services automatically generate OpenAPI 3.0 schemas with:
- Complete endpoint documentation
- Request/response schemas
- Authentication requirements
- Error response formats

## Migration Guide

### From Legacy Development Server

The development server at `packages/core/bin/server.py` provides backward compatibility but is deprecated for production use.

**Legacy → New Endpoint Mapping:**

| Legacy Route | New Route | Service |
|-------------|-----------|---------|
| `POST /query` | `POST /v1/query` | Twin |
| `POST /upload` | `POST /v1/upload` | Twin |
| `GET /status` | `GET /healthz` | Both |
| `GET /bayes` | `GET /v1/debug/bayes` | Twin |
| `GET /logs` | `GET /v1/debug/logs` | Twin |
| `GET /v1/explanation` | `POST /v1/evidence` | Twin |

### Migration Steps

1. **Update client code** to use new endpoints
2. **Add proper authentication** headers
3. **Handle new response formats** (standardized)
4. **Update error handling** for new status codes
5. **Test against Gateway** service (not development server)

## Development Setup

### Start Services
```bash
make setup
make compose-up
make run-dev
```

### Service URLs
- Gateway: http://localhost:8000
- Twin: http://localhost:8001
- Gateway Health: http://localhost:8000/healthz
- Twin Health: http://localhost:8001/healthz

### API Documentation
- Gateway Docs: http://localhost:8000/docs
- Twin Docs: http://localhost:8001/docs
- Gateway OpenAPI: http://localhost:8000/openapi.json
- Twin OpenAPI: http://localhost:8001/openapi.json

## Production Considerations

1. **Use Gateway service** as the primary entry point
2. **Enable authentication** with proper API keys
3. **Configure rate limiting** based on usage patterns
4. **Monitor health endpoints** for service availability
5. **Set up log aggregation** for error tracking
6. **Use HTTPS** in production environments
7. **Implement circuit breakers** for Twin service calls

## Examples

### Python Client
```python
import httpx

class AIVillageClient:
    def __init__(self, base_url: str, api_key: str):
        self.client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"}
        )

    async def chat(self, message: str, conversation_id: str = None):
        response = await self.client.post("/v1/chat", json={
            "message": message,
            "conversation_id": conversation_id
        })
        response.raise_for_status()
        return response.json()

    async def query(self, query: str):
        response = await self.client.post("/v1/query", json={
            "query": query
        })
        response.raise_for_status()
        return response.json()

# Usage
client = AIVillageClient("http://localhost:8000", "your-api-key")
result = await client.chat("Hello, how are you?")
```

### cURL Examples

**Chat:**
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"message": "What is artificial intelligence?"}'
```

**Query:**
```bash
curl -X POST http://localhost:8001/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Benefits of solar energy"}'
```

**Health Check:**
```bash
curl http://localhost:8000/healthz
```

This documentation covers the complete public API surface with versioning, authentication, error handling, and migration paths from the legacy development server.
