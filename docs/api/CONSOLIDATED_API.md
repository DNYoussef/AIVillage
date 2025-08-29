# API - Consolidated Documentation

## 🎯 API Architecture Overview

AIVillage exposes a multi-tiered API architecture designed for distributed AI operations:

- **Gateway Service (Port 8000)**: API entry point with rate limiting, security headers, and health monitoring
- **Twin Service (Port 8001)**: Core AI functionality with chat, RAG, upload, and debugging capabilities
- **Development Server**: Legacy development server with deprecation middleware (deprecated for production)
- **Internal APIs**: Service-to-service communication for microservices architecture

### Service Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Gateway :8000 │───▶│    Twin :8001   │
│  • Rate Limiting│    │  • Chat Engine  │
│  • CORS         │    │  • RAG Pipeline │
│  • Security     │    │  • File Upload  │
│  • Health Check │    │  • Debug Tools  │
└─────────────────┘    └─────────────────┘
```

## 🌐 API Specifications

### REST APIs

#### Production APIs

**Agent Forge Model Server** (`http://localhost:8080`)
- Production model serving infrastructure with FastAPI
- Stable, versioned endpoints for production use

**Gateway Service** (`http://localhost:8000`)
- Entry point with rate limiting and security
- Proxies requests to appropriate backend services
- Health monitoring and metrics collection

**Twin Service** (`http://localhost:8001`)
- Core AI functionality and RAG processing
- Chat, query, upload, and debugging endpoints
- Digital twin processing and evidence submission

**Communications Credit System** (`http://localhost:8081`)
- Mesh networking and resource allocation
- User credit management and transactions
- P2P network status and peer management

#### Core Endpoints

**Health Check**
```http
GET /healthz
```
Response:
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

**Chat with AI Agents**
```http
POST /v1/chat
Authorization: Bearer <api-key>
Content-Type: application/json

{
  "message": "Hello, how can you help?",
  "conversation_id": "uuid-optional",
  "agent_preference": "magi",
  "mode": "comprehensive",
  "user_context": {
    "device_type": "mobile",
    "battery_level": 75,
    "network_type": "wifi"
  }
}
```

**RAG Query Processing**
```http
POST /v1/query
Content-Type: application/json

{
  "query": "What are the benefits of renewable energy?",
  "mode": "analytical",
  "include_sources": true,
  "max_results": 5
}
```

**File Upload**
```http
POST /v1/upload
Content-Type: multipart/form-data

file: [binary file data]
```

**Agent Management**
```http
GET /agents?category=knowledge&available_only=true

POST /agents/{agent_id}/task
{
  "task_description": "Research quantum computing applications",
  "priority": "high",
  "timeout_seconds": 600
}
```

### WebSocket APIs

**Future real-time communication endpoints (planned):**

**Model Streaming**
```javascript
const ws = new WebSocket('ws://localhost:8080/stream');
ws.send(JSON.stringify({
  type: 'generate',
  prompt: 'Tell me a story...',
  stream: true
}));
```

**Agent Communication**
```javascript
const ws = new WebSocket('ws://localhost:8001/agent-stream');
ws.send(JSON.stringify({
  type: 'chat',
  agent: 'magi',
  message: 'Research quantum computing'
}));
```

### GraphQL APIs

No GraphQL APIs are currently implemented, but the OpenAPI specification provides comprehensive REST API coverage.

## 📋 API Standards & Conventions

### Authentication & Authorization

**Development Authentication**
```bash
# Using API key header
curl -H "X-API-Key: your-dev-key" "http://localhost:8000/v1/chat"
```

**Production Authentication**
```bash
# Using bearer token
curl -H "Authorization: Bearer your-token" "http://localhost:8080/generate"
```

**Supported Methods:**
1. **Bearer Token** (recommended):
   ```
   Authorization: Bearer your-api-key
   ```
2. **API Key Header**:
   ```
   x-api-key: your-api-key
   ```

### Versioning Strategy

**Current Version: v1**

All new endpoints use the `/v1/` prefix:
- `/v1/chat` - Chat endpoint
- `/v1/query` - RAG query endpoint
- `/v1/upload` - File upload endpoint
- `/v1/evidence` - Evidence submission
- `/v1/debug/*` - Debug endpoints (non-production)

**Deprecation Policy:**
- 6 months notice for breaking changes
- 12 months support for deprecated endpoints
- Migration guides provided for version transitions

**API Versioning follows semantic versioning:**
- `/v1/`: Stable API, backward compatible changes only
- `/v2beta/`: Beta API, may have breaking changes
- `/experimental/`: Experimental endpoints, no stability guarantees

### Error Handling

**Standard Error Format**
```json
{
  "detail": "Error message",
  "error_code": "VALIDATION_ERROR",
  "timestamp": "2025-01-15T10:30:00Z",
  "request_id": "req-uuid-123"
}
```

**Common Error Codes:**
- `INVALID_REQUEST`: Malformed request data
- `AUTHENTICATION_FAILED`: Invalid or missing credentials
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `RESOURCE_NOT_FOUND`: Requested resource doesn't exist
- `INTERNAL_ERROR`: Server-side error
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable

## 🔧 SDK & Integration

### SDK Documentation

**Available SDKs:**

| Language | Package Name | Version | Status |
|----------|--------------|---------|--------|
| **TypeScript/JavaScript** | `aivillage-client` | 1.0.0 | Planned |
| **Python** | `aivillage-client` | 1.0.0 | Planned |
| **Java** | `io.aivillage:aivillage-client` | 1.0.0 | Planned |
| **Swift** | `AIVillageClient` | 1.0.0 | Planned |
| **Kotlin** | `io.aivillage:aivillage-client-kotlin` | 1.0.0 | Planned |
| **Go** | `github.com/DNYoussef/AIVillage/clients/go` | 1.0.0 | Planned |
| **Rust** | `aivillage-client` | 1.0.0 | Planned |

### Integration Guides

**Python SDK Example (Planned)**
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

**JavaScript SDK Example (Planned)**
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

### Protocol Implementation

**gRPC Integration (Planned)**
- **Current Status**: HTTP-based communication
- **Planned**: gRPC primary with WebSocket fallback using JSON-encoded protobuf messages
- Connection manager negotiates highest-capability transport

**Messaging Protocol (ADR-0002)**
- gRPC primary with automatic WebSocket fallback
- Edge devices behind firewalls supported
- Future QUIC support planned

## 🎯 API Quality & Performance

### Performance Standards

**Rate Limiting:**
- **Production APIs**: 1000 requests/hour per API key
- **Development APIs**: 100 requests/minute per IP
- **Gateway Service**: 100 requests per 60 seconds per IP
- **Authenticated**: 200 requests per 60 seconds
- **Premium**: 500 requests per 60 seconds

**Response Time Targets:**
- Chat responses: <2 seconds
- RAG queries: <5 seconds
- File uploads: <30 seconds
- Health checks: <500ms

**Reliability Targets:**
- 99.9% uptime for production APIs
- 99.5% uptime for development APIs
- Graceful degradation during outages

### Validation & Testing

**API Testing Coverage:**
- Unit tests for all endpoints
- Integration tests for service communication
- Load testing for performance validation
- Security testing for vulnerability assessment

**Health Check Endpoints:**
All services expose health check endpoints:
- `GET /health`: Basic health status
- `GET /healthz`: Kubernetes-style health check
- `GET /ready`: Readiness probe
- `GET /metrics`: Prometheus metrics

**Monitoring:**
- Request/response logging at INFO level
- Error logging at ERROR level
- Performance metrics at DEBUG level
- Security events at WARN level

---

## ❌ API REALITY GAP

### Implementation Gaps

#### 1. **Service Architecture Gap**:
- **Documented**: Production Gateway + Twin microservices architecture
- **Reality**: Development server (`server.py`) with deprecation warnings
- **Evidence**: Development server code shows "DEVELOPMENT ONLY" warnings and redirects to planned microservices

#### 2. **Endpoint Implementation Gap**:
- **Documented**: `/v1/chat`, `/v1/query`, `/v1/upload` endpoints
- **Reality**: Legacy endpoints `/query`, `/upload` with deprecation middleware
- **Evidence**: Deprecation middleware maps old routes to new service endpoints

#### 3. **Authentication Gap**:
- **Documented**: Bearer token and API key authentication
- **Reality**: Optional authentication in development server
- **Evidence**: `API_KEY not set - running without authentication` warning

#### 4. **Microservices Gap**:
- **Documented**: Gateway (8000) + Twin (8001) + Agent Forge (8080) services
- **Reality**: Experimental/development implementations only
- **Evidence**: Services located in `/experiments/services/` and `/infrastructure/shared/experimental/`

### Documentation Gaps

#### 1. **Schema Validation Gap**:
- **Documented**: OpenAPI 3.0 specification with comprehensive schemas
- **Reality**: Basic FastAPI validation with custom security models
- **Evidence**: `SecureQueryRequest` and `SecureUploadFile` models in development server

#### 2. **SDK Availability Gap**:
- **Documented**: 7 language SDKs available
- **Reality**: All SDKs marked as "Planned" - none actually implemented
- **Evidence**: SDK documentation shows planned features only

#### 3. **WebSocket/gRPC Gap**:
- **Documented**: WebSocket streaming and gRPC with fallback
- **Reality**: HTTP-only implementation
- **Evidence**: ADR-0002 shows "Implementation Pending" status

#### 4. **P2P Integration Gap**:
- **Documented**: P2P mesh networking with credit system
- **Reality**: P2P components exist but not integrated with API layer
- **Evidence**: Separate P2P implementations without API bindings

### Critical API Debt

#### Missing Core Services:
1. **Gateway Service**: Only experimental implementation exists
2. **Twin Service**: Experimental implementation with limited functionality
3. **Agent Forge Production API**: Model server referenced but not implemented
4. **Credit System API**: P2P credit functionality not exposed via API

#### Performance Shortfalls:
1. **Rate Limiting**: Basic implementation vs. documented advanced features
2. **Health Monitoring**: Simple status checks vs. comprehensive health cascade
3. **Error Handling**: Basic FastAPI errors vs. standardized error system

#### Security Vulnerabilities:
1. **Optional Authentication**: Development server runs without API keys
2. **File Upload Security**: Basic validation vs. enterprise-grade security
3. **Request Sanitization**: HTML escaping only vs. comprehensive sanitization

### API Resolution Priorities

#### 1. **CRITICAL** (Immediate - 0-2 weeks):
- **Implement Gateway Service**: Move from experimental to production-ready
- **Standardize Authentication**: Enforce API key validation across all endpoints
- **Fix Development Server Warnings**: Either complete microservice migration or stabilize dev server

#### 2. **HIGH** (1-3 months):
- **Complete Twin Service**: Implement all documented endpoints with proper error handling
- **Implement SDK Foundation**: At least Python and JavaScript SDKs
- **API Testing Infrastructure**: Comprehensive test coverage for all endpoints

#### 3. **MEDIUM** (3-6 months):
- **WebSocket/gRPC Implementation**: Real-time communication protocols
- **P2P API Integration**: Expose P2P functionality through REST APIs
- **Performance Optimization**: Meet documented performance targets

#### 4. **ENHANCEMENT** (6+ months):
- **Complete SDK Ecosystem**: All 7 planned language SDKs
- **Advanced Features**: Digital twin profiles, agent task management
- **Production Hardening**: Enterprise-grade security and monitoring

### Recommended Next Steps

1. **Audit Current Implementation**: Complete inventory of working vs. documented APIs
2. **Prioritize Core Services**: Focus on Gateway + Twin service completion
3. **Establish API Testing**: Prevent future documentation drift
4. **Update Documentation**: Align docs with current reality while implementing gaps
5. **Create Migration Plan**: Smooth transition from development to production APIs

The AIVillage API ecosystem shows significant potential but requires substantial implementation work to match the comprehensive documentation. The foundation exists, but production-ready services need completion and integration.
