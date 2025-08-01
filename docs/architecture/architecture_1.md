# Architecture

The repository started as a monolithic Python project. It now includes two
microservices – a Gateway and the Twin runtime – which communicate over HTTP.
gRPC/WebSocket support proposed in ADR-0002 is not yet implemented.
Most other modules remain in-process libraries. Quiet-STaR and expert vectors
remain stubs, while a basic ADAS prototype is available.

## Service Architecture

### Production Services
- **Gateway Service** (port 8000): Entry point for all client requests, handles authentication and rate limiting
- **Twin Service** (port 8001): Core AI processing service, manages conversations and RAG pipeline

### Development/Test Only
- **server.py**: Monolithic development server restricted to test harness use only (see ADR-0010)
  - Requires `AIVILLAGE_DEV_MODE=true` environment variable
  - All production routes have been migrated to microservices
  - Emits deprecation warnings when accessed without dev mode

## API Routes

### Gateway Service (`/services/gateway`)
- `GET /healthz` - Health check endpoint
- `GET /metrics` - Prometheus metrics
- `POST /v1/chat` - Main chat endpoint (proxies to Twin)

### Twin Service (`/services/twin`)
- `POST /v1/chat` - Process chat messages
- `POST /v1/query` - RAG query processing (migrated from server.py)
- `POST /v1/upload` - Document upload for vector store (migrated from server.py)
- `GET /v1/embeddings` - Embeddings endpoint (placeholder)
- `POST /v1/evidence` - Evidence pack submission
- `POST /explain` - Graph explanation between nodes
- `GET /healthz` - Health check endpoint
- `GET /metrics` - Prometheus metrics
- `DELETE /v1/user/{user_id}` - User data deletion (privacy)
- `GET /v1/debug/bayes` - Bayes network debug info (non-production)
- `GET /v1/debug/logs` - Knowledge tracker logs (non-production)

<!--feature-matrix-start-->
| Sub-system | Status |
|------------|--------|
| Twin Runtime | ✅ v0.2.0 |
| King / Sage / Magi | ✅ |
| Self‑Evolving System | 🔴 |
| HippoRAG | 🔴 |
| Mesh Credits | ✅ |
| ADAS Optimisation | ✅ |
| ConfidenceEstimator | ✅ |
<!--feature-matrix-end-->
