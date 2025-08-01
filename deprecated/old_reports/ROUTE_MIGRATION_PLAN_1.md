# Route Migration Plan: server.py to Microservices

## Current Route Analysis

### Routes in server.py
1. **POST /query** - RAG query processing endpoint
   - Purpose: Process RAG queries through the pipeline
   - Migration target: **Twin service** (aligns with /v1/chat functionality)

2. **POST /upload** - File upload for vector store
   - Purpose: Upload documents to vector store
   - Migration target: **Twin service** (data ingestion)

3. **GET /** - Serves UI static files
   - Purpose: Development UI
   - Migration target: **Keep in server.py** (dev only) or **Gateway** (if needed for prod)

4. **GET /status** - System status endpoint
   - Purpose: Health/status check
   - Migration target: **Merge with existing /healthz endpoints**

5. **GET /bayes** - Bayes network snapshot
   - Purpose: Debug/monitoring endpoint
   - Migration target: **Twin service** (internal state exposure)

6. **GET /logs** - Knowledge tracker retrieval logs
   - Purpose: Debug/monitoring endpoint
   - Migration target: **Twin service** (alongside /metrics)

7. **GET /v1/explanation** - Evidence pack retrieval
   - Purpose: Get evidence for a chat
   - Migration target: **Already exists as /v1/evidence in Twin**

8. **GET /explain** - Graph explanation path
   - Purpose: Explain connections between nodes
   - Migration target: **Already exists as POST /explain in Twin**

### Existing Microservice Routes

#### Gateway Service
- GET /healthz
- GET /metrics
- POST /v1/chat (proxies to Twin)

#### Twin Service
- POST /v1/chat
- GET /v1/embeddings
- GET /healthz
- GET /metrics
- POST /explain
- POST /v1/evidence
- DELETE /v1/user/{user_id}

## Migration Mapping

| server.py Route | Target Service | New Route | Action Required |
|----------------|----------------|-----------|-----------------|
| POST /query | Twin | POST /v1/query | Create new endpoint |
| POST /upload | Twin | POST /v1/upload | Create new endpoint |
| GET / | Keep in server.py | - | Dev only |
| GET /status | Gateway/Twin | Use /healthz | Redirect/deprecate |
| GET /bayes | Twin | GET /v1/debug/bayes | Create debug namespace |
| GET /logs | Twin | GET /v1/debug/logs | Create debug namespace |
| GET /v1/explanation | Twin | POST /v1/evidence | Update to use existing |
| GET /explain | Twin | POST /explain | Update to use existing |

## Implementation Strategy

### Phase 1: Non-Breaking Additions
1. Add new routes to microservices
2. Keep server.py routes functional with deprecation warnings

### Phase 2: Client Updates
1. Update any clients to use new routes
2. Add request forwarding from old to new routes

### Phase 3: Deprecation
1. Add startup warnings to server.py
2. Log all production usage of deprecated routes
3. Return deprecation headers in responses

### Phase 4: Removal
1. Remove routes from server.py
2. Keep only test harness functionality

## Testing Strategy

### Unit Tests
```python
# tests/test_route_migration.py
def test_query_endpoint_via_twin():
    """Verify /v1/query works through Twin service"""
    pass

def test_upload_endpoint_via_twin():
    """Verify /v1/upload works through Twin service"""
    pass

def test_deprecated_routes_warn():
    """Verify server.py routes return deprecation warnings"""
    pass
```

### Integration Tests
```python
# tests/test_service_integration.py
def test_gateway_proxies_to_twin():
    """Verify Gateway correctly proxies requests"""
    pass

def test_healthcheck_aggregation():
    """Verify health status is properly aggregated"""
    pass
```

## Docker Compose Updates

```yaml
services:
  gateway:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]

  twin:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/healthz"]

  # server.py removed from production compose
```

## Documentation Updates Required

1. **docs/architecture.md** - Update service boundaries
2. **docs/api/README.md** - Update endpoint documentation
3. **README.md** - Update quick start to use Gateway
4. **docs/migration_notes.md** - Add route migration notes

## Deprecation Timeline

- **Week 1**: Add new routes to services
- **Week 2**: Add deprecation warnings
- **Week 3**: Update all documentation
- **Week 4**: Monitor usage and fix any issues
- **Month 2**: Remove deprecated routes from server.py
