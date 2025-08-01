# ADR-0010 Restrict server.py to Development/Test Harness Only

**Status**: Proposed
**Date**: 2025-01-16

## Context

The AIVillage monorepo contains both a monolithic `server.py` file and separate microservices (gateway and twin) that expose production APIs. Currently, `server.py` exposes routes that overlap with the microservices architecture, creating confusion about which endpoints should be used in production.

Current state analysis reveals:

### Routes in server.py
- `POST /query` - RAG query processing endpoint
- `POST /upload` - File upload for vector store
- `GET /` - Serves UI static files
- `GET /status` - System status endpoint
- `GET /bayes` - Bayes network snapshot
- `GET /logs` - Knowledge tracker retrieval logs
- `GET /v1/explanation` - Evidence pack retrieval (placeholder implementation)
- `GET /explain` - Graph explanation path

### Routes in microservices
- **Gateway service** (`services/gateway/`):
  - `GET /healthz`
  - `GET /metrics`
  - `POST /v1/chat`
- **Twin service** (`services/twin/`):
  - `POST /v1/chat`
  - `GET /v1/embeddings`
  - `GET /healthz`
  - `GET /metrics`
  - `POST /explain`
  - `POST /v1/evidence`
  - `DELETE /v1/user/{user_id}`

The overlapping functionality (e.g., `/explain` vs `/explain`, `/v1/explanation` vs `/v1/evidence`) creates ambiguity about the canonical API surface and risks divergent implementations.

## Decision

We will formally restrict `server.py` to serve only as a development and test harness, not for production use. This includes:

1. **Documentation**: Clearly mark `server.py` as "DEVELOPMENT ONLY" in code comments and documentation
2. **CI Protection**: Implement automated checks to prevent new production routes from being added to `server.py`
3. **Route Migration**: Plan migration of any production-critical functionality from `server.py` to appropriate microservices
4. **Environment Enforcement**: Add runtime warnings when `server.py` is run without explicit development flags

### CI Enforcement Strategy

Implement a `scripts/check_server_routes.py` script that:
- Parses `server.py` using AST to extract all route definitions
- Maintains an allowlist of existing development/test routes
- Fails CI if new routes are added without explicit approval
- Generates warnings for any routes that should be migrated

## Consequences

### Positive
- Clear separation between development tooling and production services
- Prevents accidental use of non-production-ready endpoints
- Encourages proper microservice architecture adoption
- Reduces security surface area in production

### Negative
- Requires migration effort for any production workloads using `server.py`
- Additional CI complexity with route checking
- May slow down rapid prototyping if developers habitually use `server.py`

### Neutral
- Development workflows remain unchanged for local testing
- Existing test suites can continue using `server.py` as a test harness

## Alternatives Considered

1. **Complete removal of server.py**: Rejected due to its value as a testing tool
2. **Dual-mode server.py**: Rejected as it would perpetuate confusion
3. **No enforcement**: Rejected as it allows continued architectural drift

## Migration Tasks

**TODO**: The following tasks should be completed to implement this decision:

- [ ] Add "DEVELOPMENT ONLY" header comments to `server.py`
- [ ] Create `scripts/check_server_routes.py` CI enforcement script
- [ ] Update CI/CD pipelines to run route checker
- [ ] Document development vs production endpoints in README
- [ ] Add startup warning to `server.py` when not in explicit dev mode
- [ ] Create ADR-0011 for specific route migration plan
- [ ] Update deployment documentation to exclude `server.py`
- [ ] Add environment variable `AIVILLAGE_DEV_MODE` requirement for `server.py`

## References

- ADR-0002: Messaging Protocol Choice (establishes microservice communication patterns)
- `services/gateway/app.py`: Production gateway implementation
- `services/twin/app.py`: Production twin service implementation
