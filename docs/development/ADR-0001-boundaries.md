# ADR-0001: Architecture Boundaries and Layer Separation

## Status
Accepted

## Context

AIVillage has grown into a complex distributed AI platform with multiple layers including production code, experimental features, mobile clients, and infrastructure components. As the system scales, maintaining clear architectural boundaries becomes critical for:

1. **Code Safety**: Preventing experimental code from affecting production systems
2. **Maintainability**: Clear separation of concerns and dependencies
3. **Testing**: Isolated testing of different architectural layers
4. **Security**: Controlled access to sensitive production components
5. **Development Velocity**: Parallel development without cross-contamination

The current structure includes:
- `packages/core/` - Core production systems
- `packages/core/experimental/` - Experimental features and research
- `clients/` - Mobile and desktop client applications
- `deploy/` - Infrastructure and deployment configurations

## Decision

We establish the following architectural boundaries and layer separation rules:

### 1. Production-Experimental Boundary

**Rule**: Production code MUST NOT import experimental modules.

**Rationale**:
- Experimental code may contain unstable APIs, incomplete features, or security vulnerabilities
- Production systems require stability and predictable behavior
- Feature flags provide controlled mechanism for experimental feature rollout

**Implementation**:
- CI/CD pipeline enforces this boundary with automated checks
- Feature flag system (`packages/core/common/flags.py`) provides controlled exposure
- Clear import patterns: `from packages.core.common import flags` (allowed) vs `from packages.core.experimental import *` (forbidden in production)

**Allowed Cross-Boundary Communication**:
- Feature flags can conditionally enable experimental paths
- Well-defined interfaces through dependency injection
- Configuration-driven component swapping

### 2. Gateway ↔ Twin ↔ MCP Service Boundaries

**Rule**: Services communicate only through defined API contracts, not direct imports.

**Service Architecture**:
```
┌─────────────┐    HTTP/gRPC    ┌─────────────┐    MCP Protocol    ┌─────────────┐
│   Gateway   │ ────────────────│    Twin     │ ─────────────────── │     MCP     │
│  (Port 8000)│                 │ (Port 8001) │                   │   Servers   │
└─────────────┘                 └─────────────┘                   └─────────────┘
```

**Interface Hardening**:
- Gateway: Rate limiting, CORS, security headers, request routing
- Twin: Core AI functionality, chat, RAG queries, agent coordination
- MCP: Model Control Protocol servers for specialized tools and memory

**Boundary Enforcement**:
- No direct Python imports between services
- All communication through HTTP/REST APIs or MCP protocol
- Shared data structures defined in `packages/core/common/schemas.py`
- Service discovery through configuration, not code coupling

### 3. Mobile Client Boundary

**Rule**: Mobile clients are independent applications that communicate only through public APIs.

**Implementation**:
- iOS/Android apps use generated SDK from OpenAPI specification
- No direct access to Python internals or database
- All functionality exposed through REST endpoints
- Client-side state management independent of server internals

### 4. Infrastructure Boundary

**Rule**: Application code MUST NOT directly import infrastructure components.

**Separation**:
- Application logic in `packages/core/` and `packages/agents/`
- Infrastructure in `deploy/`, Docker configurations, Kubernetes manifests
- Configuration management through environment variables and config files
- Infrastructure-as-Code principles with clear deployment boundaries

### 5. Feature Flag Integration Points

Feature flags provide controlled boundary crossing:

```python
from packages.core.common import is_enabled

# Production code with experimental path
def process_query(query: str, user_id: str = None):
    if is_enabled("advanced_rag_features", user_id):
        # Experimental RAG with Bayesian trust
        from packages.core.experimental.rag import AdvancedRAGProcessor
        return AdvancedRAGProcessor().process(query)
    else:
        # Production-stable RAG
        from packages.rag.core import HyperRAGOrchestrator
        return HyperRAGOrchestrator().process(query)
```

## Consequences

### Positive
- **Enhanced Stability**: Production systems isolated from experimental changes
- **Improved Security**: Clear boundaries prevent accidental exposure of sensitive components
- **Better Testing**: Isolated layers can be tested independently with proper mocking
- **Parallel Development**: Teams can work on different layers without interference
- **Controlled Rollouts**: Feature flags enable gradual exposure of experimental features
- **Clear Documentation**: Architectural boundaries make system easier to understand and onboard

### Negative
- **Additional Complexity**: Developers must understand and respect boundary rules
- **Interface Overhead**: Cross-boundary communication requires more boilerplate
- **Potential Duplication**: Some utilities may need to exist in multiple layers
- **Integration Challenges**: End-to-end features require coordination across boundaries

### Mitigations
- **Comprehensive Documentation**: Clear guidelines for each boundary type
- **Automated Enforcement**: CI/CD checks prevent boundary violations
- **Shared Utilities**: Common functionality in `packages/core/common/`
- **Integration Tests**: End-to-end tests validate cross-boundary interactions
- **Developer Training**: Team education on architectural principles

## Implementation Plan

### Phase 1: Enforcement (Complete)
- [x] CI/CD checks for production-experimental boundary
- [x] Feature flag system implementation
- [x] Service interface documentation

### Phase 2: Hardening (In Progress)
- [ ] Service API contract validation
- [ ] Generated client SDK from OpenAPI
- [ ] Cross-service integration tests
- [ ] Monitoring for boundary violations

### Phase 3: Optimization
- [ ] Performance analysis of cross-boundary calls
- [ ] Caching strategies for service communication
- [ ] Error handling standardization
- [ ] Load balancing and failover mechanisms

## Monitoring and Compliance

### Automated Checks
- **Import Analysis**: Daily scans for boundary violations in CI/CD
- **API Contract Validation**: OpenAPI schema compatibility checks
- **Feature Flag Auditing**: Tracking of experimental feature usage
- **Performance Monitoring**: Cross-boundary call latency and error rates

### Manual Reviews
- **Architecture Reviews**: Quarterly review of boundary effectiveness
- **Security Audits**: Annual assessment of boundary security implications
- **Performance Analysis**: Monthly analysis of cross-boundary communication costs

### Success Metrics
- **Zero Production Imports**: No experimental imports in production code
- **API Stability**: <1% breaking changes in service interfaces per quarter
- **Feature Flag Coverage**: >80% of experimental features behind flags
- **Test Isolation**: >90% of tests can run in isolated layer mode

## Related Decisions
- ADR-0002: API Versioning Strategy (planned)
- ADR-0003: Feature Flag Management (planned)
- ADR-0004: Service Discovery Architecture (planned)

## References
- [AIVillage Architecture Overview](../architecture/ARCHITECTURE.md)
- [Feature Flag Configuration](../../config/flags.yaml)
- [API Documentation](../api/API_VERSIONING.md)
- [CI/CD Pipeline](../../.github/workflows/main-ci.yml)

---

**Author**: AIVillage Development Team
**Date**: August 19, 2025
**Review Date**: November 19, 2025
**Status**: Accepted and Implemented
