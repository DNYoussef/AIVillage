# Definition of Done (DoD) - AIVillage

## Overview

This document defines the criteria that must be met for any work item (feature, bug fix, refactor, etc.) to be considered "Done" in the AIVillage project. All team members and contributors must ensure these criteria are met before marking work as complete or merging code.

## Universal DoD Criteria

Every work item must satisfy ALL of the following criteria before being considered Done:

### ✅ Code Quality & Standards

- [ ] **Code follows project standards**
  - Passes all linting checks (Ruff, Black, isort)
  - Follows [CLAUDE.md](../CLAUDE.md) coding guidelines
  - Uses consistent naming conventions and project patterns
  - Includes proper type hints (Python) or type annotations

- [ ] **Code is reviewed and approved**
  - At least 2 reviewers have approved (as per CODEOWNERS)
  - All review comments are addressed
  - No merge conflicts exist
  - Branch is up-to-date with target branch

- [ ] **Architecture compliance**
  - No experimental/ imports in production code (enforced by CI)
  - Follows established architecture boundaries (see ADR-0001)
  - Integration points properly documented
  - Breaking changes have migration guide

### 🧪 Testing Requirements

- [ ] **Unit tests written and passing**
  - Code coverage ≥ 60% (will increase to 70% in next sprint)
  - Critical paths have 100% coverage
  - Edge cases and error conditions tested
  - Tests are maintainable and well-documented

- [ ] **Integration tests pass**
  - All existing integration tests continue to pass
  - New integration tests added for cross-component interactions
  - P2P, RAG, Agent Forge integrations tested where applicable
  - Database migrations tested (if applicable)

- [ ] **End-to-end tests pass** (for user-facing features)
  - Critical user workflows validated
  - API endpoints tested with realistic scenarios
  - Error handling and edge cases covered
  - Performance within acceptable limits

- [ ] **Security tests pass**
  - Bandit security scanning passes
  - No hardcoded secrets or credentials
  - Input validation and sanitization tested
  - Authentication/authorization tested (if applicable)

### 📚 Documentation

- [ ] **Code is self-documenting**
  - Clear function/class names and docstrings
  - Complex logic explained with comments
  - Public APIs have comprehensive docstrings
  - Type hints provide interface clarity

- [ ] **User documentation updated** (if user-facing)
  - README.md updated if installation/setup changes
  - API documentation reflects changes (OpenAPI spec)
  - User guides updated for new features
  - Migration guides for breaking changes

- [ ] **Technical documentation**
  - Architecture decisions documented (ADRs if needed)
  - Integration points documented
  - Configuration changes documented
  - Troubleshooting information provided

### 🔒 Security Requirements

- [ ] **Security review completed**
  - Code reviewed for security vulnerabilities
  - Dependency scanning passes (Safety)
  - No SQL injection, XSS, or similar vulnerabilities
  - Sensitive data handling reviewed

- [ ] **Authentication & authorization**
  - Proper access controls implemented
  - API endpoints secured appropriately
  - Role-based permissions respected
  - Session management secure

- [ ] **Data protection**
  - PII/PHI compliance validated (if handling sensitive data)
  - Encryption used for sensitive data at rest and in transit
  - Audit trails maintained for sensitive operations
  - Data retention policies followed

### 📊 Observability & Monitoring

- [ ] **Logging implemented**
  - Appropriate log levels used (DEBUG, INFO, WARN, ERROR)
  - Structured logging with correlation IDs
  - No sensitive information in logs
  - Log messages are actionable and helpful

- [ ] **Metrics and monitoring**
  - Key performance metrics exposed
  - Health checks implemented for new services
  - Alerts configured for critical failure modes
  - Dashboards updated (if needed)

- [ ] **Error handling and resilience**
  - Graceful degradation implemented
  - Circuit breakers used for external dependencies
  - Retry logic with exponential backoff
  - Timeout handling for all network calls

### 🚀 Performance Requirements

- [ ] **Performance benchmarks**
  - No performance regression from baseline
  - Load testing completed for high-traffic features
  - Resource usage within acceptable limits
  - Response times meet SLA requirements

- [ ] **Scalability considerations**
  - Code handles expected load
  - Database queries optimized
  - Caching strategy implemented where needed
  - Horizontal scaling supported

- [ ] **Mobile/edge optimization** (if applicable)
  - Battery usage optimized
  - Bandwidth usage minimized
  - Works on resource-constrained devices
  - Offline capabilities maintained

### 🚀 Deployment & Operations

- [ ] **CI/CD pipeline passes**
  - All pre-commit hooks pass locally
  - CI pipeline (7 stages) passes completely
  - Docker builds successful (if applicable)
  - Helm charts valid (if infrastructure changes)

- [ ] **Environment compatibility**
  - Works in development, staging, and production
  - Configuration externalized appropriately
  - Secrets management properly implemented
  - Environment-specific testing completed

- [ ] **Rollback plan**
  - Safe rollback procedure documented
  - Database migrations are reversible
  - Feature flags implemented for risky changes
  - Monitoring plan for post-deployment

## Component-Specific DoD

### 🤖 AI/ML Components (Agent Forge, Agents, RAG)

- [ ] **Model validation**
  - Model performance meets acceptance criteria
  - A/B testing completed (if applicable)
  - Model interpretability/explainability provided
  - Bias and fairness evaluation completed

- [ ] **Resource management**
  - GPU/CPU usage optimized
  - Memory usage within limits
  - Model size appropriate for deployment target
  - Inference latency meets requirements

- [ ] **Data quality**
  - Training data quality validated
  - Data pipeline tested end-to-end
  - Data versioning and lineage tracked
  - Privacy-preserving techniques applied

### 🌐 P2P/Networking Components

- [ ] **Network resilience**
  - Handles network partitions gracefully
  - Message delivery guarantees implemented
  - Peer discovery and connection management tested
  - Protocol compatibility maintained

- [ ] **Mobile optimization**
  - Battery usage minimized
  - Works on cellular networks
  - BitChat offline functionality tested
  - Data usage optimized

### 📱 Mobile/Edge Components

- [ ] **Cross-platform compatibility**
  - iOS and Android testing completed
  - Platform-specific optimizations implemented
  - Native integration tested
  - App store compliance verified

- [ ] **Resource constraints**
  - Memory usage optimized for mobile
  - Battery life impact measured
  - Thermal management tested
  - Works on older device hardware

### 🔌 API Components

- [ ] **API standards**
  - OpenAPI specification updated
  - RESTful principles followed
  - Versioning strategy implemented
  - Rate limiting configured

- [ ] **Backward compatibility**
  - Breaking changes properly versioned
  - Deprecation warnings implemented
  - Migration path provided
  - Client SDK updated

## DoD Checklist Template

Use this checklist for every pull request:

```markdown
## Definition of Done Checklist

### Code Quality & Standards
- [ ] Code follows project standards (linting passes)
- [ ] Code reviewed and approved by ≥2 reviewers
- [ ] Architecture compliance validated
- [ ] No merge conflicts

### Testing
- [ ] Unit tests written (≥60% coverage)
- [ ] Integration tests pass
- [ ] End-to-end tests pass (if applicable)
- [ ] Security tests pass

### Documentation
- [ ] Code is self-documenting
- [ ] User documentation updated (if needed)
- [ ] Technical documentation complete

### Security
- [ ] Security review completed
- [ ] Authentication/authorization implemented
- [ ] Data protection validated

### Observability
- [ ] Logging implemented
- [ ] Metrics and monitoring configured
- [ ] Error handling robust

### Performance
- [ ] Performance benchmarks pass
- [ ] Scalability considerations addressed
- [ ] Mobile/edge optimized (if applicable)

### Deployment
- [ ] CI/CD pipeline passes
- [ ] Environment compatibility verified
- [ ] Rollback plan documented

### Component-Specific (check if applicable)
- [ ] AI/ML model validation complete
- [ ] P2P network resilience tested
- [ ] Mobile cross-platform compatibility
- [ ] API standards compliance
```

## DoD Enforcement

### Pre-commit Validation

The following are automatically checked by pre-commit hooks:
- Code formatting (Black, Ruff)
- Import sorting (isort)
- Type checking (MyPy)
- Security scanning (Bandit)
- Secret detection

### CI/CD Validation

The 7-stage CI/CD pipeline enforces:
1. **Pre-flight Checks** - Syntax and critical security
2. **Code Quality** - Formatting, linting, type checking
3. **Testing** - Unit, integration, coverage tests
4. **Security Scanning** - Vulnerability and dependency checks
5. **Performance Testing** - Benchmark validation
6. **Build & Package** - Artifact creation
7. **Deployment** - Environment-specific validation

### Manual Review Gates

Code reviewers must verify:
- [ ] All automated checks pass
- [ ] DoD checklist completed
- [ ] Documentation is adequate
- [ ] Security considerations addressed
- [ ] Performance impact acceptable

## DoD Exceptions

In rare cases, DoD criteria may be waived with:

1. **Explicit approval** from 2 senior team members
2. **Technical debt ticket** created for missing items
3. **Timeline** for addressing exceptions (max 2 sprints)
4. **Risk assessment** documented

Exception reasons may include:
- Critical production hotfixes
- Time-sensitive security patches
- External dependency constraints
- Technical feasibility limitations

## DoD Evolution

This DoD will evolve with the project:

- **Monthly reviews** with team retrospectives
- **Criteria updates** based on lessons learned
- **Tool improvements** to automate more checks
- **Coverage increases** as project matures

## Success Metrics

DoD effectiveness is measured by:
- **Bug escape rate** to production
- **Security incident frequency**
- **Performance regression incidents**
- **Documentation completeness score**
- **Time to onboard new contributors**

## Getting Help

If you're unsure about DoD requirements:

1. **Ask in PR reviews** - Reviewers will help guide you
2. **Check examples** - Look at recently merged PRs
3. **Team chat** - Get quick clarification
4. **Office hours** - Weekly DoD Q&A sessions

---

**Remember**: The DoD is not about creating bureaucracy, but about ensuring quality, reliability, and maintainability of the AIVillage platform. It helps us ship with confidence and maintain our high standards as we scale.

---

*This Definition of Done is a living document. All team members are encouraged to suggest improvements through PRs or team discussions.*

**Last Updated**: August 19, 2025
**Version**: 1.0
**Next Review**: September 19, 2025
