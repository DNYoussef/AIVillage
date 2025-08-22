# Refactoring Priority Matrix
## AIVillage Critical Coupling Violations

| File | LOC | Violations | Coupling Score | Dependencies | Risk Level | Effort (hrs) | Priority |
|------|-----|------------|----------------|-------------|------------|--------------|-----------|
| **TIER 1: CRITICAL - IMMEDIATE ACTION REQUIRED** |
| `base_agent_template.py` | 845 | HIGH | 37.1 | 23 agents | CRITICAL | 120 | P0 |
| `agent_orchestration_system.py` | 777 | HIGH | 25.0 | Core system | CRITICAL | 80 | P0 |
| `horticulturist_agent.py` | 913 | HIGH | 22.0 | Agriculture | CRITICAL | 60 | P0 |
| **TIER 2: HIGH - NEXT ITERATION** |
| `pii_phi_manager.py` | 1,772 | 303 | 28.0 | Compliance | HIGH | 80 | P1 |
| `curriculum_graph.py` | 1,615 | 576 | 26.0 | Education | HIGH | 60 | P1 |
| `mcp_governance_dashboard.py` | 659 | HIGH | 24.0 | Governance | HIGH | 50 | P1 |
| **TIER 3: MEDIUM - PLANNED REFACTORING** |
| `shield_agent.py` | 1,270 | 300 | 23.0 | Security | MEDIUM | 40 | P2 |
| `personalized_tutor.py` | 1,271 | 328 | 22.0 | Education | MEDIUM | 40 | P2 |
| `parent_tracker.py` | 1,503 | 291 | 21.0 | Monitoring | MEDIUM | 35 | P2 |

---

## Impact vs Effort Matrix

```
High Impact │ P0: BaseAgent    │ P1: PII/PHI     │
           │     Template     │     Manager      │
           │                  │                  │
           │ P0: Orchestrator │ P2: Shield      │
           │                  │     Agent       │
           │                  │                  │
Low Impact │ P3: Utilities    │ P2: Curriculum  │
           │                  │     Graph       │
           └─────────────────────────────────────┘
             Low Effort         High Effort
```

---

## Dependency Impact Analysis

### BaseAgentTemplate (CRITICAL)
- **Affects:** All 23 specialized agents
- **Breaking Change Risk:** VERY HIGH
- **Refactoring Strategy:** Incremental with Adapter Pattern
- **Validation Required:** Full integration test suite

### AgentOrchestrationSystem (CRITICAL)
- **Affects:** Core agent coordination
- **Breaking Change Risk:** HIGH
- **Refactoring Strategy:** Service extraction with facades
- **Validation Required:** Load testing and performance validation

### HorticulturistAgent (CRITICAL)
- **Affects:** Agriculture domain
- **Breaking Change Risk:** MEDIUM
- **Refactoring Strategy:** Domain-driven decomposition
- **Validation Required:** Domain logic validation

---

## Refactoring Sequence Plan

### Phase 1: Foundation (Weeks 1-4)
1. **BaseAgentTemplate** - Most critical, affects everything
2. **AgentOrchestrationSystem** - Core infrastructure dependency
3. **HorticulturistAgent** - Domain isolation opportunity

### Phase 2: Compliance & Security (Weeks 5-8)
1. **PII/PHI Manager** - Regulatory requirements
2. **Shield Agent** - Security implications
3. **Governance Dashboard** - Audit trail importance

### Phase 3: Domain Logic (Weeks 9-12)
1. **Curriculum Graph** - Education domain
2. **Personalized Tutor** - Related education component
3. **Parent Tracker** - Monitoring domain

---

## Risk Mitigation Strategies

### High-Risk Components
- **Feature freeze** during refactoring
- **Extensive behavioral testing** before changes
- **Canary deployment** strategy
- **Rollback plan** for each component

### Medium-Risk Components
- **Parallel development** with legacy support
- **Gradual migration** approach
- **A/B testing** for validation

### Low-Risk Components
- **Automated refactoring** tools where possible
- **Continuous integration** validation
- **Code review** focused approach
