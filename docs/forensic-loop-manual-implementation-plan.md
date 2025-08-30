# Forensic Loop Manual Implementation Plan
## Real Execution Using MCP Swarm Tools

**Issue Identified:** The playbook system is a mock implementation that only generates JSON records without actually executing anything. The forensic loop needs to be manually implemented using real MCP swarm tools and Task agents.

---

## 🔴 WHAT'S ACTUALLY HAPPENING VS WHAT SHOULD HAPPEN

### Current Reality (Mock System):
```python
# What the code actually does:
async def _spawn_agents_parallel(self, context, stage):
    spawned = []
    for agent_config in stage.agents:
        spawned.append({  # Just creates a dictionary!
            'type': agent_config.get('type'),
            'focus': agent_config.get('focus'),
            'spawn_mode': 'parallel'
        })
    return spawned  # Returns description, doesn't spawn anything!
```

### What SHOULD Happen (Real Implementation):
```python
# What it should do:
# 1. Actually initialize MCP swarm
mcp__ruv-swarm__swarm_init { topology: "mesh", maxAgents: 8 }

# 2. Actually spawn agents
mcp__ruv-swarm__agent_spawn { type: "security-manager" }
mcp__ruv-swarm__agent_spawn { type: "code-analyzer" }

# 3. Actually execute tasks with real agents
Task("Security agent: Scan for vulnerabilities...")
Task("Code analyzer: Analyze architecture...")
```

---

## 📋 14-STAGE FORENSIC LOOP - MANUAL IMPLEMENTATION

### Stage 1: Document-First Harvest
**Purpose:** Extract architectural intent from documentation

**REAL IMPLEMENTATION:**
```javascript
// Initialize swarm for documentation analysis
mcp__ruv-swarm__swarm_init { 
    topology: "mesh", 
    maxAgents: 4,
    optimization: "documentation_extraction"
}

// Spawn specialized agents
mcp__ruv-swarm__agent_spawn { type: "researcher", focus: "documentation" }
mcp__ruv-swarm__agent_spawn { type: "repo-architect", focus: "architecture_discovery" }

// Execute actual tasks
Task("Documentation researcher: Extract all architectural documentation", {
    subagent_type: "researcher",
    prompt: `
    1. Use Glob to find all *.md, *.txt, README files
    2. Read each documentation file
    3. Extract architectural decisions, design patterns, and constraints
    4. Create summary of architectural intent
    5. Output to docs/forensic/ARCHITECTURE_INTENT.md
    `
})

Task("Architecture analyzer: Map code structure", {
    subagent_type: "repo-architect", 
    prompt: `
    1. Use Glob to identify all source directories
    2. Map directory structure and module organization
    3. Identify main components and their relationships
    4. Document module boundaries and interfaces
    5. Output to docs/forensic/CODE_STRUCTURE.md
    `
})
```

### Stage 2: Flow-Aware Code Mapping
**Purpose:** Map code flows with upstream/downstream awareness

**REAL IMPLEMENTATION:**
```javascript
Task("Code flow mapper: Analyze dependencies and data flow", {
    subagent_type: "code-analyzer",
    prompt: `
    1. Use Grep to find all import statements
    2. Build dependency graph of modules
    3. Identify circular dependencies if any
    4. Map data flow patterns
    5. Find integration points between modules
    6. Output dependency graph to docs/forensic/DEPENDENCY_MAP.json
    7. Output flow analysis to docs/forensic/CODE_FLOWS.md
    `
})
```

### Stage 3: Test Infrastructure Discovery
**Purpose:** Comprehensive test discovery and categorization

**REAL IMPLEMENTATION:**
```javascript
Task("Test analyzer: Discover and categorize all tests", {
    subagent_type: "tester",
    prompt: `
    1. Use Glob to find all test files (*test*.py, *spec*.js, etc)
    2. Read each test file and categorize:
       - Unit tests
       - Integration tests
       - Behavioral tests
       - Mock-heavy tests
    3. Calculate test coverage metrics
    4. Identify flaky tests (look for skip markers, random, sleep)
    5. Output test census to docs/forensic/TEST_CENSUS.json
    6. Output test quality report to docs/forensic/TEST_QUALITY.md
    `
})
```

### Stage 4: Comprehensive Issue Discovery
**Purpose:** Multi-dimensional issue discovery and clustering

**REAL IMPLEMENTATION:**
```javascript
// Parallel issue discovery across multiple dimensions
mcp__ruv-swarm__swarm_init { 
    topology: "adaptive", 
    maxAgents: 8
}

// Security scanning
Task("Security scanner: Find vulnerabilities", {
    subagent_type: "security-manager",
    prompt: `
    1. Use Grep to find hardcoded secrets (API keys, passwords)
    2. Find SQL injection vulnerabilities
    3. Find XSS vulnerabilities
    4. Check for insecure dependencies
    5. Output to docs/forensic/SECURITY_ISSUES.json
    `
})

// Performance analysis
Task("Performance analyzer: Find bottlenecks", {
    subagent_type: "perf-analyzer",
    prompt: `
    1. Find synchronous operations that could be async
    2. Find N+1 query patterns
    3. Find inefficient algorithms (nested loops)
    4. Find memory leaks (unclosed resources)
    5. Output to docs/forensic/PERFORMANCE_ISSUES.json
    `
})

// Code quality analysis
Task("Quality analyzer: Find code smells", {
    subagent_type: "code-analyzer",
    prompt: `
    1. Find duplicate code blocks (copy-paste)
    2. Find God objects (>500 lines)
    3. Find long parameter lists (>3 params)
    4. Find circular dependencies
    5. Find magic numbers/strings
    6. Output to docs/forensic/CODE_QUALITY_ISSUES.json
    `
})

// Test issues
Task("Test issue finder: Find test problems", {
    subagent_type: "tester",
    prompt: `
    1. Find tests with no assertions
    2. Find tests that test implementation not behavior
    3. Find flaky tests (random, time-dependent)
    4. Find missing test coverage
    5. Output to docs/forensic/TEST_ISSUES.json
    `
})
```

### Stage 5: Intelligent Playbook Routing
**Purpose:** Route issues to appropriate specialist fixes

**REAL IMPLEMENTATION:**
```javascript
Task("Issue router: Categorize and route issues", {
    subagent_type: "planner",
    prompt: `
    1. Read all issue files from docs/forensic/*_ISSUES.json
    2. Cluster similar issues together
    3. Prioritize by severity:
       - Critical: Security vulnerabilities, data loss risks
       - High: Performance bottlenecks, flaky tests
       - Medium: Code duplication, architecture debt
       - Low: Documentation, style issues
    4. Create routing plan:
       - Security issues → security fixes
       - Performance issues → optimization tasks
       - Test issues → test stabilization
       - Code quality → refactoring tasks
    5. Output routing decisions to docs/forensic/ROUTING_PLAN.json
    6. Create execution batches (max 5 parallel) to docs/forensic/EXECUTION_BATCHES.json
    `
})
```

### Stage 6: Parallel Specialist Execution
**Purpose:** Fix issues in optimized batches

**REAL IMPLEMENTATION:**
```javascript
// Read execution batches
const batches = Read("docs/forensic/EXECUTION_BATCHES.json")

// Process each batch
for (const batch of batches) {
    // Spawn agents for this batch (max 5 concurrent)
    mcp__ruv-swarm__task_orchestrate {
        task: `Fix batch: ${batch.name}`,
        agents: batch.required_agents,
        phases: ["analyze", "implement", "test", "validate"]
    }
    
    // Execute fixes in parallel
    for (const issue of batch.issues) {
        if (issue.type === "security") {
            Task("Security fix implementation", {
                subagent_type: "coder",
                prompt: `Fix security issue: ${issue.description}
                        File: ${issue.file}
                        Line: ${issue.line}
                        Use Edit to apply fix
                        Write test to verify fix`
            })
        } else if (issue.type === "performance") {
            Task("Performance optimization", {
                subagent_type: "perf-analyzer",
                prompt: `Optimize performance issue: ${issue.description}
                        Current metric: ${issue.current_metric}
                        Target: ${issue.target_metric}
                        Apply optimization and measure improvement`
            })
        }
        // ... other issue types
    }
}
```

### Stage 7: Test Consolidation & Migration
**Purpose:** Consolidate tests and migrate to behavioral patterns

**REAL IMPLEMENTATION:**
```javascript
Task("Test consolidator: Improve test quality", {
    subagent_type: "tester",
    prompt: `
    1. Read TEST_CENSUS.json to get all test files
    2. For each test file:
       a. Find duplicate test logic
       b. Extract to shared fixtures/helpers
       c. Convert implementation tests to behavioral tests
       d. Remove unnecessary mocks
       e. Use MultiEdit to apply all changes
    3. Create consolidated test report
    4. Ensure all tests still pass after consolidation
    `
})
```

### Stage 8: Guard Test Generation
**Purpose:** Generate comprehensive guard tests for all changes

**REAL IMPLEMENTATION:**
```javascript
Task("Guard test generator: Create regression prevention", {
    subagent_type: "tdd-london-swarm",
    prompt: `
    1. Read all changes made in previous stages
    2. For each changed module:
       a. Create behavioral contract tests
       b. Add edge case coverage
       c. Add integration boundary tests
       d. Ensure changes don't break existing behavior
    3. Write new test files in tests/guards/
    4. Run all guard tests to verify
    `
})
```

### Stage 9: Integration & Build Validation
**Purpose:** Integrate all changes and validate build

**REAL IMPLEMENTATION:**
```javascript
Task("Integration validator: Ensure system integrity", {
    subagent_type: "production-validator",
    prompt: `
    1. Run full build process:
       - Bash "npm run build" or "python setup.py build"
    2. Run linting:
       - Bash "npm run lint" or "ruff check ."
    3. Run type checking:
       - Bash "npm run typecheck" or "mypy ."
    4. Run security scan:
       - Bash "npm audit" or "bandit -r ."
    5. Check for merge conflicts
    6. Validate all dependencies resolved
    7. Output validation report to docs/forensic/BUILD_VALIDATION.json
    `
})
```

### Stage 10: Full Test Suite Execution
**Purpose:** Execute complete test suite

**REAL IMPLEMENTATION:**
```javascript
Task("Test runner: Execute all tests", {
    subagent_type: "tester",
    prompt: `
    1. Run all test suites:
       - Bash "npm test" or "pytest"
    2. Generate coverage report:
       - Bash "npm run coverage" or "pytest --cov"
    3. Identify any failing tests
    4. For failures, determine if:
       - Test needs update
       - Code has regression
       - Environment issue
    5. Output results to docs/forensic/TEST_RESULTS.xml
    6. Output coverage to docs/forensic/COVERAGE.json
    `
})
```

### Stage 11: Documentation Synchronization
**Purpose:** Update all documentation

**REAL IMPLEMENTATION:**
```javascript
Task("Documentation updater: Sync docs with code", {
    subagent_type: "api-docs",
    prompt: `
    1. Regenerate API documentation from code
    2. Update README with any new features/changes
    3. Update CHANGELOG with all modifications
    4. Check for broken links in documentation
    5. Update architecture diagrams if structure changed
    6. Ensure all code examples in docs still work
    7. Output sync report to docs/forensic/DOC_SYNC_REPORT.json
    `
})
```

### Stage 12: Final Validation Suite
**Purpose:** Comprehensive final validation

**REAL IMPLEMENTATION:**
```javascript
// Parallel final validation
mcp__ruv-swarm__task_orchestrate {
    task: "Final system validation",
    agents: ["perf-analyzer", "security-manager", "production-validator"],
    phases: ["validate", "benchmark", "certify"]
}

Task("Performance validation", {
    subagent_type: "performance-benchmarker",
    prompt: `Run performance benchmarks and compare to baseline`
})

Task("Security validation", {
    subagent_type: "security-manager",
    prompt: `Run final security scan and verify zero criticals`
})

Task("Production readiness", {
    subagent_type: "production-validator",
    prompt: `Validate system is production ready`
})
```

### Stage 13: Forensic Report Generation
**Purpose:** Create comprehensive audit report

**REAL IMPLEMENTATION:**
```javascript
Task("Report generator: Create forensic audit report", {
    subagent_type: "adaptive-coordinator",
    prompt: `
    1. Read all forensic output files from docs/forensic/
    2. Synthesize findings:
       - Issues discovered
       - Fixes applied
       - Tests improved
       - Documentation updated
       - Performance impact
       - Security posture
    3. Calculate metrics:
       - CI pass rate
       - Test coverage
       - Security score
       - Performance improvement
    4. Generate executive summary
    5. Output final report to docs/FORENSIC_REPORT.md
    `
})
```

### Stage 14: Re-Audit Decision Engine
**Purpose:** Determine if another cycle is needed

**REAL IMPLEMENTATION:**
```javascript
Task("Decision engine: Evaluate termination criteria", {
    subagent_type: "adaptive-coordinator",
    prompt: `
    1. Read FORENSIC_REPORT.md
    2. Check termination criteria:
       - CI pass rate == 100%
       - Critical issues == 0
       - Security vulnerabilities == 0
       - Documentation sync >= 95%
       - Test flakiness <= 1%
    3. If all criteria met:
       - Mark audit as COMPLETE
    4. If not met and cycles < 12:
       - Identify remaining issues
       - Plan next cycle focus
       - Output to docs/forensic/NEXT_CYCLE_PLAN.json
    5. Output decision to docs/forensic/AUDIT_DECISION.json
    `
})
```

---

## 🚀 EXECUTION STRATEGY

### Initialize Once:
```javascript
// Start the forensic audit system
mcp__ruv-swarm__swarm_init { 
    topology: "adaptive",
    maxAgents: 12,
    optimization: "forensic_audit",
    memory_sharing: true
}
```

### Execute Stages:
1. **Sequential stages**: 1, 2, 3, 5, 7, 8, 9, 11, 13, 14
2. **Parallel stages**: 4, 6, 10, 12
3. **Use TodoWrite** to track progress through stages
4. **Use memory** to persist findings between stages

### Monitor Progress:
```javascript
// Check swarm status regularly
mcp__ruv-swarm__swarm_status { verbose: true }
mcp__ruv-swarm__task_status { detailed: true }
mcp__ruv-swarm__agent_metrics { metric: "all" }
```

---

## ⚠️ KEY DIFFERENCES FROM MOCK SYSTEM

1. **Real Tool Usage**: Actually use Read, Write, Edit, Glob, Grep, Bash
2. **Real Agent Spawning**: Use mcp__ruv-swarm__agent_spawn
3. **Real Task Execution**: Use Task tool with actual prompts
4. **Real File Generation**: Create actual output files, not just JSON records
5. **Real Testing**: Actually run tests, not just record "success"
6. **Real Validation**: Actually check criteria, not just return "passed"

---

## 📊 SUCCESS METRICS

The forensic audit is successful when:
- **All real tests pass** (not just JSON saying they passed)
- **Real files are created** in docs/forensic/
- **Actual issues are found and fixed** (using Edit tool)
- **Real metrics improve** (measured with actual commands)
- **Documentation actually updated** (files modified)

This is a REAL implementation plan that actually executes tasks, not just creates JSON records claiming they were executed.