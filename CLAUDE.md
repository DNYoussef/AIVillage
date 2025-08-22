# Claude Code + Claude Flow - Unified AI Development Platform

## 🚀 CLAUDE FLOW INTEGRATION - UNIFIED COMMAND CENTER

**CONSOLIDATED ARCHITECTURE**: All capabilities unified in `.claude/` directory
- **Hive Mind**: `.claude/hive-mind/` - Persistent memory and session management
- **Swarm Coordination**: `.claude/swarm/` - Distributed task orchestration
- **54 Specialized Agents**: `.claude/agents/` - Domain experts for every development need
- **SPARC Workflows**: Complete methodology automation via CLI and MCP
- **Performance Analytics**: `.claude/claude-flow-metrics/` - Real-time optimization and metrics

## 🚨 CRITICAL: CONCURRENT EXECUTION & FILE MANAGEMENT

**ABSOLUTE RULES**:
1. ALL operations MUST be concurrent/parallel in a single message
2. **NEVER save working files, text/mds and tests to the root folder**
3. ALWAYS organize files in appropriate subdirectories
4. **USE CLAUDE FLOW AGENTS PROACTIVELY** for complex tasks

### ⚡ GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**MANDATORY PATTERNS:**
- **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
- **Task tool**: ALWAYS spawn ALL agents in ONE message with full instructions
- **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
- **Bash commands**: ALWAYS batch ALL terminal operations in ONE message
- **Memory operations**: ALWAYS batch ALL memory store/retrieve in ONE message
- **MCP Claude Flow**: ALWAYS batch swarm coordination operations in ONE message

### 📁 File Organization Rules

**NEVER save to root folder. Use these directories:**
- `/src` - Source code files
- `/tests` - Test files
- `/docs` - Documentation and markdown files
- `/config` - Configuration files
- `/scripts` - Utility scripts
- `/examples` - Example code

## 🧠 CONNASCENCE-BASED CODING STANDARDS

### Core Principles
Connascence is our coupling taxonomy and metric. **Stronger forms are harder to spot and refactor; keep them local, reduce their degree, and push toward weaker forms.**

#### Strength Hierarchy (Weakest → Strongest)
1. **Static Forms** (visible in code without running)
   - Name (CoN) → Type (CoT) → Meaning (CoM) → Position (CoP) → Algorithm (CoA)
2. **Dynamic Forms** (only visible at runtime)
   - Execution (CoE) → Timing (CoTg) → Value (CoV) → Identity (CoI)

#### Management Strategy
- **Locality**: Strong connascence acceptable within one class/function; dangerous across modules/services
- **Degree**: How many things must change together? 2 < 200. Encapsulate to keep degree low
- **Strength**: Use stronger forms only when code is very close together

### Mandatory Refactoring Rules

#### Static Connascence
```python
# ❌ CoM - Magic values (strong, dangerous)
if user.role == 2:  # What does 2 mean?

# ✅ Use enums/constants (weak)
class Role(Enum): ADMIN = "admin"
if user.role is Role.ADMIN:

# ❌ CoP - Position dependent (fragile)
create_user("alice", True, "US")  # What is True?

# ✅ Named parameters (weak)
create_user(name="alice", email_verified=True, country="US")

# ❌ CoA - Duplicate algorithms (high degree)
def checksum_in_module_a(data): return hashlib.sha256(data).hexdigest()
def checksum_in_module_b(data): return hashlib.sha256(data).hexdigest()

# ✅ Single source of truth (degree=1)
def checksum(data): return hashlib.sha256(data).hexdigest()
```

#### Dynamic Connascence
```python
# ❌ CoE - Order dependent (brittle)
lock.acquire(); do_stuff(); lock.release()

# ✅ Context manager (implicit ordering)
with lock: do_stuff()

# ❌ CoTg - Sleep-based timing (fragile)
sleep(2); read_after_write()

# ✅ Event-driven with retries (robust)
publish(event); handle_with_retry(event)

# ❌ CoI - Global mutable state (high degree)
Config.timeout = 5  # Who else uses this?

# ✅ Dependency injection (explicit)
def handler(cfg: Config): use(cfg.timeout)
```

### Anti-Pattern Elimination

#### Prohibited Patterns
- **Big Ball of Mud**: No coherent architecture → modularize with bounded contexts
- **God Object**: One class doing everything → single responsibility principle
- **Magic Numbers/Strings**: Raw literals in logic → constants/enums only
- **Copy-Paste Programming**: Duplicate code blocks → extract functions/modules
- **Database-as-IPC**: DB coupling → APIs/events/queues
- **Sequential Coupling**: API requires strict call order → atomic operations

#### Required Practices
```python
# API Hygiene
def create_user(*, name: str, email: str, country: str):  # keyword-only after 3 params
    pass

# No ambiguous returns
def find_user(email: str) -> Union[User, None]:  # explicit, not -1 or None overload
    pass

# Algorithm centralization
class SecurityUtils:
    @staticmethod
    def hash_password(password: str) -> str:  # single source
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt())
```

### Architectural Rules

#### Module Structure
- **Max 500 lines per file**
- **No circular dependencies** (enforced in CI)
- **Max 3 positional parameters** per function
- **Strong connascence stays local** (same class/function only)

#### Boundary Definitions
```python
# Clear module boundaries with typed contracts
class UserService:
    def create_user(self, request: CreateUserRequest) -> CreateUserResponse:
        pass

# No global singletons
class DatabaseConfig:
    def __init__(self, connection_string: str, timeout: int):
        self.connection_string = connection_string
        self.timeout = timeout
```

## Project Overview

This project uses SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) methodology with Claude-Flow orchestration for systematic Test-Driven Development, enhanced with connascence-based coupling management.

## SPARC Commands

### Core Commands
- `npx claude-flow sparc modes` - List available modes
- `npx claude-flow sparc run <mode> "<task>"` - Execute specific mode
- `npx claude-flow sparc tdd "<feature>"` - Run complete TDD workflow
- `npx claude-flow sparc info <mode>` - Get mode details

### Batchtools Commands
- `npx claude-flow sparc batch <modes> "<task>"` - Parallel execution
- `npx claude-flow sparc pipeline "<task>"` - Full pipeline processing
- `npx claude-flow sparc concurrent <mode> "<tasks-file>"` - Multi-task processing

### Build Commands
- `npm run build` - Build project
- `npm run test` - Run tests
- `npm run lint` - Linting with connascence rules
- `npm run typecheck` - Type checking
- `ruff check . --fix` - Python linting with connascence rules

## SPARC Workflow Phases

1. **Specification** - Requirements analysis (`sparc run spec-pseudocode`)
2. **Pseudocode** - Algorithm design (`sparc run spec-pseudocode`)
3. **Architecture** - System design with connascence analysis (`sparc run architect`)
4. **Refinement** - TDD implementation with coupling refactoring (`sparc tdd`)
5. **Completion** - Integration with fitness functions (`sparc run integration`)

## Code Style & Best Practices

### Connascence Management
- **Modular Design**: Files under 500 lines, strong connascence local only
- **Coupling Metrics**: Track and minimize cross-module dependencies
- **Environment Safety**: Never hardcode secrets, use dependency injection
- **Test-First**: Write tests before implementation, assert behavior not internals
- **Clean Architecture**: Separate concerns with explicit boundaries
- **Documentation**: Keep updated with architectural decisions

### Refactoring Priorities
1. **Reduce Degree**: If >2 call sites co-vary, introduce facade/config
2. **Weaken Strength**: Position→Name, Meaning→Type, Algorithm→Single API
3. **Improve Locality**: Move strong connascence into same class/function
4. **Eliminate Anti-patterns**: Replace with documented alternatives

## 🎛️ CLAUDE FLOW COMMAND CENTER - 54 SPECIALIZED AGENTS

**UNIFIED LOCATION**: All agents consolidated in `.claude/agents/`

### 🔄 MCP TOOLS - SWARM COORDINATION (Primary Interface)

#### Core Coordination Commands
```bash
# Initialize intelligent swarm topology
mcp__claude-flow__swarm_init { topology: "mesh|hierarchical|adaptive", maxAgents: 6 }

# Spawn specialized agents concurrently
mcp__claude-flow__agent_spawn { type: "researcher|coder|tester", focus: "domain" }

# Orchestrate complex multi-step tasks
mcp__claude-flow__task_orchestrate { task: "description", agents: ["type1", "type2"] }
```

#### Real-Time Monitoring & Analytics
```bash
# Monitor swarm health and performance
mcp__claude-flow__swarm_status     # Health dashboard
mcp__claude-flow__agent_list       # Active agents list
mcp__claude-flow__agent_metrics    # Performance analytics
mcp__claude-flow__task_status      # Task progress tracking
mcp__claude-flow__task_results     # Execution results

# Performance benchmarking
mcp__claude-flow__benchmark_run    # Performance testing
```

#### Memory & Neural Pattern Management
```bash
# Persistent cross-session memory
mcp__claude-flow__memory_usage     # Memory analytics
mcp__claude-flow__neural_status    # Neural pattern status
mcp__claude-flow__neural_train     # Train new patterns
mcp__claude-flow__neural_patterns  # Analyze learned patterns
```

#### GitHub Workflow Integration
```bash
# Automated GitHub operations
mcp__claude-flow__github_swarm     # GitHub workflow swarm
mcp__claude-flow__repo_analyze     # Repository analysis
mcp__claude-flow__pr_enhance       # Enhanced PR management
mcp__claude-flow__issue_triage     # Intelligent issue triage
mcp__claude-flow__code_review      # AI-powered code review
```

#### System Features & Detection
```bash
# Advanced system capabilities
mcp__claude-flow__features_detect  # Feature detection
mcp__claude-flow__swarm_monitor    # Continuous monitoring
```

### 🛠️ SPARC WORKFLOW COMMANDS (CLI Interface)

#### Core SPARC Commands
```bash
# List available SPARC modes
npx claude-flow sparc modes

# Execute specific SPARC phase
npx claude-flow sparc run <mode> "<task>"
  # Modes: spec-pseudocode, architect, refinement, integration

# Complete TDD workflow (RECOMMENDED)
npx claude-flow sparc tdd "<feature description>"

# Get detailed mode information
npx claude-flow sparc info <mode>
```

#### Batch & Pipeline Processing (PERFORMANCE BOOST)
```bash
# Parallel execution across multiple phases (2.8-4.4x faster)
npx claude-flow sparc batch <modes> "<task>"
  # Example: batch spec-pseudocode,architect,refinement "user auth"

# Full pipeline processing
npx claude-flow sparc pipeline "<task>"

# Multi-task concurrent processing
npx claude-flow sparc concurrent <mode> "<tasks-file>"
```

### 🚀 54 SPECIALIZED AGENTS - BY CATEGORY

#### 🏗️ Core Development (Connascence-Enhanced)
- **`coder`** - Implementation with coupling analysis and clean patterns
- **`reviewer`** - Code review focusing on connascence violations and architectural fitness
- **`tester`** - TDD with behavioral assertions and mock elimination
- **`planner`** - Strategic planning with architectural fitness functions
- **`researcher`** - Deep analysis including coupling patterns and system architecture

#### 🕸️ Swarm Coordination (Distributed Intelligence)
- **`hierarchical-coordinator`** - Queen-led coordination with specialized worker delegation
- **`mesh-coordinator`** - Peer-to-peer mesh networks with distributed decision making
- **`adaptive-coordinator`** - Dynamic topology switching with self-organizing patterns
- **`collective-intelligence-coordinator`** - Group intelligence and consensus building
- **`swarm-memory-manager`** - Cross-agent memory sharing and persistence

#### ⚡ Consensus & Distributed Systems
- **`byzantine-coordinator`** - Byzantine fault-tolerant consensus with malicious actor detection
- **`raft-manager`** - Raft consensus algorithm with leader election and log replication
- **`gossip-coordinator`** - Gossip-based consensus for scalable eventually consistent systems
- **`consensus-builder`** - Multi-protocol consensus orchestration
- **`crdt-synchronizer`** - Conflict-free Replicated Data Types for state synchronization
- **`quorum-manager`** - Dynamic quorum adjustment and intelligent membership management
- **`security-manager`** - Comprehensive security mechanisms for distributed protocols

#### 📊 Performance & Optimization
- **`perf-analyzer`** - Performance bottleneck identification and workflow optimization
- **`performance-benchmarker`** - Comprehensive performance benchmarking and regression detection
- **`task-orchestrator`** - Central coordination for task decomposition and execution planning
- **`memory-coordinator`** - Cross-session memory management and sharing facilitation
- **`smart-agent`** - Intelligent agent coordination and dynamic spawning optimization

#### 🐙 GitHub & Repository Management
- **`github-modes`** - Comprehensive GitHub integration with workflow orchestration
- **`pr-manager`** - Complete pull request lifecycle management and coordination
- **`code-review-swarm`** - Intelligent multi-agent code review beyond static analysis
- **`issue-tracker`** - Intelligent issue management with automated tracking and coordination
- **`release-manager`** - Automated release coordination and deployment orchestration
- **`workflow-automation`** - Self-organizing CI/CD pipelines with adaptive coordination
- **`project-board-sync`** - Visual task management with AI swarm synchronization
- **`repo-architect`** - Repository structure optimization and multi-repo management
- **`multi-repo-swarm`** - Cross-repository automation and intelligent collaboration

#### 🎯 SPARC Methodology (Enhanced)
- **`sparc-coord`** - SPARC orchestration with connascence analysis and fitness functions
- **`sparc-coder`** - Transform specifications into working code with TDD practices
- **`specification`** - Requirements analysis with architectural constraints and fitness
- **`pseudocode`** - Algorithm design with coupling analysis and clean patterns
- **`architecture`** - System design with connascence management and clean boundaries
- **`refinement`** - TDD refinement with refactoring toward weaker connascence forms

#### 🔧 Specialized Development
- **`backend-dev`** - Specialized backend API development including REST and GraphQL
- **`mobile-dev`** - Expert React Native development across iOS and Android platforms
- **`ml-developer`** - Machine learning model development, training, and deployment
- **`cicd-engineer`** - GitHub Actions CI/CD pipeline creation and optimization
- **`api-docs`** - Expert OpenAPI/Swagger documentation creation and maintenance
- **`system-architect`** - System architecture design, patterns, and technical decisions
- **`code-analyzer`** - Advanced code quality analysis and comprehensive improvements
- **`base-template-generator`** - Foundational templates and boilerplate code creation

#### ✅ Testing & Validation (Enhanced)
- **`tdd-london-swarm`** - Mock-driven development with coupling isolation and behavioral focus
- **`production-validator`** - Production readiness validation with comprehensive fitness functions

#### 🔄 Migration & Planning
- **`migration-planner`** - Comprehensive migration planning for system conversions
- **`swarm-init`** - Swarm initialization and topology optimization specialist

## 🎯 Claude Code vs MCP Tools

### Claude Code Handles ALL:
- File operations (Read, Write, Edit, MultiEdit, Glob, Grep)
- Code generation with connascence awareness
- Bash commands and system operations
- Implementation work with coupling refactoring
- Project navigation and analysis
- TodoWrite and task management
- Git operations
- Package management
- Testing and debugging
- **NEW**: Connascence analysis and refactoring

### MCP Tools ONLY:
- Coordination and planning
- Memory management
- Neural features
- Performance tracking
- Swarm orchestration
- GitHub integration

**KEY**: MCP coordinates, Claude Code executes with connascence principles.

## 🚀 Quick Setup

```bash
# Add Claude Flow MCP server
claude mcp add claude-flow npx claude-flow@alpha mcp start
```

## MCP Tool Categories

### Coordination
`swarm_init`, `agent_spawn`, `task_orchestrate`

### Monitoring
`swarm_status`, `agent_list`, `agent_metrics`, `task_status`, `task_results`

### Memory & Neural
`memory_usage`, `neural_status`, `neural_train`, `neural_patterns`

### GitHub Integration
`github_swarm`, `repo_analyze`, `pr_enhance`, `issue_triage`, `code_review`

### System
`benchmark_run`, `features_detect`, `swarm_monitor`

## 📋 Agent Coordination Protocol (Enhanced)

### Every Agent MUST:

**1️⃣ BEFORE Work:**
```bash
npx claude-flow@alpha hooks pre-task --description "[task]" --connascence-check
npx claude-flow@alpha hooks session-restore --session-id "swarm-[id]"
```

**2️⃣ DURING Work:**
```bash
npx claude-flow@alpha hooks post-edit --file "[file]" --memory-key "swarm/[agent]/[step]" --coupling-analysis
npx claude-flow@alpha hooks notify --message "[what was done]" --connascence-metrics
```

**3️⃣ AFTER Work:**
```bash
npx claude-flow@alpha hooks post-task --task-id "[task]" --fitness-functions
npx claude-flow@alpha hooks session-end --export-metrics true --coupling-report
```

## 🚀 CLAUDE FLOW INTEGRATION PATTERNS

### 🎯 When to Use Claude Flow MCP Tools

#### Use MCP Tools For:
- **Multi-step complex tasks** requiring coordination
- **Performance-critical workflows** needing optimization
- **Cross-repository operations** and GitHub automation
- **Long-running processes** with state persistence
- **Team collaboration** with shared memory
- **Advanced analytics** and performance monitoring

#### Use Claude Code Tools For:
- **Direct file operations** (Read, Write, Edit, Glob, Grep)
- **Simple bash commands** and system operations
- **Immediate code generation** and analysis
- **Quick tests** and validation
- **Project navigation** and exploration

### ⚡ UNIFIED EXECUTION PATTERNS

#### Pattern 1: Swarm-Coordinated Development (RECOMMENDED)
```javascript
[UltraEfficient]:
  // Initialize intelligent swarm topology
  mcp__claude-flow__swarm_init {
    topology: "adaptive",
    maxAgents: 8,
    optimization: "performance",
    memory_sharing: true,
    coupling_analysis: true
  }

  // Spawn specialized agents concurrently
  mcp__claude-flow__agent_spawn { type: "system-architect", focus: "clean-architecture" }
  mcp__claude-flow__agent_spawn { type: "coder", focus: "coupling-refactor" }
  mcp__claude-flow__agent_spawn { type: "tester", focus: "behavioral-testing" }
  mcp__claude-flow__agent_spawn { type: "reviewer", focus: "connascence-violations" }

  // Orchestrate complex task with swarm coordination
  mcp__claude-flow__task_orchestrate {
    task: "Implement user authentication system with clean architecture",
    agents: ["system-architect", "coder", "tester", "reviewer"],
    phases: ["specification", "architecture", "implementation", "validation"],
    coupling_constraints: true
  }

  // Traditional Claude Code operations in parallel
  TodoWrite { todos: [
    {content: "Design authentication domain entities", status: "in_progress"},
    {content: "Implement weak connascence patterns", status: "pending"},
    {content: "Create behavioral test suite", status: "pending"},
    {content: "Validate architectural fitness", status: "pending"},
    {content: "Generate API documentation", status: "pending"}
  ]}

  // File system operations
  Bash "mkdir -p src/{domain,interfaces,infrastructure,tests}"
  Read "docs/architecture/ARCHITECTURE.md"
  Write "src/domain/user.py"
  Write "src/interfaces/auth_service.py"
  Write "tests/behavioral/test_auth_contracts.py"
```

#### Pattern 2: SPARC-Driven Workflow
```bash
# Complete SPARC workflow with batch optimization
npx claude-flow sparc batch specification,pseudocode,architecture,refinement "implement payment gateway"

# Combined with swarm coordination
mcp__claude-flow__swarm_init { topology: "hierarchical", sparc_integration: true }
mcp__claude-flow__task_orchestrate {
  task: "payment gateway implementation",
  methodology: "sparc",
  batch_processing: true
}
```

#### Pattern 3: GitHub Workflow Automation
```javascript
[GitHubSwarm]:
  // Initialize GitHub-focused swarm
  mcp__claude-flow__github_swarm {
    repository: "AIVillage",
    workflow: "pr-review-enhancement"
  }

  // Automated PR management
  mcp__claude-flow__pr_enhance {
    pr_number: 123,
    analysis: "comprehensive",
    auto_review: true,
    coupling_check: true
  }

  // Code review swarm deployment
  mcp__claude-flow__code_review {
    files: ["src/auth/", "tests/auth/"],
    focus: ["connascence-violations", "architectural-fitness", "security"]
  }
```

### ❌ WRONG vs ✅ CORRECT Patterns

#### ❌ WRONG - Sequential Operations (Slow)
```javascript
Message 1: mcp__claude-flow__swarm_init
Message 2: mcp__claude-flow__agent_spawn
Message 3: Task("single agent task")
Message 4: TodoWrite({ single todo })
Message 5: Write("single file")
// Results in: 4x slower execution, no coordination benefits
```

#### ✅ CORRECT - Concurrent Swarm Coordination (Fast)
```javascript
[SingleMessage]:
  // All swarm coordination in one message
  mcp__claude-flow__swarm_init { topology: "mesh", maxAgents: 6 }
  mcp__claude-flow__agent_spawn { type: "researcher" }
  mcp__claude-flow__agent_spawn { type: "coder" }
  mcp__claude-flow__agent_spawn { type: "tester" }

  // All task orchestration in one message
  mcp__claude-flow__task_orchestrate { task: "multi-phase implementation" }

  // All traditional tools in one message
  TodoWrite({ todos: [5+ items] })
  Task("agent1: comprehensive research...")
  Task("agent2: implementation with patterns...")
  Task("agent3: behavioral test creation...")

  // All file operations in one message
  Bash("mkdir -p {src,tests,docs}/{core,utils,integration}")
  Write("src/core/main.py")
  Write("tests/core/test_main.py")
  Write("docs/api/endpoints.md")
// Results in: 2.8-4.4x faster execution with full coordination
```

## Connascence Linting Rules

### Pre-commit Configuration
```yaml
# Enhanced with connascence rules
repos:
  - repo: local
    hooks:
      - id: connascence-checker
        name: Connascence Analysis
        entry: python scripts/check_connascence.py
        language: system
        pass_filenames: true

      - id: coupling-metrics
        name: Coupling Metrics
        entry: python scripts/coupling_metrics.py
        language: system

      - id: anti-pattern-detector
        name: Anti-pattern Detection
        entry: python scripts/detect_anti_patterns.py
        language: system
```

### Ruff Configuration (Enhanced)
```toml
[tool.ruff]
select = [
    "E", "F", "I", "UP",  # Base rules
    "C90",   # mccabe complexity
    "N",     # pep8-naming
    "ARG",   # flake8-unused-arguments
    "SIM",   # flake8-simplify
    "RET",   # flake8-return
]

# Custom connascence rules
[tool.ruff.mccabe]
max-complexity = 10  # Limit cyclomatic complexity

[tool.ruff.per-file-ignores]
# Stricter rules for domain code
"src/domain/**/*.py" = []  # No exceptions for domain logic
"src/services/**/*.py" = ["ARG001"]  # Allow unused args in interfaces
```

### Custom Metrics
```python
# Connascence metrics to track in CI
METRICS = {
    "positional_param_ratio": "% functions with >3 positional params",
    "magic_literal_density": "count of magic numbers/strings in conditionals",
    "duplicate_algorithm_count": "duplicate implementations vs shared APIs",
    "global_reference_count": "references to singletons/globals",
    "god_class_count": "classes >500 LOC or >20 methods",
    "coupling_violations": "strong connascence across module boundaries"
}
```

## Performance Benefits

- **84.8% SWE-Bench solve rate**
- **32.3% token reduction**
- **2.8-4.4x speed improvement**
- **27+ neural models**
- **NEW**: Reduced coupling debt through systematic refactoring

## Advanced Features (v2.0.0 + Connascence)

- 🚀 Automatic Topology Selection
- ⚡ Parallel Execution (2.8-4.4x speed)
- 🧠 Neural Training with Coupling Patterns
- 📊 Bottleneck Analysis + Coupling Metrics
- 🤖 Smart Auto-Spawning with Architecture Awareness
- 🛡️ Self-Healing Workflows
- 💾 Cross-Session Memory
- 🔗 GitHub Integration
- **NEW**: 🔗 Connascence Analysis & Refactoring
- **NEW**: 🏗️ Architectural Fitness Functions
- **NEW**: 📈 Coupling Debt Tracking

## Integration Tips

1. Start with connascence analysis
2. Scale agents with coupling awareness
3. Use memory for architectural context
4. Monitor coupling metrics regularly
5. Train patterns from clean architectures
6. Enable architectural fitness hooks
7. Use GitHub tools for coupling PRs

## Support

- Documentation: https://github.com/ruvnet/claude-flow
- Issues: https://github.com/ruvnet/claude-flow/issues
- Connascence: https://connascence.io

---

Remember: **Claude Flow coordinates, Claude Code creates with clean coupling!**

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.

# Connascence-Enhanced Agent Instructions

All agents MUST now follow these enhanced guidelines:

## For All Agents:
1. **Analyze coupling patterns** before making changes
2. **Prefer weak connascence** in all implementations
3. **Keep strong connascence local** (within same class/function)
4. **Reduce coupling degree** by introducing facades when >2 places co-vary
5. **Eliminate anti-patterns** systematically
6. **Use dependency injection** instead of globals
7. **Create behavioral tests** not implementation tests
8. **Document architectural decisions** in ADRs

## Agent-Specific Enhancements:

### `coder` agent:
- Implement using enums/constants instead of magic values
- Use keyword-only parameters for functions with >3 args
- Create single sources of truth for algorithms
- Apply RAII/context managers for resource management
- Prefer composition over inheritance

### `reviewer` agent:
- Flag connascence violations across module boundaries
- Check for anti-patterns (God objects, copy-paste, etc.)
- Verify behavioral vs implementation testing
- Ensure dependency injection patterns
- Review coupling metrics and architectural fitness

### `tester` agent:
- Write behavioral tests that assert contracts not internals
- Use property-based testing for algorithmic code
- Test concurrency and ordering where relevant
- Avoid duplicating business logic in tests
- Create characterization tests before refactoring

### `architect` agent:
- Design with connascence locality in mind
- Create explicit module boundaries with typed contracts
- Plan for weak coupling and high cohesion
- Document coupling decisions in ADRs
- Design fitness functions to maintain architectural quality

Remember: **Strong connascence is acceptable within a class/function; dangerous across modules/services. Always refactor toward weaker forms.**
